import hashlib
import json
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field


class AuditAction(str, Enum):
    """
    Standardized audit actions across tenant operations and security lifecycle.
    """
    AUTHENTICATION_SUCCESS = "AUTHENTICATION_SUCCESS"
    AUTHENTICATION_FAILURE = "AUTHENTICATION_FAILURE"
    RUN_INITIATED = "RUN_INITIATED"
    RUN_RESUMED = "RUN_RESUMED"
    RUN_COMPLETED = "RUN_COMPLETED"
    APPROVAL_REQUESTED = "APPROVAL_REQUESTED"
    APPROVAL_GRANTED = "APPROVAL_GRANTED"
    APPROVAL_DENIED = "APPROVAL_DENIED"
    POLICY_EVALUATED = "POLICY_EVALUATED"
    POLICY_UPDATED = "POLICY_UPDATED"
    SECURITY_ALERT = "SECURITY_ALERT"
    INDEX_BUILT = "INDEX_BUILT"
    REPOSITORY_CONNECTED = "REPOSITORY_CONNECTED"
    GITHUB_OPERATION = "GITHUB_OPERATION"
    GITHUB_OPERATION_FAILED = "GITHUB_OPERATION_FAILED"
    PR_CREATED = "PR_CREATED"


SENSITIVE_FIELD_SUBSTRINGS = ["token", "secret", "key", "password", "auth", "credential"]


def sanitize_audit_details(details: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively strips or redacts sensitive keys to ensure raw tokens,
    API keys, and passwords never enter audit logs or hash digests.
    """
    sanitized = {}
    for k, v in details.items():
        k_lower = str(k).lower()
        if any(sub in k_lower for sub in SENSITIVE_FIELD_SUBSTRINGS):
            if isinstance(v, str) and len(v) > 8 and not v.startswith("ak_"):
                sanitized[k] = "[REDACTED]"
            elif k_lower in ("token", "api_key", "password", "secret", "github_token"):
                sanitized[k] = "[REDACTED]"
            else:
                sanitized[k] = v
        elif isinstance(v, dict):
            sanitized[k] = sanitize_audit_details(v)
        else:
            sanitized[k] = v
    return sanitized


class AuditEvent(BaseModel):
    """
    Append-only, tamper-evident audit log event secured with SHA-256 hash chaining.
    """
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    organization_id: str
    user_id: str
    action: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    resource_type: str
    resource_id: str
    details: Dict[str, Any] = Field(default_factory=dict)
    previous_hash: str = Field(default="0" * 64)
    event_hash: str = Field(default="")

    def compute_hash(self) -> str:
        payload = {
            "event_id": self.event_id,
            "organization_id": self.organization_id,
            "user_id": self.user_id,
            "action": self.action,
            "timestamp": self.timestamp,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "details": self.details,
            "previous_hash": self.previous_hash,
        }
        raw_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw_json.encode("utf-8")).hexdigest()


class AuditLogger:
    """
    In-memory append-only audit logger maintaining per-tenant cryptographic hash chains.
    """

    def __init__(self) -> None:
        self._events: List[AuditEvent] = []
        self._last_hash_by_org: Dict[str, str] = {}

    def log(
        self,
        organization_id: str,
        user_id: str,
        action: AuditAction | str,
        resource_type: str,
        resource_id: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> AuditEvent:
        action_str = action.value if isinstance(action, AuditAction) else str(action)
        prev_hash = self._last_hash_by_org.get(organization_id, "0" * 64)

        clean_details = sanitize_audit_details(details or {})

        event = AuditEvent(
            organization_id=organization_id,
            user_id=user_id,
            action=action_str,
            resource_type=resource_type,
            resource_id=resource_id,
            details=clean_details,
            previous_hash=prev_hash,
        )
        event.event_hash = event.compute_hash()

        self._events.append(event)
        self._last_hash_by_org[organization_id] = event.event_hash
        return event

    def get_events(self, organization_id: Optional[str] = None) -> List[AuditEvent]:
        if organization_id:
            return [e for e in self._events if e.organization_id == organization_id]
        return list(self._events)

    def verify_integrity(self, organization_id: str) -> Tuple[bool, Optional[str]]:
        """
        Verifies the cryptographic hash chain for a specific tenant organization.
        """
        org_events = [e for e in self._events if e.organization_id == organization_id]
        expected_prev = "0" * 64

        for idx, event in enumerate(org_events):
            if event.previous_hash != expected_prev:
                return (
                    False,
                    f"Chain broken at event {event.event_id} (index {idx}): expected prev {expected_prev}, got {event.previous_hash}",
                )
            recomputed = event.compute_hash()
            if event.event_hash != recomputed:
                return (
                    False,
                    f"Tampered event {event.event_id} (index {idx}): recorded hash {event.event_hash} != recomputed {recomputed}",
                )
            expected_prev = event.event_hash

        return True, None

    def clear(self) -> None:
        self._events.clear()
        self._last_hash_by_org.clear()


audit_logger = AuditLogger()
