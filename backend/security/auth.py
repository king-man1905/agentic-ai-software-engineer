import hashlib
import os
import secrets
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel, Field


class AuthMode(str, Enum):
    DEVELOPMENT = "development"
    PRODUCTION = "production"


class AuthenticationError(PermissionError):
    """Base exception for authentication failures."""
    def __init__(self, message: str, code: str = "AUTHENTICATION_ERROR", status_code: int = 401):
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code


class AuthenticationRequiredError(AuthenticationError):
    """Raised when authentication credentials are missing."""
    def __init__(self, message: str = "Authentication credentials required."):
        super().__init__(message, code="AUTHENTICATION_REQUIRED", status_code=401)


class AuthenticationInvalidError(AuthenticationError):
    """Raised when credentials are malformed, invalid, or forged."""
    def __init__(self, message: str = "Authentication credentials invalid."):
        super().__init__(message, code="AUTHENTICATION_INVALID", status_code=401)


class AuthenticationExpiredError(AuthenticationError):
    """Raised when credentials have expired."""
    def __init__(self, message: str = "Authentication credentials have expired."):
        super().__init__(message, code="AUTHENTICATION_EXPIRED", status_code=401)


class TenantAccessDeniedError(AuthenticationError):
    """Raised when cross-tenant access violation occurs."""
    def __init__(self, message: str = "Tenant access denied."):
        super().__init__(message, code="TENANT_ACCESS_DENIED", status_code=403)


class RepositoryAccessDeniedError(AuthenticationError):
    """Raised when repository access is unauthorized or outside allowed boundaries."""
    def __init__(self, message: str = "Repository access denied."):
        super().__init__(message, code="REPO_ACCESS_DENIED", status_code=403)


class ApiKeyRecord(BaseModel):
    """
    Metadata for a hashed API key. Raw key is never stored.
    """
    key_id: str
    key_hash: str
    key_prefix: str
    user_id: str
    organization_id: str
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    expires_at: Optional[str] = None
    is_revoked: bool = False
    revoked_at: Optional[str] = None
    name: str = "default"


class AuthManager:
    """
    Manages API keys with cryptographic hashing, constant-time comparison,
    rotation, revocation, and expiration tracking.
    """

    def __init__(self) -> None:
        self._keys: Dict[str, ApiKeyRecord] = {}
        self._hash_to_key_id: Dict[str, str] = {}

    @staticmethod
    def hash_key(raw_key: str) -> str:
        """Computes deterministic SHA-256 digest of the raw key."""
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    def create_api_key(
        self,
        user_id: str,
        organization_id: str,
        name: str = "default",
        expires_in_days: Optional[int] = None,
        custom_raw_key: Optional[str] = None,
    ) -> Tuple[str, ApiKeyRecord]:
        """
        Creates a new API key. Returns the raw key (which must be presented to user once)
        and the persisted ApiKeyRecord.
        """
        raw_key = custom_raw_key or f"ak_live_{secrets.token_urlsafe(32)}"
        key_id = f"key_{secrets.token_hex(8)}"
        key_hash = self.hash_key(raw_key)
        key_prefix = raw_key[:12] + "..."

        expires_at = None
        if expires_in_days is not None:
            from datetime import timedelta
            exp = datetime.now(timezone.utc) + timedelta(days=expires_in_days)
            expires_at = exp.isoformat()

        record = ApiKeyRecord(
            key_id=key_id,
            key_hash=key_hash,
            key_prefix=key_prefix,
            user_id=user_id,
            organization_id=organization_id,
            expires_at=expires_at,
            name=name,
        )

        self._keys[key_id] = record
        self._hash_to_key_id[key_hash] = key_id
        return raw_key, record

    def rotate_api_key(
        self,
        key_id: str,
        expires_in_days: Optional[int] = None,
    ) -> Tuple[str, ApiKeyRecord]:
        """
        Rotates an API key: immediately revokes the existing key and issues a new one
        bound to the same user and organization.
        """
        old_record = self._keys.get(key_id)
        if not old_record:
            raise ValueError(f"Key '{key_id}' not found.")

        # Revoke old key
        self.revoke_api_key(key_id)

        # Issue new key
        return self.create_api_key(
            user_id=old_record.user_id,
            organization_id=old_record.organization_id,
            name=f"{old_record.name}_rotated",
            expires_in_days=expires_in_days,
        )

    def revoke_api_key(self, key_id: str) -> bool:
        """Marks an API key as revoked."""
        record = self._keys.get(key_id)
        if not record:
            return False
        record.is_revoked = True
        record.revoked_at = datetime.now(timezone.utc).isoformat()
        return True

    def authenticate_api_key(self, raw_key: str) -> ApiKeyRecord:
        """
        Authenticates a raw API key using constant-time hash verification.
        Validates revocation and expiration.
        """
        if not raw_key or not raw_key.strip():
            raise AuthenticationInvalidError("API key cannot be empty.")

        target_hash = self.hash_key(raw_key.strip())
        matched_record: Optional[ApiKeyRecord] = None

        # Constant-time comparison across stored hashes to avoid timing attacks
        for key_hash, key_id in self._hash_to_key_id.items():
            if secrets.compare_digest(target_hash, key_hash):
                matched_record = self._keys.get(key_id)
                break

        if not matched_record:
            raise AuthenticationInvalidError("Invalid API key.")

        if matched_record.is_revoked:
            raise AuthenticationInvalidError("API key has been revoked.")

        if matched_record.expires_at:
            exp = datetime.fromisoformat(matched_record.expires_at)
            if datetime.now(timezone.utc) > exp:
                raise AuthenticationExpiredError("API key has expired.")

        return matched_record

    def get_key_record(self, key_id: str) -> Optional[ApiKeyRecord]:
        return self._keys.get(key_id)

    def list_user_keys(self, user_id: str) -> List[ApiKeyRecord]:
        return [k for k in self._keys.values() if k.user_id == user_id]

    def clear(self) -> None:
        self._keys.clear()
        self._hash_to_key_id.clear()


auth_manager = AuthManager()
