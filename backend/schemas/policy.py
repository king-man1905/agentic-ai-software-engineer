from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class PolicyDecision(str, Enum):
    """Deterministic organization policy decisions."""
    ALLOW = "ALLOW"
    REVIEW = "REVIEW"
    BLOCK = "BLOCK"


class PolicyViolation(BaseModel):
    """Represents a specific policy rule violation."""
    rule: str = Field(description="Identifier or name of the policy rule violated.")
    message: str = Field(description="Human-readable explanation of the violation.")
    severity: str = Field(
        default="BLOCK",
        description="Severity level of the violation: BLOCK or WARNING.",
    )
    target: Optional[str] = Field(
        default=None,
        description="Target resource, file path, branch, or metric causing the violation.",
    )


class PolicyConfig(BaseModel):
    """
    Enterprise organization policy configuration with safe defaults.
    """
    allowed_repositories: Optional[List[str]] = Field(
        default=None,
        description="List of repository names or glob patterns allowed. None allows any repository.",
    )
    allowed_branches: Optional[List[str]] = Field(
        default=None,
        description="List of target feature branch prefixes or globs allowed. None allows any valid branch.",
    )
    protected_branches: List[str] = Field(
        default_factory=lambda: ["main", "master", "release/*", "production", "staging"],
        description="Branches that must never be directly targeted or mutated.",
    )
    protected_paths: List[str] = Field(
        default_factory=lambda: [
            ".env*",
            "secrets/**",
            ".github/workflows/**",
            "auth/**",
            "security/**",
            "database/migrations/**",
        ],
        description="Protected file glob patterns requiring elevated review or blocking.",
    )
    max_files_changed: int = Field(
        default=10,
        description="Maximum number of files allowed in a single changeset.",
    )
    max_lines_added: int = Field(
        default=500,
        description="Maximum lines added across all files in a changeset.",
    )
    max_lines_deleted: int = Field(
        default=200,
        description="Maximum lines deleted across all files in a changeset.",
    )
    max_revisions: int = Field(
        default=3,
        description="Maximum self-correction revision cycles allowed before blocking.",
    )
    required_quality_checks: List[str] = Field(
        default_factory=lambda: ["AST", "PYTEST", "SECURITY"],
        description="Quality checks that must succeed before changes are permitted.",
    )
    risk_thresholds: Dict[str, str] = Field(
        default_factory=lambda: {
            "LOW": "ALLOW",
            "MEDIUM": "REVIEW",
            "HIGH": "REVIEW",
            "CRITICAL": "BLOCK",
        },
        description="Mapping from assessed risk level to base policy decision.",
    )
    allowed_models: Optional[List[str]] = Field(
        default=None,
        description="Allowlist of permitted LLM model identifiers.",
    )
    allow_network: bool = Field(
        default=False,
        description="Whether sandbox test execution is permitted external network access.",
    )
    allow_dependency_changes: bool = Field(
        default=False,
        description="Whether changes to dependency manifests (requirements.txt, package.json) are allowed.",
    )
    allow_config_changes: bool = Field(
        default=False,
        description="Whether changes to infrastructure configuration files are allowed.",
    )
    allow_ci_changes: bool = Field(
        default=False,
        description="Whether changes to CI/CD workflows (.github, .gitlab-ci) are allowed.",
    )
    policy_version: str = Field(
        default="1.0.0",
        description="Version identifier of this organization policy.",
    )


class PolicyEvaluationResult(BaseModel):
    """
    Structured outcome of deterministic policy evaluation.
    """
    decision: PolicyDecision = Field(
        description="Final deterministic policy verdict: ALLOW, REVIEW, or BLOCK."
    )
    violations: List[PolicyViolation] = Field(
        default_factory=list,
        description="List of blocking policy violations.",
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Advisory warnings or review triggers.",
    )
    checks: Dict[str, str] = Field(
        default_factory=dict,
        description="Per-category check verdicts (e.g., {'repository': 'PASS', 'protected_paths': 'PASS'}).",
    )
    policy_version: str = Field(
        default="1.0.0",
        description="Version of the policy applied during evaluation.",
    )
    evaluated_at: str = Field(
        description="ISO-8601 timestamp of evaluation.",
    )


class PolicyTelemetry(BaseModel):
    """
    Observability telemetry payload for policy evaluation.
    """
    policy_version: str
    decision: str
    violations: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    risk_score: str
    risk_level: str
    repository: Optional[str] = None
    branch: Optional[str] = None
    changed_files: List[str] = Field(default_factory=list)
    patch_size: Dict[str, int] = Field(default_factory=dict)
    required_checks: List[str] = Field(default_factory=list)
    failed_checks: List[str] = Field(default_factory=list)
    timestamp: str
