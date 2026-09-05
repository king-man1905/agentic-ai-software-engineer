import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, computed_field


class FailureCategory(str, Enum):
    """
    Standardized, normalized failure categories across all platform components.
    """
    AUTHENTICATION_FAILURE = "AUTHENTICATION_FAILURE"
    TENANT_ACCESS_FAILURE = "TENANT_ACCESS_FAILURE"
    POLICY_BLOCK = "POLICY_BLOCK"
    RAG_INSUFFICIENT_CONTEXT = "RAG_INSUFFICIENT_CONTEXT"
    AST_FAILURE = "AST_FAILURE"
    TEST_FAILURE = "TEST_FAILURE"
    SECURITY_FAILURE = "SECURITY_FAILURE"
    REVISION_EXHAUSTED = "REVISION_EXHAUSTED"
    SANDBOX_FAILURE = "SANDBOX_FAILURE"
    COMMIT_FAILURE = "COMMIT_FAILURE"
    GITHUB_AUTH_FAILURE = "GITHUB_AUTH_FAILURE"
    GITHUB_PERMISSION_DENIED = "GITHUB_PERMISSION_DENIED"
    GITHUB_RATE_LIMIT = "GITHUB_RATE_LIMIT"
    GITHUB_BRANCH_CONFLICT = "GITHUB_BRANCH_CONFLICT"
    PR_CREATION_FAILURE = "PR_CREATION_FAILURE"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class TelemetryEventType(str, Enum):
    """
    Deterministic engineering execution lifecycle events.
    """
    RUN_CREATED = "RUN_CREATED"
    RUN_STARTED = "RUN_STARTED"
    ROUTER_DECISION = "ROUTER_DECISION"
    ROUTING_COMPLETED = "ROUTING_COMPLETED"
    PLAN_CREATED = "PLAN_CREATED"
    PLANNER_COMPLETE = "PLANNER_COMPLETE"
    KNOWLEDGE_RETRIEVED = "KNOWLEDGE_RETRIEVED"
    RAG_COMPLETED = "RAG_COMPLETED"
    DEVELOPMENT_COMPLETED = "DEVELOPMENT_COMPLETED"
    NODE_COMPLETE = "NODE_COMPLETE"
    QA_STARTED = "QA_STARTED"
    QA_COMPLETED = "QA_COMPLETED"
    REVISION_STARTED = "REVISION_STARTED"
    REVISION_COMPLETED = "REVISION_COMPLETED"
    POLICY_EVALUATED = "POLICY_EVALUATED"
    APPROVAL_REQUESTED = "APPROVAL_REQUESTED"
    APPROVAL_GRANTED = "APPROVAL_GRANTED"
    APPROVAL_DENIED = "APPROVAL_DENIED"
    COMMIT_STARTED = "COMMIT_STARTED"
    COMMIT_COMPLETED = "COMMIT_COMPLETED"
    GITHUB_OPERATION_STARTED = "GITHUB_OPERATION_STARTED"
    GITHUB_OPERATION_COMPLETED = "GITHUB_OPERATION_COMPLETED"
    GITHUB_PR_PUBLISHED = "GITHUB_PR_PUBLISHED"
    PR_CREATED = "PR_CREATED"
    RUN_COMPLETED = "RUN_COMPLETED"
    RUN_FAILED = "RUN_FAILED"


class TelemetryEvent(BaseModel):
    """
    A single immutable, ordered lifecycle event during an engineering run.
    """
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    run_id: str
    organization_id: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    event_type: TelemetryEventType
    duration_ms: Optional[float] = None
    safe_metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def node(self) -> Optional[str]:
        return self.safe_metadata.get("node")

    @property
    def details(self) -> Dict[str, Any]:
        return self.safe_metadata


class RunRecord(BaseModel):
    """
    Comprehensive, tenant-scoped run telemetry record.
    Never stores secrets, tokens, passwords, or raw credentials.
    """
    run_id: str
    organization_id: str
    user_id: Optional[str] = None
    repository: Optional[str] = None
    branch: Optional[str] = None
    user_message: Optional[str] = None

    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    duration_ms: Optional[float] = None

    status: str = "QUEUED"  # QUEUED, RUNNING, WAITING_APPROVAL, REVISING, COMMITTING, PUBLISHING, COMPLETED, FAILED, BLOCKED, CANCELLED
    failure_category: Optional[FailureCategory] = None
    safe_failure_message: Optional[str] = None

    provider: Optional[str] = None
    model: Optional[str] = None

    revision_count: int = 0

    qa_status: Optional[str] = None  # PASS, FAIL, None
    qa_summary: Optional[str] = None

    rag_status: Optional[str] = None  # SUFFICIENT, INSUFFICIENT, SKIPPED, None
    rag_quality_summary: Optional[str] = None

    policy_decision: Optional[str] = None  # PASS, BLOCK, WARN, None
    risk_score: Optional[float] = None

    approval_required: bool = False
    approval_status: Optional[str] = None  # PENDING, APPROVED, REJECTED, COMMITTED, None
    approval_latency_ms: Optional[float] = None

    patch_hash: Optional[str] = None
    commit_status: Optional[str] = None

    github_status: Optional[str] = None
    pr_status: Optional[str] = None  # DRAFT, PUBLISHED, FAILED, None
    pr_url: Optional[str] = None
    pr_number: Optional[int] = None

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    estimated_input_cost: Optional[float] = None
    estimated_output_cost: Optional[float] = None
    estimated_total_cost: Optional[float] = None
    currency: str = "USD"
    approval_reviewer: Optional[str] = None
    approval_decision: Optional[str] = None

    @property
    def tenant_id(self) -> str:
        return self.organization_id

    @property
    def ended_at(self) -> Optional[str]:
        return self.completed_at

    @computed_field  # type: ignore[prop-decorator]
    @property
    def project_id(self) -> Optional[str]:
        return self.repository

    @property
    def model_name(self) -> Optional[str]:
        return self.model

    @property
    def total_cost_usd(self) -> Optional[float]:
        return self.estimated_total_cost

    @property
    def duration_seconds(self) -> Optional[float]:
        return (self.duration_ms / 1000.0) if self.duration_ms is not None else None

    @property
    def prompt_tokens(self) -> int:
        return self.input_tokens

    @property
    def completion_tokens(self) -> int:
        return self.output_tokens

    @property
    def approval_latency_seconds(self) -> Optional[float]:
        return (self.approval_latency_ms / 1000.0) if self.approval_latency_ms is not None else None

    @property
    def qa_passed(self) -> Optional[bool]:
        if self.qa_status is None:
            return None
        return self.qa_status.upper() == "PASS"

    @property
    def pr_published(self) -> bool:
        return self.pr_status == "PUBLISHED" or self.github_status == "SUCCESS"


# =============================================================================
# Analytics Aggregation DTOs
# =============================================================================

class AnalyticsOverview(BaseModel):
    """
    High-level operational metrics for a tenant.
    """
    total_runs: int = 0
    successful_runs: int = 0
    failed_runs: int = 0
    waiting_approval_runs: int = 0
    avg_duration_ms: float = 0.0
    total_estimated_cost_usd: float = 0.0
    total_tokens: int = 0

    @property
    def total_cost_usd(self) -> float:
        return self.total_estimated_cost_usd

    @property
    def success_rate(self) -> float:
        return (self.successful_runs / self.total_runs) if self.total_runs > 0 else 0.0

    @property
    def approval_required_rate(self) -> float:
        return (self.waiting_approval_runs / self.total_runs) if self.total_runs > 0 else 0.0

    @property
    def avg_duration_seconds(self) -> float:
        return (self.avg_duration_ms / 1000.0) if self.avg_duration_ms else 0.0


class RAGAnalytics(BaseModel):
    """
    Aggregated Code RAG and retrieval quality metrics.
    """
    total_retrievals: int = 0
    retrieval_success_rate: float = 0.0
    insufficient_context_count: int = 0
    sufficient_context_count: int = 0


class QualityAnalytics(BaseModel):
    """
    Aggregated QA, security, and revision quality metrics.
    """
    total_evaluated: int = 0
    qa_pass_rate: float = 0.0
    test_failure_count: int = 0
    security_failure_count: int = 0
    avg_revisions_per_run: float = 0.0
    total_revisions: int = 0
    rag_analytics: Optional[RAGAnalytics] = None

    @property
    def avg_revisions(self) -> float:
        return self.avg_revisions_per_run


class ProviderUsage(BaseModel):
    provider: str
    request_count: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0


class ModelUsage(BaseModel):
    model: str
    request_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0

    @property
    def model_name(self) -> str:
        return self.model


class ModelAnalytics(BaseModel):
    """
    Aggregated model, token consumption, and cost telemetry.
    """
    by_provider: Dict[str, ProviderUsage] = Field(default_factory=dict)
    by_model: Dict[str, ModelUsage] = Field(default_factory=dict)
    total_cost_usd: float = 0.0

    @property
    def models(self) -> Dict[str, ModelUsage]:
        return self.by_model

    @property
    def providers(self) -> Dict[str, ProviderUsage]:
        return self.by_provider


class FailureAnalytics(BaseModel):
    """
    Distribution of failure categories across runs.
    """
    by_category: Dict[str, int] = Field(default_factory=dict)
    total_failures: int = 0

    @property
    def categories(self) -> Dict[str, int]:
        return self.by_category


# =============================================================================
# Evaluation Benchmark Models
# =============================================================================

class EvaluationTask(BaseModel):
    task_id: str
    repository: str = "default-org/core-lib"
    task_description: str = ""
    expected_outcome: str = "PASS"
    timeout_seconds: int = 60
    category: Optional[str] = None
    difficulty: Optional[str] = None
    prompt: Optional[str] = None
    expected_files: List[str] = Field(default_factory=list)
    max_revisions_allowed: int = 3
    target_repo: Optional[str] = None


class EvaluationResult(BaseModel):
    task_id: str
    repository: str = "default-org/core-lib"
    success: bool = True
    first_pass_success: bool = True
    test_passed: bool = True
    qa_status: str = "PASS"
    revisions: int = 0
    duration_ms: float = 0.0
    model: str = "gpt-4o"
    provider: str = "openai"
    cost_usd: float = 0.0
    error: Optional[str] = None

    run_id: Optional[str] = None
    tenant_id: Optional[str] = None
    benchmark_name: Optional[str] = None
    passed: Optional[bool] = None
    failure_reason: Optional[str] = None
    duration_seconds: Optional[float] = None
    total_tokens: Optional[int] = None


class EvaluationSummary(BaseModel):
    evaluation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    organization_id: str = "default-org"
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    task_count: int = 0
    successful_tasks: int = 0
    task_success_rate: float = 0.0
    test_pass_rate: float = 0.0
    first_pass_success_rate: float = 0.0
    avg_revisions: float = 0.0
    avg_duration_ms: float = 0.0
    total_cost_usd: float = 0.0
    results: List[EvaluationResult] = Field(default_factory=list)

    @property
    def total_tasks(self) -> int:
        return self.task_count

    @property
    def passed_tasks(self) -> int:
        return self.successful_tasks

    @property
    def pass_rate(self) -> float:
        return self.task_success_rate
