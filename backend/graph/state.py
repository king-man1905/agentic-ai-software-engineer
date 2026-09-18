from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict

from backend.schemas.routing import RoutingDecision
from backend.schemas.planning import ExecutionPlan
from backend.schemas.knowledge import KnowledgeAnswer
from backend.schemas.developer import DeveloperResult
from backend.schemas.qa import QAResult
from backend.indexer.models import CodeChunk
from backend.developer.models import FilePatch
from backend.sandbox.models import TestExecutionResult
from backend.revision.models import RevisionHistory
from backend.vcs.models import GitDiffSummary, ApprovalDecision
from backend.schemas.rag import RetrievalEvaluation, RAGTelemetry
from backend.schemas.policy import (
    PolicyConfig,
    PolicyEvaluationResult,
    PolicyTelemetry,
)


class AgentState(TypedDict, total=False):
    user_message: str
    project_id: Optional[str]
    organization_id: Optional[str]
    user_id: Optional[str]
    repository_id: Optional[str]
    run_id: Optional[str]

    routing: RoutingDecision
    plan: ExecutionPlan
    knowledge: KnowledgeAnswer
    developer_result: DeveloperResult
    qa_result: QAResult

    revision_count: int
    revision_history: Optional[RevisionHistory]
    approval_status: str
    repo_context: Optional[List[CodeChunk]]
    generated_patches: Optional[List[FilePatch]]
    # Per-file content immediately before developer_node/revision_node's own
    # disk write of that file's patch - see their docstrings and
    # QualityPipeline.check_patch_scope, which needs the true pre-patch
    # content and cannot get it from a fresh disk read once this node has
    # already overwritten the file with the patched result.
    pre_patch_snapshots: Optional[Dict[str, str]]
    # Per-file content as it existed before this RUN ever touched it -
    # captured once (developer_node's first read/write of a file) and never
    # overwritten or dropped for any later revision cycle, unlike
    # pre_patch_snapshots above (which intentionally reflects only the
    # immediate pre-write state for whichever write happened most
    # recently). QualityPipeline.check_patch_scope's destructiveness
    # measurement (detect_unsafe_additive_rewrite) must always compare a
    # candidate's applied content against this true, run-lifetime original -
    # comparing it against an intermediate (possibly also-destructive,
    # already-rejected) revision attempt's content instead can hide a
    # whole-file-replacement fabrication that merely resembles the earlier
    # rejected one.
    true_original_snapshots: Optional[Dict[str, str]]
    test_result: Optional[TestExecutionResult]
    git_diff: Optional[GitDiffSummary]
    approval: Optional[ApprovalDecision]
    patch_hash: Optional[str]
    metrics: Optional[Dict[str, Any]]

    # Phase 3 RAG & Retrieval Quality Layer
    rag_evaluation: Optional[RetrievalEvaluation]
    rag_telemetry: Optional[RAGTelemetry]
    rag_status: Optional[str]

    # Phase 4 Organization Policy Engine
    policy_config: Optional[PolicyConfig]
    policy_result: Optional[PolicyEvaluationResult]
    policy_telemetry: Optional[PolicyTelemetry]