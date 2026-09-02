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


class AgentState(TypedDict, total=False):
    user_message: str
    project_id: Optional[str]

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
    test_result: Optional[TestExecutionResult]
    git_diff: Optional[GitDiffSummary]
    approval: Optional[ApprovalDecision]
    metrics: Optional[Dict[str, Any]]