export type TelemetryEventType =
  | 'RUN_CREATED'
  | 'RUN_STARTED'
  | 'ROUTER_DECISION'
  | 'ROUTING_COMPLETED'
  | 'PLAN_CREATED'
  | 'PLANNER_COMPLETE'
  | 'KNOWLEDGE_RETRIEVED'
  | 'RAG_COMPLETED'
  | 'DEVELOPMENT_COMPLETED'
  | 'NODE_COMPLETE'
  | 'QA_STARTED'
  | 'QA_COMPLETED'
  | 'REVISION_STARTED'
  | 'REVISION_COMPLETED'
  | 'POLICY_EVALUATED'
  | 'APPROVAL_REQUESTED'
  | 'APPROVAL_GRANTED'
  | 'APPROVAL_DENIED'
  | 'COMMIT_STARTED'
  | 'COMMIT_COMPLETED'
  | 'GITHUB_OPERATION_STARTED'
  | 'GITHUB_OPERATION_COMPLETED'
  | 'GITHUB_PR_PUBLISHED'
  | 'PR_CREATED'
  | 'PROVIDER_FALLBACK'
  | 'WORKSPACE_LOCK_ACQUIRED'
  | 'WORKSPACE_LOCK_RELEASED'
  | 'WORKSPACE_LOCK_TIMEOUT'
  | 'IDEMPOTENCY_REPLAY'
  | 'IDEMPOTENCY_CONFLICT'
  | 'PR_RECONCILIATION'
  | 'RUN_CANCEL_REQUESTED'
  | 'RUN_CANCELLED'
  | 'RUN_STUCK'
  | 'SANDBOX_CANCELLED'
  | 'RUN_COMPLETED'
  | 'RUN_FAILED'
  | 'SHUTDOWN_STARTED'
  | 'SHUTDOWN_COMPLETED'
  | 'SHUTDOWN_INTERRUPTED';

export interface TelemetryEvent {
  event_id: string;
  run_id: string;
  organization_id: string;
  timestamp: string;
  event_type: TelemetryEventType | string;
  duration_ms?: number | null;
  safe_metadata: Record<string, any>;
  node?: string | null;
  tokens?: number | null;
  cost_usd?: number | null;
  details?: Record<string, any>;
}

export interface RunRecord {
  run_id: string;
  organization_id: string;
  user_id?: string | null;
  repository?: string | null;
  branch?: string | null;
  user_message?: string | null;
  created_at: string;
  started_at?: string | null;
  completed_at?: string | null;
  duration_ms?: number | null;
  status: string;
  failure_category?: string | null;
  safe_failure_message?: string | null;
  provider?: string | null;
  model?: string | null;
  revision_count: number;
  qa_status?: string | null;
  qa_summary?: string | null;
  rag_status?: string | null;
  rag_quality_summary?: string | null;
  policy_decision?: string | null;
  risk_score?: number | null;
  approval_required: boolean;
  approval_status?: string | null;
  approval_latency_ms?: number | null;
  patch_hash?: string | null;
  commit_status?: string | null;
  github_status?: string | null;
  pr_status?: string | null;
  pr_url?: string | null;
  pr_number?: number | null;
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  estimated_input_cost?: number | null;
  estimated_output_cost?: number | null;
  estimated_total_cost?: number | null;
  currency: string;
  approval_reviewer?: string | null;
  approval_decision?: string | null;
  cancel_requested?: boolean;
  cancellation_requested_at?: string | null;
  cancellation_requested_by?: string | null;
  cancellation_reason?: string | null;
  cancelled_at?: string | null;
  last_activity_at?: string | null;
  current_phase?: string | null;
  stuck_at?: string | null;
  project_id?: string | null;
  total_cost_usd?: number | null;
  duration_seconds?: number | null;
  qa_passed?: boolean | null;
  pr_published?: boolean;
}

export interface AnalyticsOverview {
  total_runs: number;
  successful_runs: number;
  failed_runs: number;
  waiting_approval_runs: number;
  avg_duration_ms: number;
  total_estimated_cost_usd: number;
  total_tokens: number;
  total_cost_usd?: number;
  success_rate?: number;
  approval_required_rate?: number;
  avg_duration_seconds?: number;
}

export interface RAGAnalytics {
  total_retrievals: number;
  retrieval_success_rate: number;
  insufficient_context_count: number;
  sufficient_context_count: number;
}

export interface QualityAnalytics {
  total_evaluated: number;
  qa_pass_rate: number;
  test_failure_count: number;
  security_failure_count: number;
  avg_revisions_per_run: number;
  total_revisions: number;
  rag_analytics?: RAGAnalytics | null;
  avg_revisions?: number;
}

export interface ProviderUsage {
  provider: string;
  request_count: number;
  total_tokens: number;
  estimated_cost_usd: number;
}

export interface ModelUsage {
  model: string;
  request_count: number;
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  estimated_cost_usd: number;
  model_name?: string;
  run_count?: number;
}

export interface ModelAnalytics {
  by_provider: Record<string, ProviderUsage>;
  by_model: Record<string, ModelUsage>;
  total_cost_usd: number;
  models?: Record<string, ModelUsage>;
  providers?: Record<string, ProviderUsage>;
}

export interface FailureAnalytics {
  by_category: Record<string, number>;
  total_failures: number;
  categories?: Record<string, number>;
}
