import { GitDiffSummary } from './diff';
import { QAResult } from './qa';
import { PolicyEvaluationResult } from './policy';
import { RunRecord, TelemetryEvent } from './telemetry';

export type { RunRecord, TelemetryEvent };

export interface CreateRunRequest {
  user_message: string;
  project_id?: string | null;
  organization_id?: string | null;
  repository_id?: string | null;
  metadata?: Record<string, any> | null;
}

export interface RunStatusResponse {
  run_id: string;
  status:
    | 'CREATED'
    | 'RUNNING'
    | 'WAITING_APPROVAL'
    | 'REVISING'
    | 'COMMITTING'
    | 'PUBLISHING'
    | 'COMPLETED'
    | 'FAILED'
    | 'BLOCKED'
    | 'CANCEL_REQUESTED'
    | 'CANCELLED'
    | 'CANCELLING'
    | 'STUCK'
    | string;
  current_node?: string | null;
  git_diff?: GitDiffSummary | null;
  qa_result?: QAResult | null;
  policy_result?: PolicyEvaluationResult | null;
  error_summary?: string | null;
  message?: string | null;
}

export interface CancelRunRequest {
  reason?: string | null;
}

export interface ResumeRunRequest {
  approved: boolean;
  reviewer?: string | null;
  reviewer_role?: string | null;
  rejection_reason?: string | null;
  patch_hash?: string | null;
  organization_id?: string | null;
}

export interface AuditEventView {
  event_id: string;
  organization_id: string;
  user_id: string;
  action: string;
  timestamp: string;
  resource_type: string;
  resource_id: string;
  details: Record<string, any>;
  event_hash: string;
  previous_hash: string;
}

export interface CreateApiKeyRequest {
  name?: string;
  expires_in_days?: number | null;
}

export interface ApiKeyResponse {
  key_id: string;
  key_prefix: string;
  raw_key?: string | null;
  user_id: string;
  organization_id: string;
  created_at: string;
  expires_at?: string | null;
  is_revoked: boolean;
  name: string;
}

export interface PublishPRRequest {
  repo_full_name: string;
  title?: string | null;
  base_branch?: string;
  draft?: boolean;
}

export interface PublishPRResponse {
  pr_number: number;
  pr_url: string;
  head_branch: string;
  base_branch: string;
  is_draft: boolean;
  status: string;
}

export interface RunListResponse {
  runs: RunRecord[];
  total: number;
  limit: number;
  offset: number;
}

export interface RunEventsResponse {
  run_id: string;
  events: TelemetryEvent[];
}

export interface HealthCheckResponse {
  status: string;
  service: string;
  version: string;
}

export interface ReadinessCheckResponse {
  is_ready: boolean;
  timestamp: string;
  components?: Record<string, any>;
}

export interface ApiError {
  status: number;
  code?: string;
  message: string;
  detail?: any;
}
