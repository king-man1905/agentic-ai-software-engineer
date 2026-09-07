export interface GitDiffSummary {
  branch_name: string;
  files_changed: string[];
  lines_added: number;
  lines_deleted: number;
  unified_diff: string;
  risk_score: 'LOW' | 'MEDIUM' | 'HIGH' | string;
  risk_reasons: string[];
  patch_hash: string;
  is_no_op?: boolean;
}

export interface ApprovalDecision {
  approved: boolean;
  reviewer?: string | null;
  rejection_reason?: string | null;
  patch_hash?: string | null;
  timestamp?: string | null;
  reviewer_role?: string | null;
  user_id?: string | null;
}
