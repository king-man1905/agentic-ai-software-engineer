export type QualityCheckStatus = 'PASS' | 'FAIL' | 'SKIPPED' | 'ERROR';

export interface QualityCheck {
  name: string;
  status: QualityCheckStatus | string;
  exit_code: number;
  duration_ms: number;
  stdout_summary: string;
  stderr_summary: string;
  reason?: string | null;
  duration_seconds?: number | null;
}

export interface QAIssue {
  file_path: string;
  issue: string;
  severity: 'LOW' | 'MEDIUM' | 'HIGH' | string;
}

export interface QAResult {
  status: 'PASS' | 'FAIL' | 'NEEDS_REVIEW' | string;
  confidence: number;
  regression_risk: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL' | string;
  checks: QualityCheck[];
  failure_category?: string | null;
  issues: QAIssue[];
  test_cases: string[];
  summary: string;
  advisory_issues?: string[];
}
