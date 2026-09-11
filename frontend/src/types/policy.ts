export type PolicyDecision = 'ALLOW' | 'REVIEW' | 'BLOCK';

export interface PolicyViolation {
  rule: string;
  message: string;
  severity: 'BLOCK' | 'WARNING' | string;
  target?: string | null;
}

export interface PolicyEvaluationResult {
  decision: PolicyDecision | string;
  violations: PolicyViolation[];
  warnings: string[];
  checks: Record<string, string>;
  requires_human_approval: boolean;
  policy_version: string;
  evaluated_at: string;
}
