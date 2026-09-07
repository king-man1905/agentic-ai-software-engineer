import React, { useState } from 'react';
import { CheckCircle2, XCircle, ShieldCheck } from 'lucide-react';
import { GitDiffSummary } from '../../types/diff';
import { QAResult } from '../../types/qa';
import { PolicyEvaluationResult } from '../../types/policy';
import { UnifiedDiffView } from './UnifiedDiffView';
import { QAGateCard } from './QAGateCard';
import { PolicyGateCard } from './PolicyGateCard';

interface ApprovalPanelProps {
  runId: string;
  gitDiff?: GitDiffSummary | null;
  qaResult?: QAResult | null;
  policyResult?: PolicyEvaluationResult | null;
  onApprove: (patchHash?: string) => Promise<void>;
  onReject: (reason: string, patchHash?: string) => Promise<void>;
  canApprove?: boolean;
}

export const ApprovalPanel: React.FC<ApprovalPanelProps> = ({
  runId,
  gitDiff,
  qaResult,
  policyResult,
  onApprove,
  onReject,
  canApprove = true,
}) => {
  const [submitting, setSubmitting] = useState(false);
  const [rejectionReason, setRejectionReason] = useState('');
  const [showRejectInput, setShowRejectInput] = useState(false);

  const risk = (gitDiff?.risk_score || 'LOW').toUpperCase();
  const patchHash = gitDiff?.patch_hash;

  const handleApprove = async () => {
    setSubmitting(true);
    try {
      await onApprove(patchHash);
    } finally {
      setSubmitting(false);
    }
  };

  const handleReject = async () => {
    if (!showRejectInput) {
      setShowRejectInput(true);
      return;
    }
    setSubmitting(true);
    try {
      await onReject(rejectionReason || 'Rejected via HITL Control Plane', patchHash);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="panel" style={{ border: '1px solid var(--brand-border)', boxShadow: 'var(--shadow-glow)' }}>
      <div className="panel-header" style={{ background: 'rgba(99, 102, 241, 0.12)' }}>
        <div className="panel-title">
          <ShieldCheck size={18} color="var(--brand-light)" />
          <span>Human-In-The-Loop Approval Gate</span>
          <span style={{ fontSize: '11px', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>
            Run #{runId.slice(0, 12)}
          </span>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span
            style={{
              padding: '3px 8px',
              borderRadius: 'var(--radius-full)',
              fontSize: '11px',
              fontWeight: 700,
              fontFamily: 'var(--font-mono)',
              textTransform: 'uppercase',
              backgroundColor: risk === 'HIGH' ? 'var(--danger-bg)' : (risk === 'MEDIUM' ? 'var(--warning-bg)' : 'var(--success-bg)'),
              color: risk === 'HIGH' ? 'var(--danger-text)' : (risk === 'MEDIUM' ? 'var(--warning-text)' : 'var(--success-text)'),
              border: `1px solid ${risk === 'HIGH' ? 'var(--danger-border)' : (risk === 'MEDIUM' ? 'var(--warning-border)' : 'var(--success-border)')}`,
            }}
          >
            {risk} RISK
          </span>
        </div>
      </div>

      <div className="panel-body" style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
        {/* Quality and Policy Cards Grid */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '16px' }}>
          <QAGateCard qaResult={qaResult} />
          <PolicyGateCard policyResult={policyResult} />
        </div>

        {/* Risk Reasons */}
        {gitDiff?.risk_reasons && gitDiff.risk_reasons.length > 0 && (
          <div
            style={{
              padding: '12px 14px',
              borderRadius: 'var(--radius-md)',
              background: 'var(--bg-surface-elevated)',
              border: '1px solid var(--border-subtle)',
            }}
          >
            <div style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-secondary)', textTransform: 'uppercase', marginBottom: '6px' }}>
              Risk Assessment Factors
            </div>
            <ul style={{ paddingLeft: '18px', fontSize: '12px', color: 'var(--text-secondary)' }}>
              {gitDiff.risk_reasons.map((r, i) => (
                <li key={i}>{r}</li>
              ))}
            </ul>
          </div>
        )}

        {/* Unified Diff View */}
        <div>
          <div style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-secondary)', textTransform: 'uppercase', marginBottom: '8px' }}>
            Proposed Code Changes (Staged on Branch: {gitDiff?.branch_name || 'agent/feature'})
          </div>
          <UnifiedDiffView
            diffText={gitDiff?.unified_diff || ''}
            patchHash={patchHash}
            filesChanged={gitDiff?.files_changed || []}
            linesAdded={gitDiff?.lines_added || 0}
            linesDeleted={gitDiff?.lines_deleted || 0}
          />
        </div>

        {/* Rejection input when expanded */}
        {showRejectInput && (
          <div className="form-group" style={{ marginBottom: 0 }}>
            <label className="form-label" htmlFor="rejection-reason">
              Reason for Rejection (Audit Trail)
            </label>
            <textarea
              id="rejection-reason"
              className="form-textarea"
              placeholder="Explain why the proposed changes are being rejected..."
              value={rejectionReason}
              onChange={(e) => setRejectionReason(e.target.value)}
              rows={2}
            />
          </div>
        )}

        {/* Action Buttons */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            paddingTop: '12px',
            borderTop: '1px solid var(--border-subtle)',
          }}
        >
          <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
            {!canApprove ? (
              <span style={{ color: 'var(--warning-text)' }}>
                Your current role does not have permission to approve changes.
              </span>
            ) : (
              <span>
                Approval binds patch hash <code>{patchHash ? patchHash.slice(0, 10) : 'none'}</code> to prevent drift.
              </span>
            )}
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <button
              onClick={handleReject}
              disabled={submitting || !canApprove}
              className="btn btn-danger btn-sm"
              style={{ padding: '8px 16px' }}
            >
              <XCircle size={14} />
              {showRejectInput ? 'Confirm Rejection' : 'Reject Changes'}
            </button>

            <button
              onClick={handleApprove}
              disabled={submitting || !canApprove}
              className="btn btn-success btn-sm"
              style={{ padding: '8px 20px' }}
            >
              <CheckCircle2 size={14} />
              Approve & Commit
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
