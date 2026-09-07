import React from 'react';
import { ShieldAlert, CheckCircle, AlertOctagon, AlertTriangle } from 'lucide-react';
import { PolicyEvaluationResult } from '../../types/policy';

interface PolicyGateCardProps {
  policyResult?: PolicyEvaluationResult | null;
}

export const PolicyGateCard: React.FC<PolicyGateCardProps> = ({ policyResult }) => {
  if (!policyResult) {
    return (
      <div className="panel" style={{ marginBottom: 0 }}>
        <div className="panel-header">
          <div className="panel-title">
            <ShieldAlert size={16} color="var(--brand-light)" />
            <span>Policy Engine Enforcement</span>
          </div>
          <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Pending</span>
        </div>
        <div className="panel-body" style={{ color: 'var(--text-muted)', fontSize: '12px' }}>
          Deterministic organization policy checks will run before approval.
        </div>
      </div>
    );
  }

  const dec = (policyResult.decision || 'REVIEW').toUpperCase();
  let badgeColor = 'var(--warning-text)';
  let badgeBg = 'var(--warning-bg)';
  let badgeBorder = 'var(--warning-border)';
  let Icon = AlertTriangle;

  if (dec === 'ALLOW') {
    badgeColor = 'var(--success-text)';
    badgeBg = 'var(--success-bg)';
    badgeBorder = 'var(--success-border)';
    Icon = CheckCircle;
  } else if (dec === 'BLOCK') {
    badgeColor = 'var(--danger-text)';
    badgeBg = 'var(--danger-bg)';
    badgeBorder = 'var(--danger-border)';
    Icon = AlertOctagon;
  }

  const checks = policyResult.checks || {};

  return (
    <div className="panel" style={{ marginBottom: 0 }}>
      <div className="panel-header">
        <div className="panel-title">
          <Icon size={16} color={badgeColor} />
          <span>Policy Engine Verdict</span>
        </div>
        <span
          className="status-pill"
          style={{
            backgroundColor: badgeBg,
            color: badgeColor,
            borderColor: badgeBorder,
          }}
        >
          {dec}
        </span>
      </div>

      <div className="panel-body" style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
        <div>
          <div style={{ fontSize: '11px', fontWeight: 600, color: 'var(--text-secondary)', textTransform: 'uppercase', marginBottom: '8px' }}>
            Policy Rules Checklist
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(140px, 1fr))', gap: '8px' }}>
            {Object.entries(checks).map(([key, val]) => {
              const v = (val || 'PASS').toUpperCase();
              const isPass = v === 'PASS';
              return (
                <div
                  key={key}
                  style={{
                    padding: '8px 10px',
                    borderRadius: 'var(--radius-sm)',
                    background: 'var(--bg-surface-elevated)',
                    border: '1px solid var(--border-subtle)',
                    fontSize: '11px',
                  }}
                >
                  <div style={{ color: 'var(--text-muted)', textTransform: 'capitalize', fontSize: '10px' }}>
                    {key.replace(/_/g, ' ')}
                  </div>
                  <div
                    style={{
                      fontWeight: 700,
                      fontFamily: 'var(--font-mono)',
                      color: isPass ? 'var(--success-text)' : 'var(--danger-text)',
                      marginTop: '2px',
                    }}
                  >
                    {v}
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {policyResult.violations && policyResult.violations.length > 0 && (
          <div
            style={{
              padding: '10px 12px',
              borderRadius: 'var(--radius-md)',
              background: 'var(--danger-bg)',
              border: '1px solid var(--danger-border)',
            }}
          >
            <div style={{ fontSize: '12px', fontWeight: 600, color: 'var(--danger-text)', marginBottom: '4px' }}>
              Blocking Policy Violations ({policyResult.violations.length})
            </div>
            <ul style={{ paddingLeft: '18px', fontSize: '12px', color: 'var(--danger-text)' }}>
              {policyResult.violations.map((v, i) => (
                <li key={i}>
                  <strong>{v.rule}:</strong> {v.message}
                </li>
              ))}
            </ul>
          </div>
        )}

        {policyResult.warnings && policyResult.warnings.length > 0 && (
          <div
            style={{
              padding: '10px 12px',
              borderRadius: 'var(--radius-md)',
              background: 'var(--warning-bg)',
              border: '1px solid var(--warning-border)',
            }}
          >
            <div style={{ fontSize: '12px', fontWeight: 600, color: 'var(--warning-text)', marginBottom: '4px' }}>
              Policy Warnings ({policyResult.warnings.length})
            </div>
            <ul style={{ paddingLeft: '18px', fontSize: '12px', color: 'var(--warning-text)' }}>
              {policyResult.warnings.map((w, i) => (
                <li key={i}>{w}</li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
};
