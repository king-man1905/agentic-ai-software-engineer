import React from 'react';
import { ShieldCheck, CheckCircle2, XCircle, MinusCircle, AlertTriangle } from 'lucide-react';
import { QAResult } from '../../types/qa';

interface QAGateCardProps {
  qaResult?: QAResult | null;
}

export const QAGateCard: React.FC<QAGateCardProps> = ({ qaResult }) => {
  if (!qaResult) {
    return (
      <div className="panel" style={{ marginBottom: 0 }}>
        <div className="panel-header">
          <div className="panel-title">
            <ShieldCheck size={16} color="var(--brand-light)" />
            <span>Deterministic QA Gate</span>
          </div>
          <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Pending</span>
        </div>
        <div className="panel-body" style={{ color: 'var(--text-muted)', fontSize: '12px' }}>
          Quality evaluation will be executed in the sandbox environment.
        </div>
      </div>
    );
  }

  const isPass = (qaResult.status || '').toUpperCase() === 'PASS';
  const confidencePct = Math.round((qaResult.confidence != null ? qaResult.confidence : 1.0) * 100);
  const risk = (qaResult.regression_risk || 'LOW').toUpperCase();

  return (
    <div className="panel" style={{ marginBottom: 0 }}>
      <div className="panel-header">
        <div className="panel-title">
          <ShieldCheck size={16} color={isPass ? 'var(--success-text)' : 'var(--danger-text)'} />
          <span>Quality Assurance Gate</span>
        </div>
        <span
          className="status-pill"
          style={{
            backgroundColor: isPass ? 'var(--success-bg)' : 'var(--danger-bg)',
            color: isPass ? 'var(--success-text)' : 'var(--danger-text)',
            borderColor: isPass ? 'var(--success-border)' : 'var(--danger-border)',
          }}
        >
          {qaResult.status}
        </span>
      </div>

      <div className="panel-body" style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
          <div
            style={{
              padding: '10px',
              borderRadius: 'var(--radius-md)',
              background: 'var(--bg-surface-elevated)',
              border: '1px solid var(--border-subtle)',
            }}
          >
            <div style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase' }}>
              Evaluation Confidence
            </div>
            <div style={{ fontSize: '18px', fontWeight: 700, color: 'var(--text-primary)', marginTop: '2px' }}>
              {confidencePct}%
            </div>
          </div>

          <div
            style={{
              padding: '10px',
              borderRadius: 'var(--radius-md)',
              background: 'var(--bg-surface-elevated)',
              border: '1px solid var(--border-subtle)',
            }}
          >
            <div style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase' }}>
              Regression Risk
            </div>
            <div
              style={{
                fontSize: '18px',
                fontWeight: 700,
                color: risk === 'LOW' ? 'var(--success-text)' : (risk === 'MEDIUM' ? 'var(--warning-text)' : 'var(--danger-text)'),
                marginTop: '2px',
              }}
            >
              {risk}
            </div>
          </div>
        </div>

        {/* Checks List */}
        <div>
          <div style={{ fontSize: '11px', fontWeight: 600, color: 'var(--text-secondary)', textTransform: 'uppercase', marginBottom: '8px' }}>
            Executed Quality Checks
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
            {qaResult.checks && qaResult.checks.length > 0 ? (
              qaResult.checks.map((chk, idx) => {
                const s = (chk.status || '').toUpperCase();
                let Icon = CheckCircle2;
                let color = 'var(--success-text)';
                if (s === 'FAIL' || s === 'ERROR') {
                  Icon = XCircle;
                  color = 'var(--danger-text)';
                } else if (s === 'SKIPPED') {
                  Icon = MinusCircle;
                  color = 'var(--text-muted)';
                }

                return (
                  <div
                    key={idx}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      padding: '8px 12px',
                      borderRadius: 'var(--radius-sm)',
                      background: 'var(--bg-surface-elevated)',
                      border: '1px solid var(--border-subtle)',
                      fontSize: '12px',
                      fontFamily: 'var(--font-mono)',
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <Icon size={14} color={color} />
                      <span style={{ textTransform: 'uppercase', color: 'var(--text-primary)' }}>
                        {chk.name}
                      </span>
                      {chk.duration_ms > 0 && (
                        <span style={{ color: 'var(--text-muted)', fontSize: '10px' }}>
                          ({(chk.duration_ms / 1000).toFixed(2)}s)
                        </span>
                      )}
                    </div>
                    <span style={{ color, fontWeight: 600, fontSize: '11px' }}>{s}</span>
                  </div>
                );
              })
            ) : (
              <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
                No granular check details available.
              </div>
            )}
          </div>
        </div>

        {qaResult.summary && (
          <div
            style={{
              padding: '10px 12px',
              borderRadius: 'var(--radius-md)',
              background: 'rgba(99, 102, 241, 0.08)',
              border: '1px solid var(--brand-border)',
              fontSize: '12px',
              color: 'var(--text-secondary)',
            }}
          >
            {qaResult.summary}
          </div>
        )}

        {qaResult.advisory_issues && qaResult.advisory_issues.length > 0 && (
          <div
            style={{
              padding: '10px 12px',
              borderRadius: 'var(--radius-md)',
              background: 'var(--warning-bg)',
              border: '1px solid var(--warning-border)',
              display: 'flex',
              flexDirection: 'column',
              gap: '6px',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '12px', fontWeight: 600, color: 'var(--warning-text)' }}>
              <AlertTriangle size={14} />
              <span>Advisory Warnings</span>
            </div>
            <ul style={{ paddingLeft: '20px', fontSize: '12px', color: 'var(--warning-text)' }}>
              {qaResult.advisory_issues.map((adv, idx) => (
                <li key={idx}>{adv}</li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
};
