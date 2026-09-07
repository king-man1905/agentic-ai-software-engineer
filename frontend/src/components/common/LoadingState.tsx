import React from 'react';
import { Loader2 } from 'lucide-react';

interface LoadingStateProps {
  message?: string;
  subtext?: string;
  inline?: boolean;
}

export const LoadingState: React.FC<LoadingStateProps> = ({
  message = 'Loading telemetry & status...',
  subtext,
  inline = false,
}) => {
  if (inline) {
    return (
      <div style={{ display: 'inline-flex', alignItems: 'center', gap: '8px' }}>
        <Loader2 className="animate-spin" size={16} color="var(--brand-light)" />
        <span style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>{message}</span>
      </div>
    );
  }

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '48px 24px',
        gap: '12px',
        color: 'var(--text-secondary)',
      }}
    >
      <Loader2
        size={28}
        color="var(--brand-primary)"
        style={{ animation: 'spin 1s linear infinite' }}
      />
      <style>{`
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      `}</style>
      <div style={{ textAlign: 'center' }}>
        <p style={{ fontSize: '13px', fontWeight: 500, color: 'var(--text-primary)' }}>
          {message}
        </p>
        {subtext && (
          <p
            style={{
              fontSize: '11px',
              color: 'var(--text-muted)',
              fontFamily: 'var(--font-mono)',
              marginTop: '4px',
            }}
          >
            {subtext}
          </p>
        )}
      </div>
    </div>
  );
};
