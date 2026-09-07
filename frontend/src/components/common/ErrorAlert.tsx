import React from 'react';
import { AlertTriangle, X } from 'lucide-react';

interface ErrorAlertProps {
  message: string;
  code?: string;
  detail?: any;
  onDismiss?: () => void;
  className?: string;
}

export const ErrorAlert: React.FC<ErrorAlertProps> = ({
  message,
  code,
  detail,
  onDismiss,
  className = '',
}) => {
  return (
    <div className={`alert alert-danger ${className}`}>
      <AlertTriangle size={18} style={{ flexShrink: 0, marginTop: '2px' }} />
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <strong style={{ fontWeight: 600 }}>{message}</strong>
          {code && (
            <span
              style={{
                fontSize: '10px',
                fontFamily: 'var(--font-mono)',
                padding: '1px 6px',
                borderRadius: 'var(--radius-xs)',
                backgroundColor: 'rgba(239, 68, 68, 0.2)',
                border: '1px solid rgba(239, 68, 68, 0.4)',
              }}
            >
              {code}
            </span>
          )}
        </div>
        {detail && typeof detail === 'object' && Object.keys(detail).length > 0 && (
          <pre
            style={{
              marginTop: '6px',
              padding: '6px 8px',
              fontSize: '11px',
              fontFamily: 'var(--font-mono)',
              backgroundColor: 'rgba(0, 0, 0, 0.3)',
              borderRadius: 'var(--radius-xs)',
              overflowX: 'auto',
            }}
          >
            {JSON.stringify(detail, null, 2)}
          </pre>
        )}
      </div>
      {onDismiss && (
        <button
          onClick={onDismiss}
          style={{
            background: 'none',
            border: 'none',
            color: 'inherit',
            cursor: 'pointer',
            padding: '2px',
          }}
          aria-label="Dismiss error"
        >
          <X size={16} />
        </button>
      )}
    </div>
  );
};
