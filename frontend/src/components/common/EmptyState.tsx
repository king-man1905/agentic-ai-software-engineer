import React from 'react';
import { LucideIcon } from 'lucide-react';

interface EmptyStateProps {
  icon?: LucideIcon;
  title: string;
  description: string;
  action?: {
    label: string;
    onClick: () => void;
  };
}

export const EmptyState: React.FC<EmptyStateProps> = ({
  icon: Icon,
  title,
  description,
  action,
}) => {
  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '56px 24px',
        textAlign: 'center',
        gap: '12px',
        background: 'rgba(15, 23, 42, 0.4)',
        borderRadius: 'var(--radius-lg)',
        border: '1px dashed var(--border-default)',
      }}
    >
      {Icon && (
        <div
          style={{
            padding: '12px',
            borderRadius: 'var(--radius-full)',
            background: 'var(--bg-surface-elevated)',
            border: '1px solid var(--border-subtle)',
            color: 'var(--text-muted)',
          }}
        >
          <Icon size={24} />
        </div>
      )}
      <div>
        <h4 style={{ fontSize: '14px', fontWeight: 600, color: 'var(--text-primary)' }}>
          {title}
        </h4>
        <p
          style={{
            fontSize: '12px',
            color: 'var(--text-secondary)',
            maxWidth: '380px',
            marginTop: '4px',
          }}
        >
          {description}
        </p>
      </div>
      {action && (
        <button
          onClick={action.onClick}
          className="btn btn-secondary btn-sm"
          style={{ marginTop: '8px' }}
        >
          {action.label}
        </button>
      )}
    </div>
  );
};
