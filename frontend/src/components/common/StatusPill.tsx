import React from 'react';

interface StatusPillProps {
  status: string;
  className?: string;
}

export const StatusPill: React.FC<StatusPillProps> = ({ status, className = '' }) => {
  const norm = (status || 'UNKNOWN').toUpperCase();

  let statusClass = 'status-running';
  let dotColor = '#818cf8';

  if (norm === 'WAITING_APPROVAL') {
    statusClass = 'status-waiting';
    dotColor = 'var(--warning-text)';
  } else if (norm === 'COMPLETED') {
    statusClass = 'status-completed';
    dotColor = 'var(--success-text)';
  } else if (norm === 'FAILED') {
    statusClass = 'status-failed';
    dotColor = 'var(--danger-text)';
  } else if (norm === 'CANCELLED' || norm === 'CANCEL_REQUESTED' || norm === 'CANCELLING') {
    statusClass = 'status-cancelled';
    dotColor = '#94a3b8';
  } else if (norm === 'BLOCKED') {
    statusClass = 'status-blocked';
    dotColor = '#fb7185';
  }

  return (
    <span className={`status-pill ${statusClass} ${className}`}>
      <span
        style={{
          width: '6px',
          height: '6px',
          borderRadius: '50%',
          backgroundColor: dotColor,
          display: 'inline-block',
        }}
      />
      {norm}
    </span>
  );
};
