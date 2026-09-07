import React from 'react';
import { Clock, Coins, Layers } from 'lucide-react';
import { Modal } from '../common/Modal';
import { TelemetryEvent } from '../../types/telemetry';

interface RunEventsTimelineProps {
  isOpen: boolean;
  onClose: () => void;
  runId: string;
  events: TelemetryEvent[];
}

export const RunEventsTimeline: React.FC<RunEventsTimelineProps> = ({
  isOpen,
  onClose,
  runId,
  events,
}) => {
  return (
    <Modal
      isOpen={isOpen}
      onClose={onClose}
      title={`Run Lifecycle Timeline: ${runId}`}
      subtitle={`${events.length} chronological telemetry event(s) recorded`}
      maxWidth="760px"
    >
      {events.length === 0 ? (
        <div style={{ color: 'var(--text-muted)', textAlign: 'center', padding: '36px 0' }}>
          No telemetry events recorded for this run.
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
          {events.map((ev) => {
            const timeFormatted = ev.timestamp
              ? new Date(ev.timestamp).toLocaleString()
              : 'Unknown time';

            return (
              <div
                key={ev.event_id}
                style={{
                  padding: '12px 14px',
                  borderRadius: 'var(--radius-md)',
                  backgroundColor: 'var(--bg-surface-elevated)',
                  border: '1px solid var(--border-subtle)',
                  fontFamily: 'var(--font-mono)',
                  fontSize: '12px',
                }}
              >
                <div
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    marginBottom: '6px',
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <span
                      style={{
                        padding: '1px 6px',
                        borderRadius: 'var(--radius-xs)',
                        backgroundColor: 'var(--brand-subtle)',
                        border: '1px solid var(--brand-border)',
                        color: 'var(--brand-light)',
                        fontSize: '10px',
                        fontWeight: 700,
                        textTransform: 'uppercase',
                      }}
                    >
                      {ev.node || 'SYSTEM'}
                    </span>
                    <strong style={{ color: 'var(--text-primary)', fontSize: '12px' }}>
                      {ev.event_type}
                    </strong>
                  </div>
                  <span style={{ color: 'var(--text-muted)', fontSize: '11px' }}>
                    {timeFormatted}
                  </span>
                </div>

                <div
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '14px',
                    color: 'var(--text-secondary)',
                    fontSize: '11px',
                    marginBottom: '6px',
                  }}
                >
                  {ev.duration_ms != null && (
                    <span style={{ display: 'flex', alignItems: 'center', gap: '4px', color: 'var(--info-text)' }}>
                      <Clock size={12} />
                      {ev.duration_ms.toFixed(1)}ms
                    </span>
                  )}
                  {ev.tokens != null && (
                    <span style={{ display: 'flex', alignItems: 'center', gap: '4px', color: '#c084fc' }}>
                      <Layers size={12} />
                      {ev.tokens.toLocaleString()} tokens
                    </span>
                  )}
                  {ev.cost_usd != null && (
                    <span style={{ display: 'flex', alignItems: 'center', gap: '4px', color: 'var(--success-text)' }}>
                      <Coins size={12} />
                      ${ev.cost_usd.toFixed(4)}
                    </span>
                  )}
                </div>

                {ev.safe_metadata && Object.keys(ev.safe_metadata).length > 0 && (
                  <pre
                    style={{
                      padding: '8px 10px',
                      borderRadius: 'var(--radius-xs)',
                      backgroundColor: 'rgba(9, 13, 22, 0.7)',
                      color: 'var(--text-secondary)',
                      fontSize: '11px',
                      overflowX: 'auto',
                      border: '1px solid rgba(51, 65, 85, 0.3)',
                    }}
                  >
                    {JSON.stringify(ev.safe_metadata, null, 2)}
                  </pre>
                )}
              </div>
            );
          })}
        </div>
      )}
    </Modal>
  );
};
