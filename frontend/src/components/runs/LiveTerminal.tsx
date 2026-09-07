import React, { useRef, useEffect, useState } from 'react';
import { Terminal, Trash2, ArrowDown, Radio } from 'lucide-react';
import { TelemetryEvent } from '../../types/telemetry';

interface LiveTerminalProps {
  events: TelemetryEvent[];
  status: string;
  onClear?: () => void;
  className?: string;
}

const TAG_STYLES: Record<string, { bg: string; text: string; border: string }> = {
  SYSTEM: { bg: 'rgba(6, 182, 212, 0.15)', text: '#67e8f9', border: 'rgba(6, 182, 212, 0.4)' },
  ROUTER: { bg: 'rgba(99, 102, 241, 0.15)', text: '#a5b4fc', border: 'rgba(99, 102, 241, 0.4)' },
  PLANNER: { bg: 'rgba(59, 130, 246, 0.15)', text: '#93c5fd', border: 'rgba(59, 130, 246, 0.4)' },
  KNOWLEDGE: { bg: 'rgba(168, 85, 247, 0.15)', text: '#d8b4fe', border: 'rgba(168, 85, 247, 0.4)' },
  DEVELOPER: { bg: 'rgba(14, 165, 233, 0.15)', text: '#7dd3fc', border: 'rgba(14, 165, 233, 0.4)' },
  SANDBOX: { bg: 'rgba(20, 184, 166, 0.15)', text: '#5eead4', border: 'rgba(20, 184, 166, 0.4)' },
  QA_GATE: { bg: 'rgba(16, 185, 129, 0.15)', text: '#6ee7b7', border: 'rgba(16, 185, 129, 0.4)' },
  REVISION: { bg: 'rgba(249, 115, 22, 0.15)', text: '#fdba74', border: 'rgba(249, 115, 22, 0.4)' },
  GIT_VCS: { bg: 'rgba(139, 92, 246, 0.15)', text: '#c4b5fd', border: 'rgba(139, 92, 246, 0.4)' },
  GATE: { bg: 'rgba(245, 158, 11, 0.15)', text: '#fde68a', border: 'rgba(245, 158, 11, 0.5)' },
  SUCCESS: { bg: 'rgba(16, 185, 129, 0.15)', text: '#34d399', border: 'rgba(16, 185, 129, 0.5)' },
  ERROR: { bg: 'rgba(239, 68, 68, 0.15)', text: '#fca5a5', border: 'rgba(239, 68, 68, 0.5)' },
  DECISION: { bg: 'rgba(217, 70, 239, 0.15)', text: '#f0abfc', border: 'rgba(217, 70, 239, 0.4)' },
};

export const LiveTerminal: React.FC<LiveTerminalProps> = ({
  events,
  status,
  onClear,
  className = '',
}) => {
  const bodyRef = useRef<HTMLDivElement>(null);
  const [autoScroll, setAutoScroll] = useState(true);

  useEffect(() => {
    if (autoScroll && bodyRef.current) {
      bodyRef.current.scrollTop = bodyRef.current.scrollHeight;
    }
  }, [events, autoScroll]);

  const normStatus = (status || '').toUpperCase();
  const isConnected = normStatus === 'RUNNING' || normStatus === 'WAITING_APPROVAL' || normStatus === 'REVISING';

  return (
    <div className={`terminal-container ${className}`}>
      <div className="terminal-header">
        <div className="terminal-title">
          <Terminal size={14} color="var(--brand-light)" />
          <span>Execution Telemetry Stream</span>
          <span style={{ fontSize: '10px', color: 'var(--text-muted)' }}>
            ({events.length} events)
          </span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              fontSize: '11px',
              padding: '2px 8px',
              borderRadius: 'var(--radius-full)',
              backgroundColor: isConnected ? 'rgba(16, 185, 129, 0.1)' : 'rgba(148, 163, 184, 0.1)',
              color: isConnected ? 'var(--success-text)' : 'var(--text-muted)',
              border: `1px solid ${isConnected ? 'var(--success-border)' : 'var(--border-subtle)'}`,
            }}
          >
            <Radio size={12} className={isConnected ? 'animate-pulse' : ''} />
            <span style={{ fontWeight: 600 }}>{isConnected ? 'LIVE' : 'DISCONNECTED'}</span>
          </div>

          <button
            onClick={() => setAutoScroll(!autoScroll)}
            className="btn btn-secondary btn-sm"
            style={{
              padding: '3px 8px',
              fontSize: '11px',
              color: autoScroll ? 'var(--brand-light)' : 'var(--text-muted)',
            }}
            title={autoScroll ? 'Disable auto-scroll' : 'Enable auto-scroll'}
          >
            <ArrowDown size={12} />
            Auto-scroll
          </button>

          {onClear && (
            <button
              onClick={onClear}
              className="btn btn-secondary btn-sm"
              style={{ padding: '3px 8px', fontSize: '11px' }}
              title="Clear terminal logs"
            >
              <Trash2 size={12} />
              Clear
            </button>
          )}
        </div>
      </div>

      <div ref={bodyRef} className="terminal-body">
        {events.length === 0 ? (
          <div style={{ color: 'var(--text-muted)', fontSize: '11px', padding: '12px 0' }}>
            $ Autonomous workflow engine listening for LangGraph execution telemetry...
          </div>
        ) : (
          events.map((ev, index) => {
            const tag = (ev.node || ev.event_type || 'SYSTEM').toUpperCase();
            const tagStyle = TAG_STYLES[tag] || {
              bg: 'rgba(51, 65, 85, 0.4)',
              text: '#cbd5e1',
              border: 'rgba(51, 65, 85, 0.6)',
            };
            const timeStr = ev.timestamp
              ? new Date(ev.timestamp).toLocaleTimeString([], { hour12: false })
              : '--:--:--';

            const summary =
              ev.safe_metadata?.message ||
              ev.safe_metadata?.reason ||
              ev.safe_metadata?.summary ||
              ev.event_type;

            return (
              <div key={ev.event_id || index} className="terminal-line">
                <span className="terminal-ts">[{timeStr}]</span>
                <span
                  className="terminal-tag"
                  style={{
                    backgroundColor: tagStyle.bg,
                    color: tagStyle.text,
                    borderColor: tagStyle.border,
                  }}
                >
                  {tag}
                </span>
                <span className="terminal-msg">{summary}</span>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
};
