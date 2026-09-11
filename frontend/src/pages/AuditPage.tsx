import React, { useState, useEffect } from 'react';
import { Shield, RefreshCw, Eye } from 'lucide-react';
import { auditApi } from '../api/audit';
import { AuditEventView } from '../types/api';
import { LoadingState } from '../components/common/LoadingState';
import { ErrorAlert } from '../components/common/ErrorAlert';
import { Modal } from '../components/common/Modal';

export const AuditPage: React.FC = () => {
  const [events, setEvents] = useState<AuditEventView[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [selectedEvent, setSelectedEvent] = useState<AuditEventView | null>(null);

  const fetchEvents = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await auditApi.getEvents();
      setEvents(data || []);
    } catch (err: any) {
      setError(err.message || 'Failed to retrieve audit events');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchEvents();
  }, []);

  return (
    <div>
      {error && <ErrorAlert message={error} />}

      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '20px' }}>
        <div>
          <h2 style={{ fontSize: '18px', fontWeight: 600, color: 'var(--text-primary)' }}>
            Tamper-Evident Audit Event Chain
          </h2>
          <p style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>
            Cryptographically linked SHA-256 hash chain recording all authentication, run execution, and human approval events.
          </p>
        </div>

        <button onClick={fetchEvents} className="btn btn-secondary btn-sm">
          <RefreshCw size={13} />
          <span>Refresh Audit Chain</span>
        </button>
      </div>

      <div className="panel">
        <div className="panel-header">
          <div className="panel-title">
            <Shield size={16} color="var(--brand-light)" />
            <span>Append-Only Audit Log ({events.length} events)</span>
          </div>
          <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
            Strictly RBAC Scoped (AUDIT_READ)
          </span>
        </div>

        <div className="table-container" style={{ border: 'none', borderRadius: 0 }}>
          {loading ? (
            <LoadingState message="Retrieving append-only audit log records..." />
          ) : events.length === 0 ? (
            <div style={{ padding: '48px', textAlign: 'center', color: 'var(--text-muted)' }}>
              No audit log entries recorded in this organization boundary.
            </div>
          ) : (
            <table className="data-table">
              <thead>
                <tr>
                  <th>Timestamp</th>
                  <th>Action</th>
                  <th>Actor / User</th>
                  <th>Resource</th>
                  <th>Event Hash (SHA-256)</th>
                  <th>Prev Hash (Chain)</th>
                  <th style={{ textAlign: 'right' }}>Details</th>
                </tr>
              </thead>
              <tbody>
                {events.map((ev) => {
                  const timeFormatted = ev.timestamp
                    ? new Date(ev.timestamp).toLocaleString()
                    : '--';

                  let actionColor = 'var(--text-primary)';
                  if (ev.action.includes('GRANTED') || ev.action.includes('CREATED')) {
                    actionColor = 'var(--success-text)';
                  } else if (ev.action.includes('DENIED') || ev.action.includes('FAILURE')) {
                    actionColor = 'var(--danger-text)';
                  }

                  return (
                    <tr key={ev.event_id}>
                      <td style={{ color: 'var(--text-muted)', fontSize: '11px', whiteSpace: 'nowrap' }}>
                        {timeFormatted}
                      </td>
                      <td style={{ fontWeight: 600, color: actionColor, fontSize: '11px', fontFamily: 'var(--font-mono)' }}>
                        {ev.action}
                      </td>
                      <td style={{ color: 'var(--text-secondary)' }}>{ev.user_id}</td>
                      <td>
                        <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>{ev.resource_type}:</span>{' '}
                        <code style={{ fontSize: '11px', color: 'var(--text-primary)' }}>{ev.resource_id}</code>
                      </td>
                      <td style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--brand-light)' }} title={ev.event_hash}>
                        {ev.event_hash ? `${ev.event_hash.slice(0, 10)}...` : '--'}
                      </td>
                      <td style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--text-muted)' }} title={ev.previous_hash}>
                        {ev.previous_hash ? `${ev.previous_hash.slice(0, 10)}...` : 'ROOT'}
                      </td>
                      <td style={{ textAlign: 'right' }}>
                        <button
                          onClick={() => setSelectedEvent(ev)}
                          className="btn btn-secondary btn-sm"
                          style={{ padding: '3px 8px', fontSize: '11px' }}
                        >
                          <Eye size={12} />
                          <span>View</span>
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
        </div>
      </div>

      {selectedEvent && (
        <Modal
          isOpen={Boolean(selectedEvent)}
          onClose={() => setSelectedEvent(null)}
          title={`Audit Event: ${selectedEvent.action}`}
          subtitle={`Event ID: ${selectedEvent.event_id}`}
        >
          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', fontSize: '12px' }}>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
              <div>
                <span style={{ color: 'var(--text-muted)' }}>Timestamp:</span>
                <div>{new Date(selectedEvent.timestamp).toLocaleString()}</div>
              </div>
              <div>
                <span style={{ color: 'var(--text-muted)' }}>Actor:</span>
                <div>{selectedEvent.user_id}</div>
              </div>
              <div>
                <span style={{ color: 'var(--text-muted)' }}>Resource:</span>
                <div>{selectedEvent.resource_type} ({selectedEvent.resource_id})</div>
              </div>
              <div>
                <span style={{ color: 'var(--text-muted)' }}>Organization:</span>
                <div>{selectedEvent.organization_id}</div>
              </div>
            </div>

            <div style={{ padding: '10px', background: 'var(--bg-surface-elevated)', borderRadius: 'var(--radius-sm)', fontFamily: 'var(--font-mono)', fontSize: '11px' }}>
              <div style={{ color: 'var(--text-muted)', marginBottom: '4px' }}>SHA-256 Event Hash:</div>
              <div style={{ color: 'var(--brand-light)', wordBreak: 'break-all' }}>{selectedEvent.event_hash}</div>
              <div style={{ color: 'var(--text-muted)', margin: '8px 0 4px' }}>Previous Block Hash:</div>
              <div style={{ color: 'var(--text-secondary)', wordBreak: 'break-all' }}>{selectedEvent.previous_hash}</div>
            </div>

            <div>
              <span style={{ color: 'var(--text-muted)', display: 'block', marginBottom: '4px' }}>Structured Details:</span>
              <pre
                style={{
                  padding: '10px',
                  borderRadius: 'var(--radius-sm)',
                  background: '#070a10',
                  color: 'var(--text-secondary)',
                  fontSize: '11px',
                  overflowX: 'auto',
                }}
              >
                {JSON.stringify(selectedEvent.details, null, 2)}
              </pre>
            </div>
          </div>
        </Modal>
      )}
    </div>
  );
};
