import React, { useState } from 'react';
import { ListOrdered, Filter, Search, RefreshCw } from 'lucide-react';
import { useRunsList } from '../hooks/useRunsList';
import { LoadingState } from '../components/common/LoadingState';
import { ErrorAlert } from '../components/common/ErrorAlert';
import { StatusPill } from '../components/common/StatusPill';
import { RunEventsTimeline } from '../components/runs/RunEventsTimeline';
import { runsApi } from '../api/runs';
import { TelemetryEvent, RunRecord } from '../types/telemetry';

interface RunsPageProps {
  onSelectRun: (runId: string) => void;
}

export const RunsPage: React.FC<RunsPageProps> = ({ onSelectRun }) => {
  const {
    runs,
    total,
    loading,
    error,
    statusFilter,
    setStatusFilter,
    projectFilter,
    setProjectFilter,
    refetch,
  } = useRunsList({ limit: 50 });

  const [timelineRunId, setTimelineRunId] = useState<string | null>(null);
  const [timelineEvents, setTimelineEvents] = useState<TelemetryEvent[]>([]);

  const openTimeline = async (runId: string) => {
    setTimelineRunId(runId);
    try {
      const data = await runsApi.getRunEvents(runId);
      setTimelineEvents(data.events || []);
    } catch {
      setTimelineEvents([]);
    }
  };

  return (
    <div>
      {error && <ErrorAlert message={error} onDismiss={() => {}} />}

      <div className="panel">
        <div className="panel-header">
          <div className="panel-title">
            <ListOrdered size={16} color="var(--brand-light)" />
            <span>Autonomous Engineering Runs ({total})</span>
          </div>

          <button onClick={refetch} className="btn btn-secondary btn-sm" style={{ padding: '4px 8px' }}>
            <RefreshCw size={12} />
            Refresh
          </button>
        </div>

        {/* Filters bar */}
        <div
          style={{
            padding: '12px 20px',
            borderBottom: '1px solid var(--border-subtle)',
            background: 'var(--bg-surface-elevated)',
            display: 'flex',
            alignItems: 'center',
            gap: '16px',
            flexWrap: 'wrap',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', minWidth: '220px', flex: 1 }}>
            <Search size={14} color="var(--text-muted)" />
            <input
              type="text"
              className="form-input"
              placeholder="Filter by repository or project..."
              value={projectFilter || ''}
              onChange={(e) => setProjectFilter(e.target.value || undefined)}
              style={{ padding: '6px 10px', fontSize: '12px' }}
            />
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Filter size={14} color="var(--text-muted)" />
            <select
              className="form-select"
              value={statusFilter || ''}
              onChange={(e) => setStatusFilter(e.target.value || undefined)}
              style={{ padding: '6px 12px', fontSize: '12px', minWidth: '160px' }}
            >
              <option value="">All Statuses</option>
              <option value="RUNNING">RUNNING</option>
              <option value="WAITING_APPROVAL">WAITING_APPROVAL</option>
              <option value="COMPLETED">COMPLETED</option>
              <option value="FAILED">FAILED</option>
              <option value="CANCELLED">CANCELLED</option>
              <option value="BLOCKED">BLOCKED</option>
            </select>
          </div>
        </div>

        {/* Runs Table */}
        <div className="table-container" style={{ border: 'none', borderRadius: 0 }}>
          {loading ? (
            <LoadingState message="Loading runs records from telemetry store..." />
          ) : runs.length === 0 ? (
            <div style={{ padding: '48px', textAlign: 'center', color: 'var(--text-muted)' }}>
              No engineering runs match the selected criteria.
            </div>
          ) : (
            <table className="data-table">
              <thead>
                <tr>
                  <th>Run ID</th>
                  <th>Repository</th>
                  <th>Status</th>
                  <th>Created At</th>
                  <th>Duration</th>
                  <th>Cost (USD)</th>
                  <th>QA</th>
                  <th>PR</th>
                  <th style={{ textAlign: 'right' }}>Actions</th>
                </tr>
              </thead>
              <tbody>
                {runs.map((r: RunRecord) => {
                  const qaVerdict =
                    r.qa_passed === true ? (
                      <span style={{ color: 'var(--success-text)', fontWeight: 600 }}>PASS</span>
                    ) : r.qa_passed === false ? (
                      <span style={{ color: 'var(--danger-text)', fontWeight: 600 }}>FAIL</span>
                    ) : (
                      <span style={{ color: 'var(--text-muted)' }}>--</span>
                    );

                  const dur = r.duration_seconds != null ? `${r.duration_seconds.toFixed(1)}s` : '--';
                  const costStr = r.total_cost_usd != null ? `$${r.total_cost_usd.toFixed(4)}` : '$0.00';
                  const createdStr = r.created_at ? new Date(r.created_at).toLocaleTimeString() : '--';

                  return (
                    <tr key={r.run_id}>
                      <td style={{ fontFamily: 'var(--font-mono)', color: 'var(--brand-light)', fontWeight: 600 }}>
                        <span
                          onClick={() => onSelectRun(r.run_id)}
                          style={{ cursor: 'pointer', textDecoration: 'underline' }}
                        >
                          {r.run_id}
                        </span>
                      </td>
                      <td style={{ maxWidth: '200px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                        {r.repository || r.project_id || '--'}
                      </td>
                      <td>
                        <StatusPill status={r.status} />
                      </td>
                      <td style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-muted)', fontSize: '11px' }}>
                        {createdStr}
                      </td>
                      <td style={{ fontFamily: 'var(--font-mono)' }}>{dur}</td>
                      <td style={{ fontFamily: 'var(--font-mono)' }}>{costStr}</td>
                      <td>{qaVerdict}</td>
                      <td>
                        {r.pr_published ? (
                          <span style={{ color: 'var(--success-text)', fontWeight: 600 }}>YES</span>
                        ) : (
                          <span style={{ color: 'var(--text-muted)' }}>NO</span>
                        )}
                      </td>
                      <td style={{ textAlign: 'right' }}>
                        <div style={{ display: 'inline-flex', gap: '6px' }}>
                          <button
                            onClick={() => openTimeline(r.run_id)}
                            className="btn btn-secondary btn-sm"
                            style={{ padding: '3px 8px', fontSize: '11px' }}
                          >
                            Timeline
                          </button>
                          <button
                            onClick={() => onSelectRun(r.run_id)}
                            className="btn btn-primary btn-sm"
                            style={{ padding: '3px 8px', fontSize: '11px' }}
                          >
                            View
                          </button>
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
        </div>
      </div>

      {timelineRunId && (
        <RunEventsTimeline
          isOpen={Boolean(timelineRunId)}
          onClose={() => setTimelineRunId(null)}
          runId={timelineRunId}
          events={timelineEvents}
        />
      )}
    </div>
  );
};
