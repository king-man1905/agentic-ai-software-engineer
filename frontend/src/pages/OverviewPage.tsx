import React, { useState } from 'react';
import {
  Activity,
  CheckCircle2,
  Clock,
  DollarSign,
  Layers,
  ArrowRight,
  ShieldCheck,
} from 'lucide-react';
import { useAnalytics } from '../hooks/useAnalytics';
import { useRunsList } from '../hooks/useRunsList';
import { LoadingState } from '../components/common/LoadingState';
import { ErrorAlert } from '../components/common/ErrorAlert';
import { StatusPill } from '../components/common/StatusPill';
import { RunEventsTimeline } from '../components/runs/RunEventsTimeline';
import { runsApi } from '../api/runs';
import { TelemetryEvent, RunRecord } from '../types/telemetry';

interface OverviewPageProps {
  onNavigateToNewRun: () => void;
  onSelectRun: (runId: string) => void;
}

export const OverviewPage: React.FC<OverviewPageProps> = ({
  onNavigateToNewRun,
  onSelectRun,
}) => {
  const { overview, quality, error: analyticsError } = useAnalytics();
  const { runs, loading: runsLoading, error: runsError } = useRunsList({ limit: 10 });

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

  const successRate = overview ? Math.round((overview.success_rate ?? (overview.total_runs ? overview.successful_runs / overview.total_runs : 0)) * 100) : 0;
  const avgDurSec = overview ? (overview.avg_duration_seconds ?? (overview.avg_duration_ms ? overview.avg_duration_ms / 1000 : 0)).toFixed(1) : '0.0';
  const totalCost = overview?.total_cost_usd != null ? `$${overview.total_cost_usd.toFixed(4)}` : '$0.0000';
  const totalTokens = (overview?.total_tokens || 0).toLocaleString();

  return (
    <div>
      {analyticsError && <ErrorAlert message={analyticsError} />}
      {runsError && <ErrorAlert message={runsError} />}

      {/* Hero Action Banner */}
      <div
        style={{
          padding: '20px 24px',
          borderRadius: 'var(--radius-lg)',
          background: 'linear-gradient(135deg, rgba(30, 27, 75, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%)',
          border: '1px solid var(--brand-border)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          marginBottom: '24px',
        }}
      >
        <div>
          <h2 style={{ fontSize: '18px', fontWeight: 700, color: '#ffffff' }}>
            Autonomous AI Engineering Pipeline
          </h2>
          <p style={{ fontSize: '13px', color: 'var(--text-secondary)', marginTop: '4px', maxWidth: '640px' }}>
            Multi-agent engineering control plane with LangGraph durable checkpointing, AST-aware patching, sandbox pytest validation, and cryptographic HITL approval.
          </p>
        </div>

        <button onClick={onNavigateToNewRun} className="btn btn-primary" style={{ padding: '10px 20px' }}>
          <span>Dispatch New Run</span>
          <ArrowRight size={16} />
        </button>
      </div>

      {/* KPI Metrics Grid */}
      <div className="metrics-grid">
        <div className="metric-card">
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <span className="metric-label">Total Runs</span>
            <Activity size={16} color="var(--brand-light)" />
          </div>
          <div className="metric-value">{overview?.total_runs ?? 0}</div>
          <div className="metric-subtext">
            {overview?.successful_runs ?? 0} successful / {overview?.failed_runs ?? 0} failed
          </div>
        </div>

        <div className="metric-card">
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <span className="metric-label">Success Rate</span>
            <CheckCircle2 size={16} color="var(--success-text)" />
          </div>
          <div className="metric-value" style={{ color: 'var(--success-text)' }}>
            {successRate}%
          </div>
          <div className="metric-subtext">Deterministic quality gates</div>
        </div>

        <div className="metric-card">
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <span className="metric-label">Avg Duration</span>
            <Clock size={16} color="var(--info-text)" />
          </div>
          <div className="metric-value">{avgDurSec}s</div>
          <div className="metric-subtext">Per autonomous workflow</div>
        </div>

        <div className="metric-card">
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <span className="metric-label">Total Cost</span>
            <DollarSign size={16} color="var(--success-text)" />
          </div>
          <div className="metric-value" style={{ color: 'var(--success-text)' }}>
            {totalCost}
          </div>
          <div className="metric-subtext">{totalTokens} tokens consumed</div>
        </div>
      </div>

      {/* Quality & RAG Summary Panel */}
      {quality && (
        <div className="panel">
          <div className="panel-header">
            <div className="panel-title">
              <ShieldCheck size={16} color="var(--brand-light)" />
              <span>Quality Assurance & RAG Diagnostics</span>
            </div>
          </div>
          <div className="panel-body">
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px' }}>
              <div>
                <div style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase' }}>
                  QA Gate Pass Rate
                </div>
                <div style={{ fontSize: '20px', fontWeight: 700, color: 'var(--success-text)', marginTop: '2px' }}>
                  {Math.round((quality.qa_pass_rate || 0) * 100)}%
                </div>
              </div>

              <div>
                <div style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase' }}>
                  Avg Revisions Per Run
                </div>
                <div style={{ fontSize: '20px', fontWeight: 700, color: 'var(--text-primary)', marginTop: '2px' }}>
                  {(quality.avg_revisions || quality.avg_revisions_per_run || 0).toFixed(1)}
                </div>
              </div>

              {quality.rag_analytics && (
                <div>
                  <div style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase' }}>
                    RAG Retrieval Success
                  </div>
                  <div style={{ fontSize: '20px', fontWeight: 700, color: 'var(--brand-light)', marginTop: '2px' }}>
                    {Math.round((quality.rag_analytics.retrieval_success_rate || 0) * 100)}%
                  </div>
                </div>
              )}

              <div>
                <div style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase' }}>
                  Security & Test Failures
                </div>
                <div style={{ fontSize: '20px', fontWeight: 700, color: 'var(--danger-text)', marginTop: '2px' }}>
                  {(quality.test_failure_count || 0) + (quality.security_failure_count || 0)}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Recent Runs Table */}
      <div className="panel">
        <div className="panel-header">
          <div className="panel-title">
            <Layers size={16} color="var(--brand-light)" />
            <span>Recent Autonomous Engineering Runs</span>
          </div>
          <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
            Showing latest {runs.length} runs
          </span>
        </div>

        <div className="table-container" style={{ border: 'none', borderRadius: 0 }}>
          {runsLoading ? (
            <LoadingState message="Loading recent runs..." />
          ) : runs.length === 0 ? (
            <div style={{ padding: '36px', textAlign: 'center', color: 'var(--text-muted)' }}>
              No runs executed yet. Click &quot;Dispatch New Run&quot; above to launch an autonomous task.
            </div>
          ) : (
            <table className="data-table">
              <thead>
                <tr>
                  <th>Run ID</th>
                  <th>Repository / Task</th>
                  <th>Status</th>
                  <th>Duration</th>
                  <th>Cost (USD)</th>
                  <th>QA Verdict</th>
                  <th>PR Published</th>
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

                  return (
                    <tr key={r.run_id}>
                      <td style={{ fontFamily: 'var(--font-mono)', color: 'var(--brand-light)', fontWeight: 600 }}>
                        <span
                          onClick={() => onSelectRun(r.run_id)}
                          style={{ cursor: 'pointer', textDecoration: 'underline' }}
                        >
                          {r.run_id.slice(0, 14)}...
                        </span>
                      </td>
                      <td style={{ maxWidth: '240px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                        {r.repository || r.project_id || r.user_message || '--'}
                      </td>
                      <td>
                        <StatusPill status={r.status} />
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
                            Details
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
