import React from 'react';
import {
  Cpu,
  ShieldCheck,
  AlertTriangle,
  RefreshCw,
} from 'lucide-react';
import { useAnalytics } from '../hooks/useAnalytics';
import { LoadingState } from '../components/common/LoadingState';
import { ErrorAlert } from '../components/common/ErrorAlert';

export const AnalyticsPage: React.FC = () => {
  const { overview, quality, models, failures, loading, error, refetch } = useAnalytics();

  const successRate = overview ? Math.round((overview.success_rate ?? (overview.total_runs ? overview.successful_runs / overview.total_runs : 0)) * 100) : 0;
  const avgDurSec = overview ? (overview.avg_duration_seconds ?? (overview.avg_duration_ms ? overview.avg_duration_ms / 1000 : 0)).toFixed(1) : '0.0';
  const totalCost = overview?.total_cost_usd != null ? `$${overview.total_cost_usd.toFixed(4)}` : '$0.00';
  const totalTokens = (overview?.total_tokens || 0).toLocaleString();

  const modelUsageList = models?.models ? Object.values(models.models) : [];
  const failureCategories = failures?.categories || {};

  return (
    <div>
      {error && <ErrorAlert message={error} />}

      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '20px' }}>
        <div>
          <h2 style={{ fontSize: '18px', fontWeight: 600, color: 'var(--text-primary)' }}>
            Observability & Cost Telemetry
          </h2>
          <p style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>
            Real-time analytics aggregated across all autonomous workflows in your organization.
          </p>
        </div>

        <button onClick={refetch} className="btn btn-secondary btn-sm">
          <RefreshCw size={13} />
          <span>Refresh Metrics</span>
        </button>
      </div>

      {loading ? (
        <LoadingState message="Aggregating observability telemetry from SQLite store..." />
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
          {/* Top KPI Cards */}
          <div className="metrics-grid" style={{ marginBottom: 0 }}>
            <div className="metric-card">
              <span className="metric-label">Total Workflows</span>
              <div className="metric-value">{overview?.total_runs ?? 0}</div>
              <div className="metric-subtext">
                {overview?.successful_runs ?? 0} succeeded • {overview?.failed_runs ?? 0} failed
              </div>
            </div>

            <div className="metric-card">
              <span className="metric-label">Overall Success Rate</span>
              <div className="metric-value" style={{ color: 'var(--success-text)' }}>
                {successRate}%
              </div>
              <div className="metric-subtext">
                {overview?.waiting_approval_runs ?? 0} currently awaiting review
              </div>
            </div>

            <div className="metric-card">
              <span className="metric-label">Average Run Duration</span>
              <div className="metric-value">{avgDurSec}s</div>
              <div className="metric-subtext">Across complete agent graph</div>
            </div>

            <div className="metric-card">
              <span className="metric-label">Total Incurred Cost</span>
              <div className="metric-value" style={{ color: 'var(--success-text)' }}>
                {totalCost}
              </div>
              <div className="metric-subtext">{totalTokens} tokens processed</div>
            </div>
          </div>

          {/* Quality & RAG Telemetry */}
          {quality && (
            <div className="panel" style={{ marginBottom: 0 }}>
              <div className="panel-header">
                <div className="panel-title">
                  <ShieldCheck size={16} color="var(--brand-light)" />
                  <span>Quality Assurance, Self-Correction & RAG Retrieval</span>
                </div>
              </div>

              <div className="panel-body">
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '16px' }}>
                  <div style={{ padding: '12px', background: 'var(--bg-surface-elevated)', borderRadius: 'var(--radius-md)' }}>
                    <div style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase' }}>
                      QA Gate Pass Rate
                    </div>
                    <div style={{ fontSize: '22px', fontWeight: 700, color: 'var(--success-text)', marginTop: '4px' }}>
                      {Math.round((quality.qa_pass_rate || 0) * 100)}%
                    </div>
                    <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '2px' }}>
                      {quality.total_evaluated} evaluated
                    </div>
                  </div>

                  <div style={{ padding: '12px', background: 'var(--bg-surface-elevated)', borderRadius: 'var(--radius-md)' }}>
                    <div style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase' }}>
                      Avg Self-Correction Cycles
                    </div>
                    <div style={{ fontSize: '22px', fontWeight: 700, color: 'var(--text-primary)', marginTop: '4px' }}>
                      {(quality.avg_revisions || quality.avg_revisions_per_run || 0).toFixed(2)}
                    </div>
                    <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '2px' }}>
                      {quality.total_revisions} total revisions
                    </div>
                  </div>

                  {quality.rag_analytics && (
                    <div style={{ padding: '12px', background: 'var(--bg-surface-elevated)', borderRadius: 'var(--radius-md)' }}>
                      <div style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase' }}>
                        Codebase Retrieval Accuracy
                      </div>
                      <div style={{ fontSize: '22px', fontWeight: 700, color: 'var(--brand-light)', marginTop: '4px' }}>
                        {Math.round((quality.rag_analytics.retrieval_success_rate || 0) * 100)}%
                      </div>
                      <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '2px' }}>
                        {quality.rag_analytics.total_retrievals} retrievals executed
                      </div>
                    </div>
                  )}

                  <div style={{ padding: '12px', background: 'var(--bg-surface-elevated)', borderRadius: 'var(--radius-md)' }}>
                    <div style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase' }}>
                      Test & Security Blockers
                    </div>
                    <div style={{ fontSize: '22px', fontWeight: 700, color: 'var(--danger-text)', marginTop: '4px' }}>
                      {(quality.test_failure_count || 0) + (quality.security_failure_count || 0)}
                    </div>
                    <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '2px' }}>
                      {quality.security_failure_count || 0} security / {quality.test_failure_count || 0} pytest
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Model & Token Consumption Breakdown */}
          <div className="panel" style={{ marginBottom: 0 }}>
            <div className="panel-header">
              <div className="panel-title">
                <Cpu size={16} color="var(--brand-light)" />
                <span>LLM Provider & Model Consumption Telemetry</span>
              </div>
            </div>

            <div className="table-container" style={{ border: 'none', borderRadius: 0 }}>
              {modelUsageList.length === 0 ? (
                <div style={{ padding: '30px', textAlign: 'center', color: 'var(--text-muted)' }}>
                  No LLM model telemetry recorded yet.
                </div>
              ) : (
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Model Name</th>
                      <th>Workflows</th>
                      <th>Input Tokens</th>
                      <th>Output Tokens</th>
                      <th>Total Tokens</th>
                      <th>Estimated Cost</th>
                    </tr>
                  </thead>
                  <tbody>
                    {modelUsageList.map((m, idx) => (
                      <tr key={idx}>
                        <td style={{ fontFamily: 'var(--font-mono)', fontWeight: 600, color: 'var(--brand-light)' }}>
                          {m.model || m.model_name || 'unknown'}
                        </td>
                        <td>{m.request_count || m.run_count || 0}</td>
                        <td style={{ fontFamily: 'var(--font-mono)' }}>{(m.input_tokens || 0).toLocaleString()}</td>
                        <td style={{ fontFamily: 'var(--font-mono)' }}>{(m.output_tokens || 0).toLocaleString()}</td>
                        <td style={{ fontFamily: 'var(--font-mono)', fontWeight: 600 }}>{(m.total_tokens || 0).toLocaleString()}</td>
                        <td style={{ fontFamily: 'var(--font-mono)', color: 'var(--success-text)', fontWeight: 600 }}>
                          {m.estimated_cost_usd != null ? `$${m.estimated_cost_usd.toFixed(4)}` : '$0.00'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
          </div>

          {/* Failure Category Distribution */}
          <div className="panel" style={{ marginBottom: 0 }}>
            <div className="panel-header">
              <div className="panel-title">
                <AlertTriangle size={16} color="var(--danger-text)" />
                <span>Failure Category Distribution</span>
              </div>
              <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
                {failures?.total_failures || 0} total failures
              </span>
            </div>

            <div className="panel-body">
              {Object.keys(failureCategories).length === 0 ? (
                <div style={{ color: 'var(--text-muted)', fontSize: '13px' }}>
                  No failure telemetry records observed.
                </div>
              ) : (
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
                  {Object.entries(failureCategories).map(([cat, count]) => (
                    <div
                      key={cat}
                      style={{
                        padding: '6px 12px',
                        borderRadius: 'var(--radius-full)',
                        backgroundColor: 'var(--danger-bg)',
                        border: '1px solid var(--danger-border)',
                        color: 'var(--danger-text)',
                        fontSize: '12px',
                        fontWeight: 600,
                        display: 'flex',
                        alignItems: 'center',
                        gap: '6px',
                      }}
                    >
                      <span>{cat}</span>
                      <span
                        style={{
                          padding: '1px 6px',
                          borderRadius: 'var(--radius-full)',
                          backgroundColor: 'rgba(239, 68, 68, 0.25)',
                          fontSize: '11px',
                          fontFamily: 'var(--font-mono)',
                        }}
                      >
                        {Number(count)}
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
