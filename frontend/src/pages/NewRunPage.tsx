import React, { useState } from 'react';
import { Play, Sparkles } from 'lucide-react';
import { runsApi } from '../api/runs';
import { ErrorAlert } from '../components/common/ErrorAlert';

interface NewRunPageProps {
  onRunCreated: (runId: string) => void;
}

export const NewRunPage: React.FC<NewRunPageProps> = ({ onRunCreated }) => {
  const [repo, setRepo] = useState('agentic-ai-org/core-service');
  const [taskMessage, setTaskMessage] = useState('');
  const [projectId, setProjectId] = useState('');
  const [useIdempotency, setUseIdempotency] = useState(false);
  const [idempotencyKey, setIdempotencyKey] = useState('');

  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const generateKey = () => {
    setIdempotencyKey(`idem_${Math.random().toString(36).substring(2, 12)}`);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!taskMessage.trim()) {
      setError('Please provide a task or issue description for the autonomous agent.');
      return;
    }

    setSubmitting(true);
    setError(null);

    try {
      const trimmedRepo = repo.trim();
      const derivedProjectId = projectId.trim() || repo.split('/')[1] || repo;
      const res = await runsApi.createRun(
        {
          user_message: taskMessage.trim(),
          project_id: derivedProjectId,
          // The backend resolves/authorizes a registered repository by this
          // exact full "org/repo" identifier (see tenant_manager.
          // authorize_repository_access) - send it verbatim, never derived
          // or truncated, so workspace provisioning can find it. Omitted
          // entirely when there's no repository name to report, rather than
          // sending a fabricated/empty value.
          repository_id: trimmedRepo || undefined,
          metadata: {
            github_repo: trimmedRepo,
            triggered_by: 'control_plane_ui',
            timestamp: new Date().toISOString(),
          },
        },
        useIdempotency && idempotencyKey.trim() ? idempotencyKey.trim() : undefined
      );

      if (res.run_id) {
        onRunCreated(res.run_id);
      } else {
        throw new Error('Run created but no run_id returned by gateway.');
      }
    } catch (err: any) {
      setError(err.message || 'Failed to dispatch autonomous run.');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div style={{ maxWidth: '780px', margin: '0 auto' }}>
      <div className="panel">
        <div className="panel-header">
          <div className="panel-title">
            <Play size={16} color="var(--brand-light)" />
            <span>Configure Autonomous Engineering Run</span>
          </div>
          <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
            Dispatches asynchronous LangGraph execution
          </span>
        </div>

        <div className="panel-body">
          {error && <ErrorAlert message={error} onDismiss={() => setError(null)} />}

          <form onSubmit={handleSubmit}>
            <div className="form-group">
              <label className="form-label" htmlFor="repo-input">
                Target GitHub Repository
              </label>
              <input
                id="repo-input"
                type="text"
                className="form-input"
                placeholder="e.g. organization/repository"
                value={repo}
                onChange={(e) => setRepo(e.target.value)}
                required
              />
              <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
                Repository scoped to your tenant organization boundaries.
              </span>
            </div>

            <div className="form-group">
              <label className="form-label" htmlFor="task-message">
                Task Directive / Issue Description
              </label>
              <textarea
                id="task-message"
                className="form-textarea"
                placeholder="Describe the software engineering task, bug to resolve, or feature to implement..."
                value={taskMessage}
                onChange={(e) => setTaskMessage(e.target.value)}
                rows={5}
                required
              />
              <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
                The Router agent will analyze this directive and determine if Planning, RAG context, or revision loops are required.
              </span>
            </div>

            <div className="form-group">
              <label className="form-label" htmlFor="project-id-input">
                Workspace Project Identifier (Optional)
              </label>
              <input
                id="project-id-input"
                type="text"
                className="form-input"
                placeholder="e.g. core-service (defaults to repository name)"
                value={projectId}
                onChange={(e) => setProjectId(e.target.value)}
              />
            </div>

            {/* Idempotency Key Protection */}
            <div
              style={{
                padding: '12px 14px',
                borderRadius: 'var(--radius-md)',
                backgroundColor: 'var(--bg-surface-elevated)',
                border: '1px solid var(--border-subtle)',
                marginBottom: '20px',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: useIdempotency ? '10px' : 0 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <input
                    id="enable-idempotency"
                    type="checkbox"
                    checked={useIdempotency}
                    onChange={(e) => {
                      setUseIdempotency(e.target.checked);
                      if (e.target.checked && !idempotencyKey) generateKey();
                    }}
                    style={{ accentColor: 'var(--brand-primary)', width: '16px', height: '16px' }}
                  />
                  <label htmlFor="enable-idempotency" style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-primary)', cursor: 'pointer' }}>
                    Enable Idempotent Dispatch Protection
                  </label>
                </div>
                {useIdempotency && (
                  <button
                    type="button"
                    onClick={generateKey}
                    className="btn btn-secondary btn-sm"
                    style={{ padding: '2px 8px', fontSize: '10px' }}
                  >
                    Generate Key
                  </button>
                )}
              </div>

              {useIdempotency && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                  <input
                    type="text"
                    className="form-input"
                    placeholder="Idempotency-Key header value..."
                    value={idempotencyKey}
                    onChange={(e) => setIdempotencyKey(e.target.value)}
                    style={{ fontFamily: 'var(--font-mono)', fontSize: '11px' }}
                  />
                  <span style={{ fontSize: '10px', color: 'var(--text-muted)' }}>
                    Guarantees against duplicate run dispatch if network retries occur.
                  </span>
                </div>
              )}
            </div>

            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end', gap: '12px' }}>
              <button
                type="submit"
                disabled={submitting || !taskMessage.trim()}
                className="btn btn-primary"
                style={{ padding: '10px 24px' }}
              >
                <Sparkles size={16} />
                <span>{submitting ? 'Dispatching Run...' : 'Launch Autonomous Run'}</span>
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
};
