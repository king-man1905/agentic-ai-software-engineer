import React, { useState } from 'react';
import {
  ArrowLeft,
  Square,
  GitPullRequest,
  ExternalLink,
  CheckCircle,
  AlertOctagon,
  RefreshCw,
} from 'lucide-react';
import { useRunPolling } from '../hooks/useRunPolling';
import { runsApi } from '../api/runs';
import { PipelineStepper } from '../components/runs/PipelineStepper';
import { LiveTerminal } from '../components/runs/LiveTerminal';
import { ApprovalPanel } from '../components/hitl/ApprovalPanel';
import { StatusPill } from '../components/common/StatusPill';
import { ErrorAlert } from '../components/common/ErrorAlert';
import { Modal } from '../components/common/Modal';

interface RunDetailPageProps {
  runId: string;
  onBack: () => void;
  canApprove?: boolean;
}

export const RunDetailPage: React.FC<RunDetailPageProps> = ({
  runId,
  onBack,
  canApprove = true,
}) => {
  const { runStatus, events, error, isPolling, refetch } = useRunPolling(runId);

  const [cancelModalOpen, setCancelModalOpen] = useState(false);
  const [cancelReason, setCancelReason] = useState('');
  const [cancelling, setCancelling] = useState(false);
  const [cancelError, setCancelError] = useState<string | null>(null);

  const [prModalOpen, setPrModalOpen] = useState(false);
  const [prRepo, setPrRepo] = useState('');
  const [prTitle, setPrTitle] = useState('');
  const [prBaseBranch, setPrBaseBranch] = useState('main');
  const [prIsDraft, setPrIsDraft] = useState(true);
  const [publishingPr, setPublishingPr] = useState(false);
  const [prSuccessUrl, setPrSuccessUrl] = useState<string | null>(null);
  const [prError, setPrError] = useState<string | null>(null);

  const handleCancelRun = async () => {
    setCancelling(true);
    setCancelError(null);
    try {
      await runsApi.cancelRun(runId, { reason: cancelReason || 'Cancelled via Control Plane' });
      setCancelModalOpen(false);
      refetch();
    } catch (err: any) {
      setCancelError(err.message || 'Failed to cancel run.');
    } finally {
      setCancelling(false);
    }
  };

  const handleApprove = async (patchHash?: string) => {
    await runsApi.resumeRun(runId, {
      approved: true,
      reviewer: 'Control Plane Reviewer',
      patch_hash: patchHash,
    });
    refetch();
  };

  const handleReject = async (reason: string, patchHash?: string) => {
    await runsApi.resumeRun(runId, {
      approved: false,
      reviewer: 'Control Plane Reviewer',
      rejection_reason: reason,
      patch_hash: patchHash,
    });
    refetch();
  };

  const handlePublishPR = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!prRepo.trim()) {
      setPrError('Target repository name (owner/repo) is required.');
      return;
    }
    setPublishingPr(true);
    setPrError(null);
    try {
      const res = await runsApi.publishPR(runId, {
        repo_full_name: prRepo.trim(),
        title: prTitle.trim() || undefined,
        base_branch: prBaseBranch.trim() || 'main',
        draft: prIsDraft,
      });
      setPrSuccessUrl(res.pr_url);
    } catch (err: any) {
      setPrError(err.message || 'Failed to publish Pull Request.');
    } finally {
      setPublishingPr(false);
    }
  };

  const status = runStatus?.status || 'RUNNING';
  const isTerminal = ['COMPLETED', 'FAILED', 'CANCELLED', 'BLOCKED'].includes(status.toUpperCase());
  const canCancel = !isTerminal && status !== 'CANCEL_REQUESTED' && status !== 'CANCELLING';
  const canPublish = status.toUpperCase() === 'COMPLETED' && runStatus?.git_diff && !runStatus.git_diff.is_no_op;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
      {/* Top Navigation Bar */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <button onClick={onBack} className="btn btn-secondary btn-sm">
          <ArrowLeft size={14} />
          <span>Back to Runs</span>
        </button>

        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <button onClick={refetch} className="btn btn-secondary btn-sm" title="Refresh status">
            <RefreshCw size={13} />
            <span>Refresh</span>
          </button>

          {canCancel && (
            <button
              onClick={() => setCancelModalOpen(true)}
              className="btn btn-danger btn-sm"
            >
              <Square size={13} />
              <span>Cancel Run</span>
            </button>
          )}

          {canPublish && (
            <button
              onClick={() => {
                setPrRepo(runStatus?.git_diff?.branch_name ? '' : '');
                setPrModalOpen(true);
              }}
              className="btn btn-primary btn-sm"
            >
              <GitPullRequest size={13} />
              <span>Publish GitHub PR</span>
            </button>
          )}
        </div>
      </div>

      {error && <ErrorAlert message={error} />}

      {/* Run Header Info Panel */}
      <div className="panel" style={{ marginBottom: 0 }}>
        <div className="panel-header">
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <span style={{ fontSize: '15px', fontWeight: 700, fontFamily: 'var(--font-mono)', color: 'var(--text-primary)' }}>
              {runId}
            </span>
            <StatusPill status={status} />
          </div>

          <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
            {isPolling ? (
              <span style={{ display: 'inline-flex', alignItems: 'center', gap: '6px', color: 'var(--brand-light)' }}>
                <span style={{ width: '6px', height: '6px', borderRadius: '50%', backgroundColor: 'var(--brand-light)', animation: 'pulseGlow 1.5s infinite' }} />
                Authoritative Polling Active
              </span>
            ) : (
              <span>Terminal State Reached</span>
            )}
          </div>
        </div>

        <div className="panel-body" style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          {runStatus?.error_summary && (
            <div className="alert alert-danger" style={{ marginBottom: 0 }}>
              <AlertOctagon size={16} />
              <div>
                <strong>Execution Failure:</strong> {runStatus.error_summary}
              </div>
            </div>
          )}

          {/* Stepper Visualizer */}
          <PipelineStepper
            currentNode={runStatus?.current_node}
            status={status}
            prNumber={runStatus?.pr_number}
            prUrl={runStatus?.pr_url}
            prStatus={runStatus?.pr_status}
          />
        </div>
      </div>

      {/* Human-in-the-loop Gate when WAITING_APPROVAL */}
      {status.toUpperCase() === 'WAITING_APPROVAL' && (
        <ApprovalPanel
          runId={runId}
          gitDiff={runStatus?.git_diff}
          qaResult={runStatus?.qa_result}
          policyResult={runStatus?.policy_result}
          onApprove={handleApprove}
          onReject={handleReject}
          canApprove={canApprove}
        />
      )}

      {/* Live Telemetry Terminal */}
      <LiveTerminal events={events} status={status} />

      {/* Cancellation Confirmation Modal */}
      <Modal
        isOpen={cancelModalOpen}
        onClose={() => setCancelModalOpen(false)}
        title="Request Run Cancellation"
        subtitle={`Terminates background worker for run: ${runId}`}
        footer={
          <>
            <button
              onClick={() => setCancelModalOpen(false)}
              className="btn btn-secondary btn-sm"
              disabled={cancelling}
            >
              Dismiss
            </button>
            <button
              onClick={handleCancelRun}
              className="btn btn-danger btn-sm"
              disabled={cancelling}
            >
              {cancelling ? 'Cancelling...' : 'Confirm Cancellation'}
            </button>
          </>
        }
      >
        <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
          {cancelError && <ErrorAlert message={cancelError} />}
          <p style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>
            Are you sure you want to stop this engineering run? The agent runner and active sandboxes will be safely cancelled, workspace locks released, and terminal cancellation status recorded in the audit trail.
          </p>
          <div className="form-group" style={{ marginBottom: 0 }}>
            <label className="form-label" htmlFor="cancel-reason">
              Reason for Cancellation (Audit Context)
            </label>
            <input
              id="cancel-reason"
              type="text"
              className="form-input"
              placeholder="e.g. Directive outdated, duplicate run, priority changed..."
              value={cancelReason}
              onChange={(e) => setCancelReason(e.target.value)}
            />
          </div>
        </div>
      </Modal>

      {/* Publish GitHub PR Modal */}
      <Modal
        isOpen={prModalOpen}
        onClose={() => {
          setPrModalOpen(false);
          setPrSuccessUrl(null);
          setPrError(null);
        }}
        title="Publish GitHub Pull Request"
        subtitle={`Creates or reconciles Pull Request on GitHub for run: ${runId}`}
      >
        {prSuccessUrl ? (
          <div style={{ textAlign: 'center', padding: '20px 0', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '12px' }}>
            <CheckCircle size={36} color="var(--success-text)" />
            <h4 style={{ fontSize: '16px', fontWeight: 600, color: 'var(--text-primary)' }}>
              Pull Request Successfully Published!
            </h4>
            <a
              href={prSuccessUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="btn btn-primary"
              style={{ marginTop: '8px' }}
            >
              <span>Open PR on GitHub</span>
              <ExternalLink size={14} />
            </a>
          </div>
        ) : (
          <form onSubmit={handlePublishPR} style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
            {prError && <ErrorAlert message={prError} />}

            <div className="form-group">
              <label className="form-label" htmlFor="pr-repo">
                Target Repository (owner/repo)
              </label>
              <input
                id="pr-repo"
                type="text"
                className="form-input"
                placeholder="e.g. owner/repo"
                value={prRepo}
                onChange={(e) => setPrRepo(e.target.value)}
                required
              />
            </div>

            <div className="form-group">
              <label className="form-label" htmlFor="pr-title">
                Custom PR Title (Optional)
              </label>
              <input
                id="pr-title"
                type="text"
                className="form-input"
                placeholder="e.g. fix: automated agent patch"
                value={prTitle}
                onChange={(e) => setPrTitle(e.target.value)}
              />
            </div>

            <div className="form-group">
              <label className="form-label" htmlFor="pr-base-branch">
                Base Target Branch
              </label>
              <input
                id="pr-base-branch"
                type="text"
                className="form-input"
                placeholder="main"
                value={prBaseBranch}
                onChange={(e) => setPrBaseBranch(e.target.value)}
              />
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', padding: '6px 0' }}>
              <input
                id="pr-draft"
                type="checkbox"
                checked={prIsDraft}
                onChange={(e) => setPrIsDraft(e.target.checked)}
                style={{ accentColor: 'var(--brand-primary)', width: '16px', height: '16px' }}
              />
              <label htmlFor="pr-draft" style={{ fontSize: '12px', color: 'var(--text-secondary)', cursor: 'pointer' }}>
                Open Pull Request in Draft mode
              </label>
            </div>

            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '10px', marginTop: '10px' }}>
              <button
                type="button"
                onClick={() => setPrModalOpen(false)}
                className="btn btn-secondary btn-sm"
                disabled={publishingPr}
              >
                Cancel
              </button>
              <button
                type="submit"
                className="btn btn-primary btn-sm"
                disabled={publishingPr || !prRepo.trim()}
              >
                {publishingPr ? 'Publishing PR...' : 'Publish PR'}
              </button>
            </div>
          </form>
        )}
      </Modal>
    </div>
  );
};
