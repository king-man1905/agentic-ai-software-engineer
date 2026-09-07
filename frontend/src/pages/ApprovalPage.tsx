import React, { useState, useEffect } from 'react';
import { CheckSquare, RefreshCw } from 'lucide-react';
import { useRunsList } from '../hooks/useRunsList';
import { runsApi } from '../api/runs';
import { RunStatusResponse } from '../types/api';
import { RunRecord } from '../types/telemetry';
import { ApprovalPanel } from '../components/hitl/ApprovalPanel';
import { LoadingState } from '../components/common/LoadingState';
import { EmptyState } from '../components/common/EmptyState';
import { ErrorAlert } from '../components/common/ErrorAlert';

interface ApprovalPageProps {
  canApprove?: boolean;
}

export const ApprovalPage: React.FC<ApprovalPageProps> = ({ canApprove = true }) => {
  const { runs, loading: listLoading, error: listError, refetch } = useRunsList({
    status: 'WAITING_APPROVAL',
  });

  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [activeStatus, setActiveStatus] = useState<RunStatusResponse | null>(null);
  const [loadingStatus, setLoadingStatus] = useState<boolean>(false);
  const [actionError, setActionError] = useState<string | null>(null);

  useEffect(() => {
    if (runs.length > 0 && !selectedRunId) {
      setSelectedRunId(runs[0].run_id);
    } else if (runs.length === 0) {
      setSelectedRunId(null);
      setActiveStatus(null);
    }
  }, [runs, selectedRunId]);

  useEffect(() => {
    if (!selectedRunId) return;
    setLoadingStatus(true);
    runsApi
      .getRunStatus(selectedRunId)
      .then((res: RunStatusResponse) => setActiveStatus(res))
      .catch((err: any) => setActionError(err.message))
      .finally(() => setLoadingStatus(false));
  }, [selectedRunId]);

  const handleApprove = async (patchHash?: string) => {
    if (!selectedRunId) return;
    setActionError(null);
    try {
      await runsApi.resumeRun(selectedRunId, {
        approved: true,
        reviewer: 'HITL Reviewer',
        patch_hash: patchHash,
      });
      refetch();
    } catch (err: any) {
      setActionError(err.message || 'Failed to approve changes.');
    }
  };

  const handleReject = async (reason: string, patchHash?: string) => {
    if (!selectedRunId) return;
    setActionError(null);
    try {
      await runsApi.resumeRun(selectedRunId, {
        approved: false,
        reviewer: 'HITL Reviewer',
        rejection_reason: reason,
        patch_hash: patchHash,
      });
      refetch();
    } catch (err: any) {
      setActionError(err.message || 'Failed to submit rejection.');
    }
  };

  return (
    <div>
      {listError && <ErrorAlert message={listError} />}
      {actionError && <ErrorAlert message={actionError} onDismiss={() => setActionError(null)} />}

      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px' }}>
        <div>
          <h2 style={{ fontSize: '18px', fontWeight: 600, color: 'var(--text-primary)' }}>
            Human-in-the-Loop Approval Queue
          </h2>
          <p style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>
            Review and cryptographically authorize staged code modifications before committing to feature branches.
          </p>
        </div>

        <button onClick={refetch} className="btn btn-secondary btn-sm">
          <RefreshCw size={13} />
          <span>Refresh Queue</span>
        </button>
      </div>

      {listLoading ? (
        <LoadingState message="Checking approval queue..." />
      ) : runs.length === 0 ? (
        <EmptyState
          icon={CheckSquare}
          title="Approval Queue Clear"
          description="There are currently no engineering runs paused at the Human-in-the-Loop approval gate."
        />
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          {/* Pending Runs Selector */}
          {runs.length > 1 && (
            <div
              style={{
                display: 'flex',
                gap: '8px',
                overflowX: 'auto',
                paddingBottom: '4px',
              }}
            >
              {runs.map((r: RunRecord) => {
                const isSelected = r.run_id === selectedRunId;
                return (
                  <button
                    key={r.run_id}
                    onClick={() => setSelectedRunId(r.run_id)}
                    className={`btn btn-sm ${isSelected ? 'btn-primary' : 'btn-secondary'}`}
                    style={{ fontFamily: 'var(--font-mono)', fontSize: '11px' }}
                  >
                    {r.run_id.slice(0, 14)}...
                  </button>
                );
              })}
            </div>
          )}

          {/* Active Run Approval Panel */}
          {loadingStatus ? (
            <LoadingState message="Fetching diff and QA artifacts..." />
          ) : selectedRunId && activeStatus ? (
            <ApprovalPanel
              runId={selectedRunId}
              gitDiff={activeStatus.git_diff}
              qaResult={activeStatus.qa_result}
              policyResult={activeStatus.policy_result}
              onApprove={handleApprove}
              onReject={handleReject}
              canApprove={canApprove}
            />
          ) : null}
        </div>
      )}
    </div>
  );
};
