import React from 'react';
import {
  GitBranch,
  FileCode,
  ShieldCheck,
  CheckCircle2,
  PauseCircle,
  Brain,
  Layers,
  Wrench,
  Search,
} from 'lucide-react';

interface PipelineStepperProps {
  currentNode?: string | null;
  status: string;
  prNumber?: number | null;
  prUrl?: string | null;
  prStatus?: string | null;
}

interface StepDef {
  key: string;
  name: string;
  icon: React.ReactNode;
}

const STEPS: StepDef[] = [
  { key: 'router', name: 'Router', icon: <GitBranch size={14} /> },
  { key: 'planner', name: 'Planner', icon: <Brain size={14} /> },
  { key: 'knowledge', name: 'RAG', icon: <Search size={14} /> },
  { key: 'developer', name: 'Developer', icon: <FileCode size={14} /> },
  { key: 'qa', name: 'QA Sandbox', icon: <ShieldCheck size={14} /> },
  { key: 'revision', name: 'Revision', icon: <Wrench size={14} /> },
  { key: 'git_prepare', name: 'Git VCS', icon: <Layers size={14} /> },
  { key: 'approval', name: 'Approval Gate', icon: <PauseCircle size={14} /> },
  { key: 'commit', name: 'Commit', icon: <CheckCircle2 size={14} /> },
  { key: 'pr', name: 'PR Publish', icon: <GitBranch size={14} /> },
];

export const PipelineStepper: React.FC<PipelineStepperProps> = ({
  currentNode,
  status,
  prNumber,
  prUrl,
  prStatus,
}) => {
  const normStatus = (status || 'CREATED').toUpperCase();
  const activeNode = currentNode ? currentNode.toLowerCase() : (normStatus === 'RUNNING' ? 'router' : '');

  // Determine active step index for pipeline stages 0..8 (Router through Commit)
  let targetIdx = STEPS.findIndex((s) => s.key === activeNode);
  if (normStatus === 'WAITING_APPROVAL') {
    targetIdx = STEPS.findIndex((s) => s.key === 'approval');
  } else if (normStatus === 'COMPLETED') {
    targetIdx = STEPS.length - 1; // All steps prior to PR Publish completed
  } else if (targetIdx === -1 && normStatus === 'RUNNING') {
    targetIdx = 0;
  }

  // Authoritative PR Publish state
  const hasValidPr = Boolean(
    prNumber !== undefined &&
    prNumber !== null &&
    prNumber > 0 &&
    prUrl !== undefined &&
    prUrl !== null &&
    prUrl.trim().length > 0
  );
  const isPrFailed = (prStatus || '').toUpperCase() === 'FAILED';
  const isPrPublishing =
    normStatus === 'PUBLISHING' ||
    (prStatus || '').toUpperCase() === 'PUBLISHING' ||
    activeNode === 'pr';

  return (
    <div className="stepper-container" style={{ padding: '8px 0', gap: '6px' }}>
      {STEPS.map((step, idx) => {
        let stateClass = 'pending';
        let statusBadge = '○';
        let subLabel: string | null = null;

        if (step.key === 'pr') {
          // PR Publish step: must NOT automatically become completed/green on COMPLETED status
          if (hasValidPr) {
            stateClass = 'completed';
            statusBadge = '✓';
            subLabel = `PR #${prNumber}`;
          } else if (isPrFailed) {
            stateClass = 'failed';
            statusBadge = '✗';
            subLabel = 'Failed';
          } else if (isPrPublishing) {
            stateClass = 'active';
            statusBadge = '▶';
            subLabel = 'Publishing';
          } else {
            stateClass = 'pending';
            statusBadge = '○';
            subLabel = 'Not published';
          }
        } else {
          // Pre-PR pipeline stages (Router through Commit) retain existing behavior
          if (normStatus === 'COMPLETED' || idx < targetIdx) {
            stateClass = 'completed';
            statusBadge = '✓';
          } else if (idx === targetIdx) {
            if (normStatus === 'WAITING_APPROVAL') {
              stateClass = 'waiting';
              statusBadge = '⏸';
            } else if (normStatus === 'FAILED' || normStatus === 'BLOCKED') {
              stateClass = 'failed';
              statusBadge = '✗';
            } else {
              stateClass = 'active';
              statusBadge = '▶';
            }
          }
        }

        return (
          <div
            key={step.key}
            data-testid={`step-${step.key}`}
            className={`step-node ${stateClass}`}
            title={step.key === 'pr' && prUrl ? prUrl : undefined}
            style={{
              flex: '1 1 0',
              minWidth: '100px',
              padding: '8px 10px',
              borderWidth: '1px',
              borderStyle: 'solid',
            }}
          >
            <div className="step-node-header">
              <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                {step.icon}
                <span>0{idx + 1}</span>
              </span>
              <span style={{ fontWeight: 700 }}>{statusBadge}</span>
            </div>
            <div className="step-node-title" style={{ marginTop: '2px', fontSize: '11px' }}>
              {step.name}
            </div>
            {subLabel && (
              <div
                style={{
                  fontSize: '9px',
                  marginTop: '2px',
                  fontWeight: 500,
                  opacity: stateClass === 'pending' ? 0.7 : 0.95,
                }}
              >
                {subLabel}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
};
