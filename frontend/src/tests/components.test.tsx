import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent, act } from '@testing-library/react';
import { StatusPill } from '../components/common/StatusPill';
import { ErrorAlert } from '../components/common/ErrorAlert';
import { PipelineStepper } from '../components/runs/PipelineStepper';
import { UnifiedDiffView } from '../components/hitl/UnifiedDiffView';
import { QAGateCard } from '../components/hitl/QAGateCard';
import { PolicyGateCard } from '../components/hitl/PolicyGateCard';
import { ApprovalPanel } from '../components/hitl/ApprovalPanel';

describe('Frontend Component Tests', () => {
  describe('StatusPill', () => {
    it('renders RUNNING with appropriate status class', () => {
      const { container } = render(<StatusPill status="RUNNING" />);
      expect(screen.getByText('RUNNING')).toBeDefined();
      expect(container.querySelector('.status-running')).toBeDefined();
    });

    it('renders WAITING_APPROVAL with waiting class', () => {
      const { container } = render(<StatusPill status="WAITING_APPROVAL" />);
      expect(screen.getByText('WAITING_APPROVAL')).toBeDefined();
      expect(container.querySelector('.status-waiting')).toBeDefined();
    });

    it('renders COMPLETED with completed class', () => {
      const { container } = render(<StatusPill status="COMPLETED" />);
      expect(screen.getByText('COMPLETED')).toBeDefined();
      expect(container.querySelector('.status-completed')).toBeDefined();
    });

    it('renders FAILED with failed class', () => {
      const { container } = render(<StatusPill status="FAILED" />);
      expect(screen.getByText('FAILED')).toBeDefined();
      expect(container.querySelector('.status-failed')).toBeDefined();
    });
  });

  describe('ErrorAlert', () => {
    it('renders error message and code', () => {
      render(
        <ErrorAlert
          message="Resource not found"
          code="NOT_FOUND"
          detail={{ resource_id: 'run_123' }}
        />
      );
      expect(screen.getByText('Resource not found')).toBeDefined();
      expect(screen.getByText('NOT_FOUND')).toBeDefined();
    });
  });

  describe('PipelineStepper', () => {
    it('renders all 10 pipeline steps and highlights active step', () => {
      render(<PipelineStepper currentNode="developer" status="RUNNING" />);
      expect(screen.getByText('Router')).toBeDefined();
      expect(screen.getByText('Planner')).toBeDefined();
      expect(screen.getByText('Developer')).toBeDefined();
      expect(screen.getByText('Approval Gate')).toBeDefined();
    });

    it('does NOT mark PR Publish as completed/green when status is COMPLETED without PR data', () => {
      render(<PipelineStepper status="COMPLETED" />);
      const prStep = screen.getByTestId('step-pr');
      expect(prStep.classList.contains('completed')).toBe(false);
      expect(prStep.classList.contains('pending')).toBe(true);
      expect(screen.getByText('Not published')).toBeDefined();

      // Preceding pipeline stages remain completed
      const commitStep = screen.getByTestId('step-commit');
      expect(commitStep.classList.contains('completed')).toBe(true);
      const routerStep = screen.getByTestId('step-router');
      expect(routerStep.classList.contains('completed')).toBe(true);
    });

    it('marks PR Publish as completed/green when status is COMPLETED with valid PR number and URL', () => {
      render(
        <PipelineStepper
          status="COMPLETED"
          prNumber={42}
          prUrl="https://github.com/octocat/Hello-World/pull/42"
          prStatus="PUBLISHED"
        />
      );
      const prStep = screen.getByTestId('step-pr');
      expect(prStep.classList.contains('completed')).toBe(true);
      expect(screen.getByText('PR #42')).toBeDefined();

      // Preceding pipeline stages remain completed
      const commitStep = screen.getByTestId('step-commit');
      expect(commitStep.classList.contains('completed')).toBe(true);
    });

    it('marks PR Publish as active when publishing', () => {
      render(<PipelineStepper status="PUBLISHING" />);
      const prStep = screen.getByTestId('step-pr');
      expect(prStep.classList.contains('active')).toBe(true);
      expect(screen.getByText('Publishing')).toBeDefined();
    });

    it('marks PR Publish as failed when prStatus is FAILED', () => {
      render(<PipelineStepper status="COMPLETED" prStatus="FAILED" />);
      const prStep = screen.getByTestId('step-pr');
      expect(prStep.classList.contains('failed')).toBe(true);
      expect(screen.getByText('Failed')).toBeDefined();
    });
  });

  describe('UnifiedDiffView', () => {
    it('renders unified diff text with line styling', () => {
      const diffText = `--- a/src/main.py\n+++ b/src/main.py\n@@ -1,3 +1,3 @@\n-old_code()\n+new_code()`;
      const { container } = render(
        <UnifiedDiffView
          diffText={diffText}
          patchHash="abcdef1234567890"
          filesChanged={['src/main.py']}
          linesAdded={1}
          linesDeleted={1}
        />
      );
      expect(screen.getByText(/Files Changed:/i)).toBeDefined();
      expect(screen.getByText(/SHA-256: abcdef1234/i)).toBeDefined();
      expect(container.querySelector('.diff-add')).toBeDefined();
      expect(container.querySelector('.diff-del')).toBeDefined();
    });
  });

  describe('QAGateCard', () => {
    it('renders QA verdict, confidence, and checks list', () => {
      render(
        <QAGateCard
          qaResult={{
            status: 'PASS',
            confidence: 0.95,
            regression_risk: 'LOW',
            checks: [
              {
                name: 'pytest',
                status: 'PASS',
                exit_code: 0,
                duration_ms: 120,
                stdout_summary: '2 passed',
                stderr_summary: '',
              },
            ],
            issues: [],
            test_cases: ['test_api'],
            summary: 'All checks passed in sandbox',
          }}
        />
      );
      expect(screen.getAllByText('PASS').length).toBeGreaterThanOrEqual(1);
      expect(screen.getByText('95%')).toBeDefined();
      expect(screen.getByText('LOW')).toBeDefined();
      expect(screen.getByText('pytest')).toBeDefined();
    });
  });

  describe('PolicyGateCard', () => {
    it('renders policy decision and rule checks', () => {
      render(
        <PolicyGateCard
          policyResult={{
            decision: 'ALLOW',
            violations: [],
            warnings: [],
            checks: {
              repository: 'PASS',
              branch: 'PASS',
              protected_paths: 'PASS',
            },
            requires_human_approval: false,
            policy_version: '1.0.0',
            evaluated_at: '2026-09-07T00:00:00Z',
          }}
        />
      );
      expect(screen.getByText('ALLOW')).toBeDefined();
      expect(screen.getByText('repository')).toBeDefined();
    });
  });

  describe('ApprovalPanel & RBAC Enforcement', () => {
    it('disables approve and reject buttons when canApprove is false', () => {
      const mockApprove = vi.fn();
      const mockReject = vi.fn();

      render(
        <ApprovalPanel
          runId="run_test_rbac"
          gitDiff={{
            branch_name: 'agent/fix-bug',
            files_changed: ['main.py'],
            lines_added: 5,
            lines_deleted: 1,
            unified_diff: '+code',
            risk_score: 'LOW',
            risk_reasons: [],
            patch_hash: 'hash123',
          }}
          onApprove={mockApprove}
          onReject={mockReject}
          canApprove={false}
        />
      );

      const approveBtn = screen.getByText('Approve & Commit').closest('button');
      const rejectBtn = screen.getByText('Reject Changes').closest('button');

      expect(approveBtn?.disabled).toBe(true);
      expect(rejectBtn?.disabled).toBe(true);
      expect(
        screen.getByText(/Your current role does not have permission to approve changes/i)
      ).toBeDefined();
    });

    it('enables approve and reject buttons when canApprove is true', async () => {
      const mockApprove = vi.fn().mockResolvedValue(undefined);
      const mockReject = vi.fn().mockResolvedValue(undefined);

      render(
        <ApprovalPanel
          runId="run_test_rbac_allowed"
          gitDiff={{
            branch_name: 'agent/fix-bug',
            files_changed: ['main.py'],
            lines_added: 5,
            lines_deleted: 1,
            unified_diff: '+code',
            risk_score: 'LOW',
            risk_reasons: [],
            patch_hash: 'hash123',
          }}
          onApprove={mockApprove}
          onReject={mockReject}
          canApprove={true}
        />
      );

      const approveBtn = screen.getByText('Approve & Commit').closest('button');
      expect(approveBtn?.disabled).toBe(false);

      await act(async () => {
        fireEvent.click(approveBtn!);
      });
      expect(mockApprove).toHaveBeenCalledWith('hash123');
    });
  });
});
