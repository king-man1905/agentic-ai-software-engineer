import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, act } from '@testing-library/react';
import { NewRunPage } from '../pages/NewRunPage';
import { runsApi } from '../api/runs';

vi.mock('../api/runs', () => ({
  runsApi: {
    createRun: vi.fn(),
  },
}));

const mockedCreateRun = runsApi.createRun as unknown as ReturnType<typeof vi.fn>;

describe('NewRunPage - repository_id wiring', () => {
  beforeEach(() => {
    mockedCreateRun.mockReset();
    mockedCreateRun.mockResolvedValue({ run_id: 'run_123', status: 'RUNNING' });
  });

  const fillAndSubmit = async (repoValue: string | null, taskMessage = 'Fix the bug') => {
    const onRunCreated = vi.fn();
    render(<NewRunPage onRunCreated={onRunCreated} />);

    const repoInput = screen.getByLabelText('Target GitHub Repository') as HTMLInputElement;
    if (repoValue !== null) {
      fireEvent.change(repoInput, { target: { value: repoValue } });
    }

    const taskInput = screen.getByLabelText('Task Directive / Issue Description');
    fireEvent.change(taskInput, { target: { value: taskMessage } });

    const submitBtn = screen.getByText('Launch Autonomous Run').closest('button')!;
    await act(async () => {
      fireEvent.click(submitBtn);
    });

    return onRunCreated;
  };

  it('includes the exact repository_id (unsplit, unmodified) alongside project_id and metadata.github_repo', async () => {
    await fillAndSubmit('acme-org/widgets-service');

    expect(mockedCreateRun).toHaveBeenCalledTimes(1);
    const payload = mockedCreateRun.mock.calls[0][0];

    expect(payload.repository_id).toBe('acme-org/widgets-service');
    // project_id keeps its existing derived (split-by-org) behavior - untouched.
    expect(payload.project_id).toBe('widgets-service');
    // metadata.github_repo keeps its existing raw-repo behavior - untouched.
    expect(payload.metadata.github_repo).toBe('acme-org/widgets-service');
  });

  it('never derives/truncates repository_id from the repo name - it matches the full raw input, not just the project_id half', async () => {
    await fillAndSubmit('another-org/some-repo');
    const payload = mockedCreateRun.mock.calls[0][0];

    expect(payload.repository_id).toBe('another-org/some-repo');
    expect(payload.repository_id).not.toBe(payload.project_id);
  });

  it('does not fabricate a repository_id when the repo field is blank', async () => {
    // The repo input is HTML `required`, so a truly empty string never
    // reaches submission via the browser's own validation - a whitespace-
    // only value is what actually reaches handleSubmit's trim() logic.
    await fillAndSubmit('   ');
    const payload = mockedCreateRun.mock.calls[0][0];

    expect(payload.repository_id).toBeUndefined();
    // project_id/metadata behavior for a blank repo is unchanged by this fix.
    expect(payload.metadata.github_repo).toBe('');
  });

  it('preserves existing New Run behavior: default repo value, project_id override, and onRunCreated callback', async () => {
    const onRunCreated = vi.fn();
    render(<NewRunPage onRunCreated={onRunCreated} />);

    // Default repo value is pre-filled (existing behavior), left untouched.
    const repoInput = screen.getByLabelText('Target GitHub Repository') as HTMLInputElement;
    expect(repoInput.value).toBe('agentic-ai-org/core-service');

    const projectIdInput = screen.getByLabelText('Workspace Project Identifier (Optional)');
    fireEvent.change(projectIdInput, { target: { value: 'custom-project' } });

    const taskInput = screen.getByLabelText('Task Directive / Issue Description');
    fireEvent.change(taskInput, { target: { value: 'Add a feature' } });

    const submitBtn = screen.getByText('Launch Autonomous Run').closest('button')!;
    await act(async () => {
      fireEvent.click(submitBtn);
    });

    const payload = mockedCreateRun.mock.calls[0][0];
    expect(payload.project_id).toBe('custom-project');
    expect(payload.repository_id).toBe('agentic-ai-org/core-service');
    expect(payload.metadata.github_repo).toBe('agentic-ai-org/core-service');
    expect(onRunCreated).toHaveBeenCalledWith('run_123');
  });
});
