import React, { useState } from 'react';
import { Layout } from './components/layout/Layout';
import { NavPage } from './components/layout/Sidebar';
import { OverviewPage } from './pages/OverviewPage';
import { NewRunPage } from './pages/NewRunPage';
import { RunsPage } from './pages/RunsPage';
import { RunDetailPage } from './pages/RunDetailPage';
import { ApprovalPage } from './pages/ApprovalPage';
import { AnalyticsPage } from './pages/AnalyticsPage';
import { AuditPage } from './pages/AuditPage';
import { SettingsPage } from './pages/SettingsPage';
import { useRunsList } from './hooks/useRunsList';
import { useAuth } from './hooks/useAuth';
import './styles/globals.css';
import './styles/components.css';

export const App: React.FC = () => {
  const [currentPage, setCurrentPage] = useState<NavPage>('overview');
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);

  const { runs: pendingApprovalRuns, refetch: refetchPending } = useRunsList({
    status: 'WAITING_APPROVAL',
  });
  const { context } = useAuth();

  const canApprove =
    context?.role === 'OWNER' ||
    context?.role === 'ADMIN' ||
    context?.role === 'REVIEWER' ||
    context?.role === 'SECURITY_REVIEWER' ||
    (context?.permissions?.includes('RUN_APPROVE') ?? true);

  const handleNavigate = (page: NavPage) => {
    setSelectedRunId(null);
    setCurrentPage(page);
    refetchPending();
  };

  const handleSelectRun = (runId: string) => {
    setSelectedRunId(runId);
  };

  const handleRunCreated = (runId: string) => {
    setSelectedRunId(runId);
    refetchPending();
  };

  return (
    <Layout
      currentPage={currentPage}
      onNavigate={handleNavigate}
      waitingApprovalCount={pendingApprovalRuns.length}
      onRefresh={refetchPending}
    >
      {selectedRunId ? (
        <RunDetailPage
          runId={selectedRunId}
          onBack={() => setSelectedRunId(null)}
          canApprove={canApprove}
        />
      ) : currentPage === 'overview' ? (
        <OverviewPage
          onNavigateToNewRun={() => handleNavigate('new-run')}
          onSelectRun={handleSelectRun}
        />
      ) : currentPage === 'new-run' ? (
        <NewRunPage onRunCreated={handleRunCreated} />
      ) : currentPage === 'runs' ? (
        <RunsPage onSelectRun={handleSelectRun} />
      ) : currentPage === 'approvals' ? (
        <ApprovalPage canApprove={canApprove} />
      ) : currentPage === 'analytics' ? (
        <AnalyticsPage />
      ) : currentPage === 'audit' ? (
        <AuditPage />
      ) : currentPage === 'settings' ? (
        <SettingsPage />
      ) : (
        <OverviewPage
          onNavigateToNewRun={() => handleNavigate('new-run')}
          onSelectRun={handleSelectRun}
        />
      )}
    </Layout>
  );
};

export default App;
