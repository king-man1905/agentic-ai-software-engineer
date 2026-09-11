import React, { useState } from 'react';
import { Sidebar, NavPage } from './Sidebar';
import { Header } from './Header';
import { ApiKeyModal } from '../auth/ApiKeyModal';
import { useAuth } from '../../hooks/useAuth';

interface LayoutProps {
  currentPage: NavPage;
  onNavigate: (page: NavPage) => void;
  waitingApprovalCount?: number;
  onRefresh?: () => void;
  children: React.ReactNode;
}

const PAGE_TITLES: Record<NavPage, string> = {
  overview: 'System Overview & Control Plane',
  'new-run': 'Dispatch Autonomous Run',
  runs: 'Engineering Runs Explorer',
  approvals: 'Human-in-the-Loop Approval Queue',
  analytics: 'Platform Observability & Cost Telemetry',
  audit: 'Tamper-Evident Audit Event Hash Chain',
  settings: 'Identity & Gateway Connection Settings',
};

export const Layout: React.FC<LayoutProps> = ({
  currentPage,
  onNavigate,
  waitingApprovalCount = 0,
  onRefresh,
  children,
}) => {
  const { apiKey, hasApiKey, storageType, context, updateApiKey, clearApiKey } = useAuth();
  const [isApiKeyModalOpen, setIsApiKeyModalOpen] = useState(false);

  return (
    <div className="app-shell">
      <Sidebar
        currentPage={currentPage}
        onNavigate={onNavigate}
        waitingApprovalCount={waitingApprovalCount}
      />

      <div className="main-content">
        <Header
          pageTitle={PAGE_TITLES[currentPage] || 'Control Plane'}
          tenantContext={context}
          hasApiKey={hasApiKey}
          onOpenApiKeyModal={() => setIsApiKeyModalOpen(true)}
          onRefresh={onRefresh}
        />

        <main className="page-body">{children}</main>
      </div>

      <ApiKeyModal
        isOpen={isApiKeyModalOpen}
        onClose={() => setIsApiKeyModalOpen(false)}
        currentKey={apiKey}
        storageType={storageType}
        onSave={updateApiKey}
        onClear={clearApiKey}
      />
    </div>
  );
};
