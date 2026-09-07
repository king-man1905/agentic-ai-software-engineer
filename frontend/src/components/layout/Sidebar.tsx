import React from 'react';
import {
  LayoutDashboard,
  Play,
  ListOrdered,
  CheckSquare,
  BarChart3,
  FileText,
  Settings,
  Cpu,
} from 'lucide-react';

export type NavPage =
  | 'overview'
  | 'new-run'
  | 'runs'
  | 'approvals'
  | 'analytics'
  | 'audit'
  | 'settings';

interface SidebarProps {
  currentPage: NavPage;
  onNavigate: (page: NavPage) => void;
  waitingApprovalCount?: number;
}

export const Sidebar: React.FC<SidebarProps> = ({
  currentPage,
  onNavigate,
  waitingApprovalCount = 0,
}) => {
  const navItems: { id: NavPage; label: string; icon: React.ReactNode; badge?: number }[] = [
    { id: 'overview', label: 'Overview', icon: <LayoutDashboard size={16} /> },
    { id: 'new-run', label: 'New Run', icon: <Play size={16} /> },
    { id: 'runs', label: 'Runs Explorer', icon: <ListOrdered size={16} /> },
    {
      id: 'approvals',
      label: 'HITL Approvals',
      icon: <CheckSquare size={16} />,
      badge: waitingApprovalCount > 0 ? waitingApprovalCount : undefined,
    },
    { id: 'analytics', label: 'Analytics & Observability', icon: <BarChart3 size={16} /> },
    { id: 'audit', label: 'Audit Log Chain', icon: <FileText size={16} /> },
    { id: 'settings', label: 'Settings & Identity', icon: <Settings size={16} /> },
  ];

  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <div className="sidebar-logo-icon">
          <Cpu size={18} />
        </div>
        <div>
          <div className="sidebar-logo-text">AI Software Engineer</div>
          <div className="sidebar-subtitle">Control Plane v1.0.0</div>
        </div>
      </div>

      <nav className="sidebar-nav">
        {navItems.map((item) => {
          const isActive = currentPage === item.id;
          return (
            <button
              key={item.id}
              onClick={() => onNavigate(item.id)}
              className={`nav-item ${isActive ? 'active' : ''}`}
            >
              {item.icon}
              <span style={{ flex: 1, textAlign: 'left' }}>{item.label}</span>
              {item.badge != null && item.badge > 0 && (
                <span
                  style={{
                    padding: '2px 6px',
                    borderRadius: 'var(--radius-full)',
                    backgroundColor: 'var(--warning-bg)',
                    color: 'var(--warning-text)',
                    fontSize: '10px',
                    fontWeight: 700,
                    border: '1px solid var(--warning-border)',
                  }}
                >
                  {item.badge}
                </span>
              )}
            </button>
          );
        })}
      </nav>

      <div className="sidebar-footer">
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '11px', color: 'var(--text-muted)' }}>
          <span style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: 'var(--success-text)' }} />
          <span>Production Fast-Gateway</span>
        </div>
      </div>
    </aside>
  );
};
