import React, { useEffect, useState } from 'react';
import { Key, User, Building, RefreshCw, AlertCircle } from 'lucide-react';
import { TenantContext } from '../../types/tenant';
import { healthApi } from '../../api/health';

interface HeaderProps {
  pageTitle: string;
  tenantContext?: TenantContext | null;
  hasApiKey: boolean;
  onOpenApiKeyModal: () => void;
  onRefresh?: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  pageTitle,
  tenantContext,
  hasApiKey,
  onOpenApiKeyModal,
  onRefresh,
}) => {
  const [healthStatus, setHealthStatus] = useState<'healthy' | 'unhealthy' | 'checking'>('checking');

  useEffect(() => {
    let mounted = true;
    const check = async () => {
      try {
        const res = await healthApi.getHealth();
        if (mounted && res.status === 'healthy') setHealthStatus('healthy');
      } catch {
        if (mounted) setHealthStatus('unhealthy');
      }
    };
    check();
    const interval = setInterval(check, 30000);
    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, []);

  return (
    <header className="top-header">
      <div className="header-title-area">
        <h1 className="header-page-title">{pageTitle}</h1>
      </div>

      <div className="header-actions">
        {/* Gateway Health Indicator */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            fontSize: '11px',
            padding: '4px 8px',
            borderRadius: 'var(--radius-full)',
            background: 'var(--bg-surface-elevated)',
            border: '1px solid var(--border-subtle)',
          }}
          title={healthStatus === 'healthy' ? 'FastAPI Gateway is healthy' : 'FastAPI Gateway unreachable'}
        >
          {healthStatus === 'healthy' ? (
            <>
              <span style={{ width: '6px', height: '6px', borderRadius: '50%', backgroundColor: 'var(--success-text)' }} />
              <span style={{ color: 'var(--text-secondary)' }}>Gateway Online</span>
            </>
          ) : (
            <>
              <AlertCircle size={12} color="var(--danger-text)" />
              <span style={{ color: 'var(--danger-text)' }}>Gateway Offline</span>
            </>
          )}
        </div>

        {/* Tenant Context Pill */}
        {tenantContext && (
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              padding: '4px 10px',
              borderRadius: 'var(--radius-md)',
              background: 'var(--bg-surface-elevated)',
              border: '1px solid var(--border-subtle)',
              fontSize: '11px',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '4px', color: 'var(--text-secondary)' }}>
              <Building size={12} />
              <span>{tenantContext.organization_name || tenantContext.organization_id}</span>
            </div>
            <span style={{ color: 'var(--text-muted)' }}>•</span>
            <div style={{ display: 'flex', alignItems: 'center', gap: '4px', color: 'var(--text-primary)' }}>
              <User size={12} />
              <span>{tenantContext.user_name || tenantContext.user_id}</span>
            </div>
            <span
              style={{
                padding: '1px 5px',
                borderRadius: 'var(--radius-xs)',
                backgroundColor: 'rgba(99, 102, 241, 0.15)',
                color: 'var(--brand-light)',
                fontWeight: 700,
                fontSize: '9px',
                textTransform: 'uppercase',
              }}
            >
              {tenantContext.role}
            </span>
          </div>
        )}

        {/* API Key Modal Button */}
        <button
          onClick={onOpenApiKeyModal}
          className={`btn btn-sm ${hasApiKey ? 'btn-secondary' : 'btn-primary'}`}
          style={{ padding: '5px 12px', fontSize: '11px' }}
        >
          <Key size={13} />
          <span>{hasApiKey ? 'API Key Set' : 'Set API Key'}</span>
        </button>

        {onRefresh && (
          <button
            onClick={onRefresh}
            className="btn btn-secondary btn-sm"
            style={{ padding: '5px 8px' }}
            title="Refresh active view"
            aria-label="Refresh view"
          >
            <RefreshCw size={13} />
          </button>
        )}
      </div>
    </header>
  );
};
