import React, { useState } from 'react';
import { Settings, Shield, Key, Building, User, Check, Copy, Plus } from 'lucide-react';
import { useAuth } from '../hooks/useAuth';
import { authApi } from '../api/auth';
import { getApiBaseUrl } from '../api/client';
import { ErrorAlert } from '../components/common/ErrorAlert';
import { Modal } from '../components/common/Modal';

export const SettingsPage: React.FC = () => {
  const { context, apiKey, storageType, updateApiKey, clearApiKey } = useAuth();
  const [createdKey, setCreatedKey] = useState<string | null>(null);
  const [keyModalOpen, setKeyModalOpen] = useState(false);
  const [keyName, setKeyName] = useState('cli-token');
  const [keyTtl, setKeyTtl] = useState<number | undefined>(30);
  const [creating, setCreating] = useState(false);
  const [keyError, setKeyError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  const baseUrl = getApiBaseUrl() || window.location.origin;

  const handleCreateKey = async (e: React.FormEvent) => {
    e.preventDefault();
    setCreating(true);
    setKeyError(null);
    try {
      const res = await authApi.createApiKey({
        name: keyName.trim() || 'default',
        expires_in_days: keyTtl ? Number(keyTtl) : null,
      });
      if (res.raw_key) {
        setCreatedKey(res.raw_key);
      }
    } catch (err: any) {
      setKeyError(err.message || 'Failed to generate API key.');
    } finally {
      setCreating(false);
    }
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div style={{ maxWidth: '840px', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '24px' }}>
      <div>
        <h2 style={{ fontSize: '18px', fontWeight: 600, color: 'var(--text-primary)' }}>
          Settings & Identity Management
        </h2>
        <p style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>
          Review authenticated tenant context, client connection parameters, and provision API credentials.
        </p>
      </div>

      {/* Gateway Connection Panel */}
      <div className="panel" style={{ marginBottom: 0 }}>
        <div className="panel-header">
          <div className="panel-title">
            <Settings size={16} color="var(--brand-light)" />
            <span>FastAPI Gateway Connection</span>
          </div>
        </div>
        <div className="panel-body" style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
          <div className="form-group" style={{ marginBottom: 0 }}>
            <label className="form-label">Effective API Base URL</label>
            <input
              type="text"
              className="form-input"
              value={baseUrl}
              readOnly
              style={{ fontFamily: 'var(--font-mono)', background: 'rgba(0,0,0,0.3)' }}
            />
            <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
              Configured via <code>VITE_API_BASE_URL</code> environment variable.
            </span>
          </div>
        </div>
      </div>

      {/* Authenticated Tenant Context Panel */}
      <div className="panel" style={{ marginBottom: 0 }}>
        <div className="panel-header">
          <div className="panel-title">
            <Shield size={16} color="var(--brand-light)" />
            <span>Authoritative Tenant & Role Context</span>
          </div>
          {context?.role && (
            <span
              style={{
                padding: '2px 8px',
                borderRadius: 'var(--radius-xs)',
                backgroundColor: 'rgba(99, 102, 241, 0.15)',
                color: 'var(--brand-light)',
                fontWeight: 700,
                fontSize: '11px',
                textTransform: 'uppercase',
                border: '1px solid var(--brand-border)',
              }}
            >
              Role: {context.role}
            </span>
          )}
        </div>

        <div className="panel-body" style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          {context ? (
            <>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '16px' }}>
                <div style={{ padding: '12px', background: 'var(--bg-surface-elevated)', borderRadius: 'var(--radius-md)' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase' }}>
                    <Building size={12} />
                    <span>Organization</span>
                  </div>
                  <div style={{ fontSize: '15px', fontWeight: 600, color: 'var(--text-primary)', marginTop: '4px' }}>
                    {context.organization_name}
                  </div>
                  <div style={{ fontSize: '11px', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                    {context.organization_id}
                  </div>
                </div>

                <div style={{ padding: '12px', background: 'var(--bg-surface-elevated)', borderRadius: 'var(--radius-md)' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase' }}>
                    <User size={12} />
                    <span>User Identity</span>
                  </div>
                  <div style={{ fontSize: '15px', fontWeight: 600, color: 'var(--text-primary)', marginTop: '4px' }}>
                    {context.user_name}
                  </div>
                  <div style={{ fontSize: '11px', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                    {context.user_id}
                  </div>
                </div>
              </div>

              <div>
                <div style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-secondary)', textTransform: 'uppercase', marginBottom: '8px' }}>
                  Granular Role Permissions ({context.permissions?.length || 0})
                </div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
                  {context.permissions && context.permissions.length > 0 ? (
                    context.permissions.map((p: string) => (
                      <span
                        key={p}
                        style={{
                          padding: '3px 8px',
                          borderRadius: 'var(--radius-xs)',
                          background: 'rgba(16, 185, 129, 0.1)',
                          border: '1px solid rgba(16, 185, 129, 0.3)',
                          color: 'var(--success-text)',
                          fontSize: '11px',
                          fontFamily: 'var(--font-mono)',
                        }}
                      >
                        ✓ {p}
                      </span>
                    ))
                  ) : (
                    <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>No permissions assigned</span>
                  )}
                </div>
              </div>
            </>
          ) : (
            <div style={{ color: 'var(--text-muted)', fontSize: '13px' }}>
              No active tenant context. Please provide a valid API key or verify backend connectivity.
            </div>
          )}
        </div>
      </div>

      {/* API Key Management */}
      <div className="panel" style={{ marginBottom: 0 }}>
        <div className="panel-header">
          <div className="panel-title">
            <Key size={16} color="var(--brand-light)" />
            <span>API Key Credentials</span>
          </div>

          <button onClick={() => setKeyModalOpen(true)} className="btn btn-primary btn-sm">
            <Plus size={13} />
            <span>Provision API Key</span>
          </button>
        </div>

        <div className="panel-body" style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <div>
              <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-primary)' }}>
                Client Active API Key
              </div>
              <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
                {apiKey ? `Active bearer token configured (${storageType} storage)` : 'No API key set in client'}
              </div>
            </div>

            {apiKey && (
              <button onClick={clearApiKey} className="btn btn-danger btn-sm">
                Remove Active Key
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Provision API Key Modal */}
      <Modal
        isOpen={keyModalOpen}
        onClose={() => {
          setKeyModalOpen(false);
          setCreatedKey(null);
          setKeyError(null);
        }}
        title="Provision New API Key"
        subtitle="Generates cryptographically hashed credentials scoped to your organization"
      >
        {createdKey ? (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
            <div
              style={{
                padding: '12px',
                borderRadius: 'var(--radius-md)',
                backgroundColor: 'rgba(16, 185, 129, 0.1)',
                border: '1px solid var(--success-border)',
                color: 'var(--success-text)',
                fontSize: '13px',
              }}
            >
              <strong>API Key Successfully Generated!</strong>
              <p style={{ fontSize: '11px', marginTop: '4px' }}>
                Please copy this key immediately. For security reasons, the raw key is never stored in plaintext and cannot be recovered later.
              </p>
            </div>

            <div style={{ position: 'relative' }}>
              <input
                type="text"
                className="form-input"
                readOnly
                value={createdKey}
                style={{ fontFamily: 'var(--font-mono)', fontSize: '12px', paddingRight: '70px' }}
              />
              <button
                type="button"
                onClick={() => copyToClipboard(createdKey)}
                className="btn btn-primary btn-sm"
                style={{ position: 'absolute', right: '4px', top: '4px', padding: '4px 10px' }}
              >
                {copied ? <Check size={12} /> : <Copy size={12} />}
                <span>{copied ? 'Copied' : 'Copy'}</span>
              </button>
            </div>

            <button
              type="button"
              onClick={() => {
                updateApiKey(createdKey, storageType === 'local');
                setKeyModalOpen(false);
                setCreatedKey(null);
              }}
              className="btn btn-success btn-sm"
              style={{ marginTop: '8px' }}
            >
              Use This Key in Dashboard Now
            </button>
          </div>
        ) : (
          <form onSubmit={handleCreateKey} style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
            {keyError && <ErrorAlert message={keyError} />}

            <div className="form-group">
              <label className="form-label" htmlFor="key-name">
                Key Label / Name
              </label>
              <input
                id="key-name"
                type="text"
                className="form-input"
                placeholder="e.g. ci-cd-runner, desktop-client"
                value={keyName}
                onChange={(e) => setKeyName(e.target.value)}
                required
              />
            </div>

            <div className="form-group">
              <label className="form-label" htmlFor="key-ttl">
                Expiration (Days)
              </label>
              <input
                id="key-ttl"
                type="number"
                className="form-input"
                placeholder="30 (leave blank for no expiration)"
                value={keyTtl ?? ''}
                onChange={(e) => setKeyTtl(e.target.value ? parseInt(e.target.value, 10) : undefined)}
              />
            </div>

            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '10px', marginTop: '10px' }}>
              <button
                type="button"
                onClick={() => setKeyModalOpen(false)}
                className="btn btn-secondary btn-sm"
              >
                Cancel
              </button>
              <button
                type="submit"
                className="btn btn-primary btn-sm"
                disabled={creating || !keyName.trim()}
              >
                {creating ? 'Generating...' : 'Generate API Key'}
              </button>
            </div>
          </form>
        )}
      </Modal>
    </div>
  );
};
