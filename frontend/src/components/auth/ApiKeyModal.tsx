import React, { useState } from 'react';
import { Key, Shield } from 'lucide-react';
import { Modal } from '../common/Modal';

interface ApiKeyModalProps {
  isOpen: boolean;
  onClose: () => void;
  currentKey: string;
  storageType: 'session' | 'local';
  onSave: (key: string, persistInLocal: boolean) => void;
  onClear: () => void;
}

export const ApiKeyModal: React.FC<ApiKeyModalProps> = ({
  isOpen,
  onClose,
  currentKey,
  storageType,
  onSave,
  onClear,
}) => {
  const [inputValue, setInputValue] = useState(currentKey);
  const [persistLocal, setPersistLocal] = useState(storageType === 'local');

  const handleSave = (e: React.FormEvent) => {
    e.preventDefault();
    onSave(inputValue, persistLocal);
    onClose();
  };

  const handleClear = () => {
    setInputValue('');
    onClear();
    onClose();
  };

  return (
    <Modal
      isOpen={isOpen}
      onClose={onClose}
      title="API Key & Authentication Configuration"
      subtitle="Secures requests to the Agentic AI API Gateway"
      footer={
        <>
          {currentKey && (
            <button
              type="button"
              onClick={handleClear}
              className="btn btn-secondary btn-sm"
              style={{ marginRight: 'auto', color: 'var(--danger-text)' }}
            >
              Clear Key
            </button>
          )}
          <button type="button" onClick={onClose} className="btn btn-secondary btn-sm">
            Cancel
          </button>
          <button type="button" onClick={handleSave} className="btn btn-primary btn-sm">
            Save Credentials
          </button>
        </>
      }
    >
      <form onSubmit={handleSave} style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
        <div
          style={{
            padding: '12px 14px',
            borderRadius: 'var(--radius-md)',
            backgroundColor: 'var(--brand-subtle)',
            border: '1px solid var(--brand-border)',
            display: 'flex',
            alignItems: 'flex-start',
            gap: '10px',
            fontSize: '12px',
            color: 'var(--text-secondary)',
          }}
        >
          <Shield size={18} color="var(--brand-light)" style={{ flexShrink: 0, marginTop: '2px' }} />
          <div>
            <strong style={{ color: 'var(--brand-light)', display: 'block', marginBottom: '2px' }}>
              Production Authentication Mode
            </strong>
            When the backend is running with <code>AUTH_MODE=production</code>, all requests
            require a valid API key sent as <code>Authorization: Bearer &lt;key&gt;</code>.
            In development mode, blank credentials fall back to the local sandbox identity.
          </div>
        </div>

        <div className="form-group" style={{ marginBottom: 0 }}>
          <label className="form-label" htmlFor="api-key-input">
            API Key (Bearer Token)
          </label>
          <div style={{ position: 'relative' }}>
            <input
              id="api-key-input"
              type="password"
              className="form-input"
              placeholder="ak_live_..."
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              style={{ paddingLeft: '34px', fontFamily: 'var(--font-mono)' }}
              autoComplete="off"
              autoFocus
            />
            <Key
              size={16}
              color="var(--text-muted)"
              style={{
                position: 'absolute',
                left: '10px',
                top: '50%',
                transform: 'translateY(-50%)',
              }}
            />
          </div>
          <span style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '4px' }}>
            Never commit or expose API keys. The key is held client-side and attached directly to HTTP request headers.
          </span>
        </div>

        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '10px',
            padding: '10px 12px',
            borderRadius: 'var(--radius-md)',
            background: 'var(--bg-surface-elevated)',
            border: '1px solid var(--border-subtle)',
          }}
        >
          <input
            id="persist-local-checkbox"
            type="checkbox"
            checked={persistLocal}
            onChange={(e) => setPersistLocal(e.target.checked)}
            style={{ accentColor: 'var(--brand-primary)', width: '16px', height: '16px' }}
          />
          <label
            htmlFor="persist-local-checkbox"
            style={{ fontSize: '12px', color: 'var(--text-secondary)', cursor: 'pointer' }}
          >
            Persist across browser restarts (stores in <code>localStorage</code> instead of <code>sessionStorage</code>)
          </label>
        </div>
      </form>
    </Modal>
  );
};
