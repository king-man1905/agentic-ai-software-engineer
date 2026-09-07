import React from 'react';
import { Copy, Check } from 'lucide-react';

interface UnifiedDiffViewProps {
  diffText: string;
  patchHash?: string;
  filesChanged?: string[];
  linesAdded?: number;
  linesDeleted?: number;
}

export const UnifiedDiffView: React.FC<UnifiedDiffViewProps> = ({
  diffText,
  patchHash,
  filesChanged = [],
  linesAdded = 0,
  linesDeleted = 0,
}) => {
  const [copied, setCopied] = React.useState(false);

  const handleCopy = () => {
    if (!diffText) return;
    navigator.clipboard.writeText(diffText);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const lines = (diffText || '').split('\n');

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '8px 12px',
          borderRadius: 'var(--radius-md)',
          background: 'var(--bg-surface-elevated)',
          border: '1px solid var(--border-subtle)',
          fontSize: '12px',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <span style={{ color: 'var(--text-secondary)' }}>
            Files Changed: <strong style={{ color: 'var(--text-primary)' }}>{filesChanged.length}</strong>
          </span>
          <span style={{ color: 'var(--success-text)', fontWeight: 600 }}>+{linesAdded}</span>
          <span style={{ color: 'var(--danger-text)', fontWeight: 600 }}>-{linesDeleted}</span>
          {patchHash && (
            <span
              style={{
                fontFamily: 'var(--font-mono)',
                fontSize: '11px',
                padding: '2px 8px',
                borderRadius: 'var(--radius-xs)',
                backgroundColor: 'rgba(99, 102, 241, 0.1)',
                border: '1px solid var(--brand-border)',
                color: 'var(--brand-light)',
              }}
              title={`Cryptographic SHA-256 Patch Hash: ${patchHash}`}
            >
              SHA-256: {patchHash.slice(0, 10)}...
            </span>
          )}
        </div>

        <button
          onClick={handleCopy}
          className="btn btn-secondary btn-sm"
          style={{ padding: '2px 8px', fontSize: '11px' }}
        >
          {copied ? <Check size={12} color="var(--success-text)" /> : <Copy size={12} />}
          {copied ? 'Copied' : 'Copy Diff'}
        </button>
      </div>

      <div className="diff-container">
        {lines.length === 0 || !diffText.trim() ? (
          <div style={{ padding: '20px', textAlign: 'center', color: 'var(--text-muted)' }}>
            No unified diff content produced.
          </div>
        ) : (
          lines.map((line, index) => {
            let lineClass = 'diff-normal';
            if (line.startsWith('+++') || line.startsWith('---') || line.startsWith('@@')) {
              lineClass = 'diff-hdr';
            } else if (line.startsWith('+')) {
              lineClass = 'diff-add';
            } else if (line.startsWith('-')) {
              lineClass = 'diff-del';
            }

            return (
              <div key={index} className={`diff-line ${lineClass}`}>
                {line || ' '}
              </div>
            );
          })
        )}
      </div>
    </div>
  );
};
