import { useState, useEffect, useCallback } from 'react';
import { TenantContext } from '../types/tenant';
import { authApi } from '../api/auth';
import { getApiKey, setApiKey, clearApiKey, getStorageType } from '../api/client';

export function useAuth() {
  const [apiKey, setApiKeyState] = useState<string>(getApiKey());
  const [storageType, setStorageTypeState] = useState<'session' | 'local'>(getStorageType());
  const [context, setContext] = useState<TenantContext | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const fetchContext = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const ctx = await authApi.getTenantContext();
      setContext(ctx);
    } catch (err: any) {
      setContext(null);
      setError(err.message || 'Failed to authenticate with backend');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchContext();
  }, [fetchContext, apiKey]);

  const updateKey = useCallback(
    (newKey: string, persistInLocal = false) => {
      setApiKey(newKey, persistInLocal);
      setApiKeyState(newKey.trim());
      setStorageTypeState(persistInLocal ? 'local' : 'session');
    },
    []
  );

  const removeKey = useCallback(() => {
    clearApiKey();
    setApiKeyState('');
    setContext(null);
  }, []);

  return {
    apiKey,
    hasApiKey: Boolean(apiKey),
    storageType,
    context,
    loading,
    error,
    refreshContext: fetchContext,
    updateApiKey: updateKey,
    clearApiKey: removeKey,
  };
}
