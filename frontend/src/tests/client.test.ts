import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import {
  getApiKey,
  setApiKey,
  clearApiKey,
  request,
  ApiClientError,
  getStorageType,
} from '../api/client';

describe('API Client & Authentication', () => {
  beforeEach(() => {
    sessionStorage.clear();
    localStorage.clear();
    vi.restoreAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('manages API key in sessionStorage by default', () => {
    expect(getApiKey()).toBe('');
    setApiKey('test-key-123', false);
    expect(getStorageType()).toBe('session');
    expect(getApiKey()).toBe('test-key-123');
    expect(sessionStorage.getItem('agentic_dashboard_api_key')).toBe('test-key-123');
    expect(localStorage.getItem('agentic_dashboard_api_key')).toBeNull();

    clearApiKey();
    expect(getApiKey()).toBe('');
  });

  it('persists API key in localStorage when requested', () => {
    setApiKey('local-key-456', true);
    expect(getStorageType()).toBe('local');
    expect(getApiKey()).toBe('local-key-456');
    expect(localStorage.getItem('agentic_dashboard_api_key')).toBe('local-key-456');
    expect(sessionStorage.getItem('agentic_dashboard_api_key')).toBeNull();
  });

  it('attaches Authorization: Bearer header when API key is configured', async () => {
    setApiKey('sec-token-xyz');

    const mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      headers: new Headers({ 'content-type': 'application/json' }),
      json: async () => ({ status: 'healthy' }),
    });
    global.fetch = mockFetch;

    const res = await request<{ status: string }>('/health');
    expect(res.status).toBe('healthy');
    expect(mockFetch).toHaveBeenCalledTimes(1);

    const callArgs = mockFetch.mock.calls[0];
    const headers = callArgs[1].headers as Headers;
    expect(headers.get('Authorization')).toBe('Bearer sec-token-xyz');
  });

  it('omits Authorization header when no API key is set', async () => {
    clearApiKey();

    const mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      headers: new Headers({ 'content-type': 'application/json' }),
      json: async () => ({ status: 'healthy' }),
    });
    global.fetch = mockFetch;

    await request('/health');
    const headers = mockFetch.mock.calls[0][1].headers as Headers;
    expect(headers.get('Authorization')).toBeNull();
  });

  it('handles 401 Unauthorized with descriptive ApiClientError', async () => {
    setApiKey('invalid-token');

    global.fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 401,
      headers: new Headers({ 'content-type': 'application/json' }),
      json: async () => ({
        detail: 'AUTHENTICATION_INVALID: Invalid Bearer token signature.',
      }),
    });

    await expect(request('/api/v1/runs')).rejects.toThrow(ApiClientError);

    try {
      await request('/api/v1/runs');
    } catch (err: any) {
      expect(err).toBeInstanceOf(ApiClientError);
      expect(err.status).toBe(401);
      expect(err.code).toBe('AUTHENTICATION_INVALID');
      expect(err.message).toContain('Invalid Bearer token signature');
    }
  });

  it('handles 403 Forbidden with RBAC permission denial code', async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 403,
      headers: new Headers({ 'content-type': 'application/json' }),
      json: async () => ({
        detail: 'PERMISSION_DENIED: Role VIEWER lacks RUN_CREATE permission.',
      }),
    });

    try {
      await request('/api/v1/runs', { method: 'POST' });
      expect.unreachable('Should have thrown 403 ApiClientError');
    } catch (err: any) {
      expect(err).toBeInstanceOf(ApiClientError);
      expect(err.status).toBe(403);
      expect(err.code).toBe('PERMISSION_DENIED');
      expect(err.message).toContain('Role VIEWER lacks RUN_CREATE permission');
    }
  });

  it('handles 409 Conflict for idempotency and state conflicts', async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 409,
      headers: new Headers({ 'content-type': 'application/json' }),
      json: async () => ({
        detail: 'IDEMPOTENCY_CONFLICT: Key already completed with different payload.',
      }),
    });

    try {
      await request('/api/v1/runs', { method: 'POST' });
    } catch (err: any) {
      expect(err).toBeInstanceOf(ApiClientError);
      expect(err.status).toBe(409);
      expect(err.code).toBe('IDEMPOTENCY_CONFLICT');
    }
  });

  it('handles 503 Service Unavailable when server is draining', async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 503,
      headers: new Headers({ 'content-type': 'application/json' }),
      json: async () => ({
        detail: 'Server is shutting down (state: DRAINING). New runs are rejected.',
      }),
    });

    try {
      await request('/api/v1/runs', { method: 'POST' });
    } catch (err: any) {
      expect(err).toBeInstanceOf(ApiClientError);
      expect(err.status).toBe(503);
      expect(err.message).toContain('Server is shutting down');
    }
  });

  it('handles network failure / API unavailable gracefully', async () => {
    global.fetch = vi.fn().mockRejectedValue(new Error('Failed to fetch'));

    try {
      await request('/health');
    } catch (err: any) {
      expect(err).toBeInstanceOf(ApiClientError);
      expect(err.status).toBe(0);
      expect(err.code).toBe('NETWORK_ERROR');
      expect(err.message).toContain('Unable to connect to the API Gateway');
    }
  });
});
