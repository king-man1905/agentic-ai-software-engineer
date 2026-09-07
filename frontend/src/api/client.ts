import { ApiError } from '../types/api';

const API_KEY_STORAGE_KEY = 'agentic_dashboard_api_key';
const API_STORAGE_TYPE_KEY = 'agentic_dashboard_api_storage_type';

export function getStorageType(): 'session' | 'local' {
  try {
    return (localStorage.getItem(API_STORAGE_TYPE_KEY) as 'session' | 'local') || 'session';
  } catch {
    return 'session';
  }
}

export function setStorageType(type: 'session' | 'local'): void {
  try {
    localStorage.setItem(API_STORAGE_TYPE_KEY, type);
  } catch {
    // ignore
  }
}

export function getApiKey(): string {
  try {
    const fromSession = sessionStorage.getItem(API_KEY_STORAGE_KEY);
    if (fromSession) return fromSession.trim();
    const fromLocal = localStorage.getItem(API_KEY_STORAGE_KEY);
    return (fromLocal || '').trim();
  } catch {
    return '';
  }
}

export function setApiKey(key: string, persistInLocal = false): void {
  const cleanKey = key.trim();
  try {
    if (persistInLocal) {
      setStorageType('local');
      if (cleanKey) {
        localStorage.setItem(API_KEY_STORAGE_KEY, cleanKey);
      } else {
        localStorage.removeItem(API_KEY_STORAGE_KEY);
      }
      sessionStorage.removeItem(API_KEY_STORAGE_KEY);
    } else {
      setStorageType('session');
      if (cleanKey) {
        sessionStorage.setItem(API_KEY_STORAGE_KEY, cleanKey);
      } else {
        sessionStorage.removeItem(API_KEY_STORAGE_KEY);
      }
      localStorage.removeItem(API_KEY_STORAGE_KEY);
    }
  } catch {
    // Storage access restricted
  }
}

export function clearApiKey(): void {
  try {
    sessionStorage.removeItem(API_KEY_STORAGE_KEY);
    localStorage.removeItem(API_KEY_STORAGE_KEY);
  } catch {
    // ignore
  }
}

export function getApiBaseUrl(): string {
  const envUrl = import.meta.env.VITE_API_BASE_URL;
  if (envUrl && typeof envUrl === 'string') {
    return envUrl.replace(/\/+$/, '');
  }
  return '';
}

export class ApiClientError extends Error {
  status: number;
  code?: string;
  detail?: any;

  constructor(apiError: ApiError) {
    super(apiError.message);
    this.name = 'ApiClientError';
    this.status = apiError.status;
    this.code = apiError.code;
    this.detail = apiError.detail;
  }
}

export async function request<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const baseUrl = getApiBaseUrl();
  const cleanEndpoint = endpoint.startsWith('/') ? endpoint : `/${endpoint}`;
  const url = `${baseUrl}${cleanEndpoint}`;

  const headers = new Headers(options.headers || {});
  if (!headers.has('Content-Type') && !(options.body instanceof FormData)) {
    headers.set('Content-Type', 'application/json');
  }

  const apiKey = getApiKey();
  if (apiKey && !headers.has('Authorization')) {
    headers.set('Authorization', `Bearer ${apiKey}`);
  }

  let response: Response;
  try {
    response = await fetch(url, {
      ...options,
      headers,
    });
  } catch (networkErr: any) {
    throw new ApiClientError({
      status: 0,
      code: 'NETWORK_ERROR',
      message:
        'Unable to connect to the API Gateway. Ensure backend service is running.',
      detail: networkErr.message,
    });
  }

  if (response.status === 204) {
    return {} as T;
  }

  let responseData: any = null;
  const contentType = response.headers.get('content-type') || '';
  if (contentType.includes('application/json')) {
    try {
      responseData = await response.json();
    } catch {
      responseData = null;
    }
  } else {
    try {
      responseData = await response.text();
    } catch {
      responseData = null;
    }
  }

  if (!response.ok) {
    let message = 'An unexpected API error occurred.';
    let code = `HTTP_${response.status}`;

    if (responseData && typeof responseData === 'object') {
      if (typeof responseData.detail === 'string') {
        message = responseData.detail;
        if (message.includes(':')) {
          const parts = message.split(':');
          code = parts[0].trim();
        }
      } else if (responseData.message) {
        message = responseData.message;
      }
    }

    if (response.status === 401) {
      message = message || 'Authentication required. Please enter a valid API key.';
      code = code || 'UNAUTHORIZED';
    } else if (response.status === 403) {
      message = message || 'Permission denied. Your role cannot perform this operation.';
      code = code || 'FORBIDDEN';
    } else if (response.status === 404) {
      message = message || 'Requested resource not found.';
      code = code || 'NOT_FOUND';
    } else if (response.status === 409) {
      message = message || 'Conflict or lifecycle state prevents this action.';
      code = code || 'CONFLICT';
    } else if (response.status === 503) {
      message = message || 'Backend service unavailable or gracefully shutting down.';
      code = code || 'SERVICE_UNAVAILABLE';
    }

    throw new ApiClientError({
      status: response.status,
      code,
      message,
      detail: responseData,
    });
  }

  return responseData as T;
}

export const api = {
  get: <T>(endpoint: string, headers?: Record<string, string>) =>
    request<T>(endpoint, { method: 'GET', headers }),
  post: <T>(endpoint: string, body?: any, headers?: Record<string, string>) =>
    request<T>(endpoint, {
      method: 'POST',
      body: body ? JSON.stringify(body) : undefined,
      headers,
    }),
  put: <T>(endpoint: string, body?: any, headers?: Record<string, string>) =>
    request<T>(endpoint, {
      method: 'PUT',
      body: body ? JSON.stringify(body) : undefined,
      headers,
    }),
  delete: <T>(endpoint: string, headers?: Record<string, string>) =>
    request<T>(endpoint, { method: 'DELETE', headers }),
};
