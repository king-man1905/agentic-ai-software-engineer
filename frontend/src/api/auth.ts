import { api } from './client';
import { TenantContext } from '../types/tenant';
import { CreateApiKeyRequest, ApiKeyResponse } from '../types/api';

export const authApi = {
  getTenantContext: async (): Promise<TenantContext> => {
    return api.get<TenantContext>('/api/v1/tenant/context');
  },

  createApiKey: async (request: CreateApiKeyRequest): Promise<ApiKeyResponse> => {
    return api.post<ApiKeyResponse>('/api/v1/auth/keys', request);
  },

  rotateApiKey: async (keyId: string): Promise<ApiKeyResponse> => {
    return api.post<ApiKeyResponse>(`/api/v1/auth/keys/${encodeURIComponent(keyId)}/rotate`);
  },

  revokeApiKey: async (keyId: string): Promise<void> => {
    return api.delete<void>(`/api/v1/auth/keys/${encodeURIComponent(keyId)}`);
  },
};
