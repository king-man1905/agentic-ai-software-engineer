import { api } from './client';
import { RegisterRepositoryRequest, RepositoryResponse } from '../types/api';

export const repositoriesApi = {
  register: async (payload: RegisterRepositoryRequest): Promise<RepositoryResponse> => {
    return api.post<RepositoryResponse>('/api/v1/repositories', payload);
  },
};
