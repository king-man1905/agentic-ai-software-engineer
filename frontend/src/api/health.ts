import { api } from './client';
import { HealthCheckResponse, ReadinessCheckResponse } from '../types/api';

export const healthApi = {
  getHealth: async (): Promise<HealthCheckResponse> => {
    return api.get<HealthCheckResponse>('/health');
  },

  getReadiness: async (): Promise<ReadinessCheckResponse> => {
    return api.get<ReadinessCheckResponse>('/health/ready');
  },
};
