import { api } from './client';
import {
  AnalyticsOverview,
  QualityAnalytics,
  ModelAnalytics,
  FailureAnalytics,
} from '../types/telemetry';

export const analyticsApi = {
  getOverview: async (): Promise<AnalyticsOverview> => {
    return api.get<AnalyticsOverview>('/api/v1/analytics/overview');
  },

  getQuality: async (): Promise<QualityAnalytics> => {
    return api.get<QualityAnalytics>('/api/v1/analytics/quality');
  },

  getModels: async (): Promise<ModelAnalytics> => {
    return api.get<ModelAnalytics>('/api/v1/analytics/models');
  },

  getFailures: async (): Promise<FailureAnalytics> => {
    return api.get<FailureAnalytics>('/api/v1/analytics/failures');
  },
};
