import { api } from './client';
import {
  CreateRunRequest,
  RunStatusResponse,
  ResumeRunRequest,
  CancelRunRequest,
  PublishPRRequest,
  PublishPRResponse,
  RunListResponse,
  RunEventsResponse,
} from '../types/api';

export const runsApi = {
  createRun: async (
    payload: CreateRunRequest,
    idempotencyKey?: string
  ): Promise<RunStatusResponse> => {
    const headers: Record<string, string> = {};
    if (idempotencyKey) {
      headers['Idempotency-Key'] = idempotencyKey;
    }
    return api.post<RunStatusResponse>('/api/v1/runs', payload, headers);
  },

  listRuns: async (params?: {
    status?: string;
    projectId?: string;
    limit?: number;
    offset?: number;
  }): Promise<RunListResponse> => {
    const query = new URLSearchParams();
    if (params?.status) query.set('status', params.status);
    if (params?.projectId) query.set('project_id', params.projectId);
    if (params?.limit) query.set('limit', String(params.limit));
    if (params?.offset) query.set('offset', String(params.offset));

    const qs = query.toString();
    const endpoint = `/api/v1/runs${qs ? `?${qs}` : ''}`;
    return api.get<RunListResponse>(endpoint);
  },

  getRunStatus: async (runId: string): Promise<RunStatusResponse> => {
    return api.get<RunStatusResponse>(`/api/v1/runs/${encodeURIComponent(runId)}`);
  },

  getRunEvents: async (runId: string): Promise<RunEventsResponse> => {
    return api.get<RunEventsResponse>(`/api/v1/runs/${encodeURIComponent(runId)}/events`);
  },

  resumeRun: async (
    runId: string,
    decision: ResumeRunRequest
  ): Promise<RunStatusResponse> => {
    return api.post<RunStatusResponse>(
      `/api/v1/runs/${encodeURIComponent(runId)}/resume`,
      decision
    );
  },

  cancelRun: async (
    runId: string,
    request: CancelRunRequest = {}
  ): Promise<RunStatusResponse> => {
    return api.post<RunStatusResponse>(
      `/api/v1/runs/${encodeURIComponent(runId)}/cancel`,
      request
    );
  },

  publishPR: async (
    runId: string,
    request: PublishPRRequest
  ): Promise<PublishPRResponse> => {
    return api.post<PublishPRResponse>(
      `/api/v1/runs/${encodeURIComponent(runId)}/publish-pr`,
      request
    );
  },
};
