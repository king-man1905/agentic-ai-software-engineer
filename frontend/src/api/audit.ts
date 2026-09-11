import { api } from './client';
import { AuditEventView } from '../types/api';

export const auditApi = {
  getEvents: async (): Promise<AuditEventView[]> => {
    return api.get<AuditEventView[]>('/api/v1/audit/events');
  },
};
