import { useState, useEffect, useCallback } from 'react';
import { RunRecord } from '../types/telemetry';
import { runsApi } from '../api/runs';

export function useRunsList(initialParams?: {
  status?: string;
  projectId?: string;
  limit?: number;
}) {
  const [runs, setRuns] = useState<RunRecord[]>([]);
  const [total, setTotal] = useState<number>(0);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const [statusFilter, setStatusFilter] = useState<string | undefined>(
    initialParams?.status
  );
  const [projectFilter, setProjectFilter] = useState<string | undefined>(
    initialParams?.projectId
  );

  const fetchRuns = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await runsApi.listRuns({
        status: statusFilter || undefined,
        projectId: projectFilter || undefined,
        limit: initialParams?.limit || 50,
      });
      setRuns(data.runs || []);
      setTotal(data.total || 0);
    } catch (err: any) {
      setError(err.message || 'Failed to retrieve runs list');
    } finally {
      setLoading(false);
    }
  }, [statusFilter, projectFilter, initialParams?.limit]);

  useEffect(() => {
    fetchRuns();
  }, [fetchRuns]);

  return {
    runs,
    total,
    loading,
    error,
    statusFilter,
    setStatusFilter,
    projectFilter,
    setProjectFilter,
    refetch: fetchRuns,
  };
}
