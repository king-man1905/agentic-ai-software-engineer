import { useState, useEffect, useRef, useCallback } from 'react';
import { RunStatusResponse, TelemetryEvent } from '../types/api';
import { runsApi } from '../api/runs';

const TERMINAL_STATUSES = new Set(['COMPLETED', 'FAILED', 'CANCELLED', 'BLOCKED']);

export function useRunPolling(runId: string | null, intervalMs = 2000) {
  const [runStatus, setRunStatus] = useState<RunStatusResponse | null>(null);
  const [events, setEvents] = useState<TelemetryEvent[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [isPolling, setIsPolling] = useState<boolean>(false);

  const timerRef = useRef<any>(null);

  const fetchStatus = useCallback(async () => {
    if (!runId) return;
    try {
      const [statusRes, eventsRes] = await Promise.all([
        runsApi.getRunStatus(runId),
        runsApi.getRunEvents(runId).catch(() => ({ run_id: runId, events: [] })),
      ]);

      setRunStatus(statusRes);
      if (eventsRes?.events) {
        setEvents(eventsRes.events);
      }
      setError(null);

      if (TERMINAL_STATUSES.has(statusRes.status.toUpperCase())) {
        setIsPolling(false);
        if (timerRef.current) {
          clearInterval(timerRef.current);
          timerRef.current = null;
        }
      }
    } catch (err: any) {
      setError(err.message || 'Failed to poll run status');
    }
  }, [runId]);

  useEffect(() => {
    if (!runId) {
      setRunStatus(null);
      setEvents([]);
      setIsPolling(false);
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
      return;
    }

    setLoading(true);
    fetchStatus().finally(() => setLoading(false));

    setIsPolling(true);
    timerRef.current = setInterval(fetchStatus, intervalMs);

    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
    };
  }, [runId, intervalMs, fetchStatus]);

  const stopPolling = useCallback(() => {
    setIsPolling(false);
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  return {
    runStatus,
    events,
    loading,
    error,
    isPolling,
    refetch: fetchStatus,
    stopPolling,
  };
}
