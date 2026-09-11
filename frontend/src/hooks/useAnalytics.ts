import { useState, useEffect, useCallback } from 'react';
import {
  AnalyticsOverview,
  QualityAnalytics,
  ModelAnalytics,
  FailureAnalytics,
} from '../types/telemetry';
import { analyticsApi } from '../api/analytics';

export function useAnalytics() {
  const [overview, setOverview] = useState<AnalyticsOverview | null>(null);
  const [quality, setQuality] = useState<QualityAnalytics | null>(null);
  const [models, setModels] = useState<ModelAnalytics | null>(null);
  const [failures, setFailures] = useState<FailureAnalytics | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const fetchAnalytics = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [o, q, m, f] = await Promise.all([
        analyticsApi.getOverview().catch(() => null),
        analyticsApi.getQuality().catch(() => null),
        analyticsApi.getModels().catch(() => null),
        analyticsApi.getFailures().catch(() => null),
      ]);
      if (o) setOverview(o);
      if (q) setQuality(q);
      if (m) setModels(m);
      if (f) setFailures(f);
    } catch (err: any) {
      setError(err.message || 'Failed to retrieve observability analytics');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchAnalytics();
  }, [fetchAnalytics]);

  return {
    overview,
    quality,
    models,
    failures,
    loading,
    error,
    refetch: fetchAnalytics,
  };
}
