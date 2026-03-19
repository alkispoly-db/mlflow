import { useMemo, useCallback } from 'react';
import { MetricViewType, AggregationType, TraceMetricKey } from '@databricks/web-shared/model-trace-explorer';
import { useTraceMetricsQuery } from './useTraceMetricsQuery';
import { formatTimestampForTraceMetrics, useTimestampValueMap } from '../utils/chartUtils';
import { useOverviewChartContext } from '../OverviewChartContext';

export interface TokenUsageChartDataPoint {
  name: string;
  inputTokens: number;
  outputTokens: number;
  cacheReadTokens: number;
  cacheCreationTokens: number;
  timestampMs: number;
}

export interface UseTraceTokenUsageChartDataResult {
  /** Processed chart data with all time buckets filled */
  chartData: TokenUsageChartDataPoint[];
  /** Total tokens (input + output) in the time range */
  totalTokens: number;
  /** Total input tokens in the time range */
  totalInputTokens: number;
  /** Total output tokens in the time range */
  totalOutputTokens: number;
  /** Total cache read tokens in the time range */
  totalCacheReadTokens: number;
  /** Total cache creation tokens in the time range */
  totalCacheCreationTokens: number;
  /** Whether data is currently being fetched */
  isLoading: boolean;
  /** Error if data fetching failed */
  error: unknown;
  /** Whether there are any data points */
  hasData: boolean;
}

/**
 * Custom hook that fetches and processes token usage chart data.
 * Encapsulates all data-fetching and processing logic for the token usage chart.
 * Uses OverviewChartContext to get chart props.
 *
 * @returns Processed chart data, loading state, and error state
 */
// Stable array reference — defined outside the hook so it never triggers re-renders.
const TOKEN_TIME_SERIES_METRIC_NAMES = [
  TraceMetricKey.INPUT_TOKENS,
  TraceMetricKey.OUTPUT_TOKENS,
  TraceMetricKey.CACHE_READ_INPUT_TOKENS,
  TraceMetricKey.CACHE_CREATION_INPUT_TOKENS,
];

export function useTraceTokenUsageChartData(): UseTraceTokenUsageChartDataResult {
  const { experimentIds, startTimeMs, endTimeMs, timeIntervalSeconds, timeBuckets, filters } =
    useOverviewChartContext();

  // Single multi-metric query replaces four separate per-metric queries (Q5–Q8).
  // The server returns one row per (time_bucket, metric_name) combination.
  const {
    data: tokenTimeSeriesData,
    isLoading: isLoadingTokens,
    error: tokenError,
  } = useTraceMetricsQuery({
    experimentIds,
    startTimeMs,
    endTimeMs,
    viewType: MetricViewType.TRACES,
    metricNames: TOKEN_TIME_SERIES_METRIC_NAMES,
    aggregations: [{ aggregation_type: AggregationType.SUM }],
    timeIntervalSeconds,
    filters,
  });

  // Fetch total tokens (without time bucketing) for the header
  const {
    data: totalTokensData,
    isLoading: isLoadingTotal,
    error: totalError,
  } = useTraceMetricsQuery({
    experimentIds,
    startTimeMs,
    endTimeMs,
    viewType: MetricViewType.TRACES,
    metricName: TraceMetricKey.TOTAL_TOKENS,
    aggregations: [{ aggregation_type: AggregationType.SUM }],
    filters,
  });

  const allDataPoints = useMemo(() => tokenTimeSeriesData?.data_points || [], [tokenTimeSeriesData?.data_points]);

  const inputDataPoints = useMemo(
    () => allDataPoints.filter((dp) => dp.metric_name === TraceMetricKey.INPUT_TOKENS),
    [allDataPoints],
  );
  const outputDataPoints = useMemo(
    () => allDataPoints.filter((dp) => dp.metric_name === TraceMetricKey.OUTPUT_TOKENS),
    [allDataPoints],
  );
  const cacheReadDataPoints = useMemo(
    () => allDataPoints.filter((dp) => dp.metric_name === TraceMetricKey.CACHE_READ_INPUT_TOKENS),
    [allDataPoints],
  );
  const cacheCreationDataPoints = useMemo(
    () => allDataPoints.filter((dp) => dp.metric_name === TraceMetricKey.CACHE_CREATION_INPUT_TOKENS),
    [allDataPoints],
  );

  const isLoading = isLoadingTokens || isLoadingTotal;
  const error = tokenError || totalError;

  // Extract total tokens from the response
  const totalTokens = totalTokensData?.data_points?.[0]?.values?.[AggregationType.SUM] || 0;

  // Calculate total input and output tokens from time-bucketed data
  const totalInputTokens = useMemo(
    () => inputDataPoints.reduce((sum, dp) => sum + (dp.values?.[AggregationType.SUM] || 0), 0),
    [inputDataPoints],
  );
  const totalOutputTokens = useMemo(
    () => outputDataPoints.reduce((sum, dp) => sum + (dp.values?.[AggregationType.SUM] || 0), 0),
    [outputDataPoints],
  );
  const totalCacheReadTokens = useMemo(
    () => cacheReadDataPoints.reduce((sum, dp) => sum + (dp.values?.[AggregationType.SUM] || 0), 0),
    [cacheReadDataPoints],
  );
  const totalCacheCreationTokens = useMemo(
    () => cacheCreationDataPoints.reduce((sum, dp) => sum + (dp.values?.[AggregationType.SUM] || 0), 0),
    [cacheCreationDataPoints],
  );

  // Create maps of tokens by timestamp using shared utility
  const sumExtractor = useCallback(
    (dp: { values?: Record<string, number> }) => dp.values?.[AggregationType.SUM] || 0,
    [],
  );
  const inputTokensMap = useTimestampValueMap(inputDataPoints, sumExtractor);
  const outputTokensMap = useTimestampValueMap(outputDataPoints, sumExtractor);
  const cacheReadTokensMap = useTimestampValueMap(cacheReadDataPoints, sumExtractor);
  const cacheCreationTokensMap = useTimestampValueMap(cacheCreationDataPoints, sumExtractor);

  // Prepare chart data - fill in all time buckets with 0 for missing data
  const chartData = useMemo(() => {
    return timeBuckets.map((timestampMs) => ({
      name: formatTimestampForTraceMetrics(timestampMs, timeIntervalSeconds),
      inputTokens: inputTokensMap.get(timestampMs) || 0,
      outputTokens: outputTokensMap.get(timestampMs) || 0,
      cacheReadTokens: cacheReadTokensMap.get(timestampMs) || 0,
      cacheCreationTokens: cacheCreationTokensMap.get(timestampMs) || 0,
      timestampMs,
    }));
  }, [timeBuckets, inputTokensMap, outputTokensMap, cacheReadTokensMap, cacheCreationTokensMap, timeIntervalSeconds]);

  return {
    chartData,
    totalTokens,
    totalInputTokens,
    totalOutputTokens,
    totalCacheReadTokens,
    totalCacheCreationTokens,
    isLoading,
    error,
    hasData: inputDataPoints.length > 0 || outputDataPoints.length > 0,
  };
}
