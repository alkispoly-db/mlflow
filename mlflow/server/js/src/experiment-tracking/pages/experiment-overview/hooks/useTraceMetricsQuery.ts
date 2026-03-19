import { useQuery } from '../../../../common/utils/reactQueryHooks';
import { fetchOrFail, getAjaxUrl } from '../../../../common/utils/FetchUtils';
import { catchNetworkErrorIfExists } from '../../../utils/NetworkUtils';
import type { MetricViewType } from '@databricks/web-shared/model-trace-explorer';
import {
  type QueryTraceMetricsRequest,
  type QueryTraceMetricsResponse,
  type MetricAggregation,
} from '@databricks/web-shared/model-trace-explorer';

const TRACE_METRICS_QUERY_KEY = 'traceMetrics';

// Extends the shared request type with the new metric_names field not yet in the package.
type TraceMetricsRequestBody = QueryTraceMetricsRequest & { metric_names?: string[] };

/**
 * Query aggregated trace metrics for experiments
 */
async function queryTraceMetrics(params: TraceMetricsRequestBody): Promise<QueryTraceMetricsResponse> {
  return fetchOrFail(getAjaxUrl('ajax-api/3.0/mlflow/traces/metrics'), {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(params),
  })
    .then((res) => res.json())
    .catch(catchNetworkErrorIfExists);
}

interface UseTraceMetricsQueryParams {
  experimentIds: string[];
  startTimeMs?: number;
  endTimeMs?: number;
  viewType: MetricViewType;
  /** @deprecated Use metricNames instead. */
  metricName?: string;
  /** One or more metric names to query in a single request. Prefer this over metricName. */
  metricNames?: string[];
  aggregations: MetricAggregation[];
  /** Optional: Time interval for grouping. If not provided, no time grouping is applied. */
  timeIntervalSeconds?: number;
  /** Optional: Filter expressions to apply (e.g. `trace.status="ERROR"`) */
  filters?: string[];
  /** Optional: Dimensions to group metrics by (e.g. `assessment_name`) */
  dimensions?: string[];
  /** Optional: Whether the query is enabled. Defaults to true. */
  enabled?: boolean;
}

export function useTraceMetricsQuery({
  experimentIds,
  startTimeMs,
  endTimeMs,
  viewType,
  metricName,
  metricNames,
  aggregations,
  timeIntervalSeconds,
  filters,
  dimensions,
  enabled = true,
}: UseTraceMetricsQueryParams) {
  // metricNames takes precedence; fall back to the deprecated metricName.
  const resolvedMetricNames = metricNames ?? (metricName ? [metricName] : undefined);

  const queryParams: TraceMetricsRequestBody = {
    experiment_ids: experimentIds,
    view_type: viewType,
    metric_names: resolvedMetricNames,
    aggregations,
    time_interval_seconds: timeIntervalSeconds,
    start_time_ms: startTimeMs,
    end_time_ms: endTimeMs,
    filters,
    dimensions,
  };

  return useQuery({
    queryKey: [
      TRACE_METRICS_QUERY_KEY,
      experimentIds,
      startTimeMs,
      endTimeMs,
      viewType,
      resolvedMetricNames,
      aggregations,
      timeIntervalSeconds,
      filters,
      dimensions,
    ],
    queryFn: async () => {
      const response = await queryTraceMetrics(queryParams);
      return response;
    },
    enabled: experimentIds.length > 0 && !!resolvedMetricNames?.length && enabled,
    refetchOnWindowFocus: false,
  });
}
