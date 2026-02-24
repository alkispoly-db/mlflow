# Plan: Rate-Limited Pipelined Execution for `mlflow.genai.evaluate`

## Status: COMPLETE

## Summary

Implemented decoupled, pipelined predict/score execution with independent rate limits
for `mlflow.genai.evaluate`. Items now flow through predict→score as a pipeline,
with scoring starting as soon as each prediction completes (no waiting for all predictions).

## Changes Made

### Phase 1: Rate Limiter Module (DONE)
- **NEW**: `mlflow/genai/evaluation/rate_limiter.py`
  - `RPSRateLimiter(requests_per_second, clock, sleep)` — thread-safe token bucket algorithm
    - Accepts injectable `clock` and `sleep` callables for deterministic testing
    - Epsilon tolerance (`1e-9`) in token comparison to prevent floating point drift
  - `NoOpRateLimiter` — zero-overhead passthrough when rate limiting is disabled (set rate to 0)

### Phase 2: Environment Variables (DONE)
- **MODIFIED**: `mlflow/environment_variables.py`
  - Added `MLFLOW_GENAI_EVAL_PREDICT_RATE_LIMIT` (float, default 20 rps)
  - Added `MLFLOW_GENAI_EVAL_SCORER_RATE_LIMIT` (float, default auto-derived)
    - Auto-derived as `predict_rate × num_scorers` to match pipeline throughput
    - Can be explicitly overridden via the environment variable
  - Rate limiting is ON by default; set to 0 to disable

### Phase 3: Pipelined Harness (DONE)
- **MODIFIED**: `mlflow/genai/evaluation/harness.py`
  - Replaced `_run_single()` with `_run_predict()` + `_run_score()`
  - Rewrote `run()` to use two independent `ThreadPoolExecutor` pools
  - Pipeline loop uses `concurrent.futures.wait(FIRST_COMPLETED)` to interleave
  - Rate limiters applied: predict_limiter in `_run_predict()`, scorer_limiter in `_compute_eval_scores()`
  - Progress bar ticks on score completion (end-to-end per item)
  - Thread-time split (predict % vs score %) logged at INFO level after pipeline completes

### Phase 4: Session Utils Update (DONE)
- **MODIFIED**: `mlflow/genai/evaluation/session_utils.py`
  - Added `scorer_rate_limiter` parameter to `evaluate_session_level_scorers()`
  - Rate limiter applied before each multi-turn scorer invocation

### Phase 5: Tests — No sleep() (DONE)
- **NEW**: `tests/genai/evaluate/test_rate_limiter.py` — 7 unit tests using `FakeClock`
  - Uses deterministic `FakeClock` (injected via constructor) — no `time.sleep()` calls
  - Tests: invalid rate, burst tokens, sleep-on-exhaustion, sustained rate, idle refill,
    partial refill, NoOp
- **MODIFIED**: `tests/genai/evaluate/test_evaluation.py` — 4 integration tests + 1 fix
  - `test_predict_rate_limiter_is_wired_to_predict_fn` — mocks `RPSRateLimiter.acquire` to verify wiring
  - `test_scorer_rate_limiter_is_wired_to_scorers` — mocks acquire to verify scorer wiring
  - `test_pipelining_scores_while_predicts_pending` — uses `threading.Event` gate pattern
  - `test_no_rate_limit_backward_compat` — verifies no-change default behavior
  - Fixed `test_max_scorer_workers_env_var` to find scorer pool by thread name prefix

### Phase 6: Auto-derive max_workers from rate limits (DONE)
- **MODIFIED**: `mlflow/genai/evaluation/harness.py`
  - When `MLFLOW_GENAI_EVAL_MAX_WORKERS` is not explicitly set by the user, derive it from
    rate limits: `min(50, max(10, int(max_rate * 5)))` (assumes ~5s LLM call latency)
  - Falls back to default of 10 when no rate limits are set either
  - `MLFLOW_GENAI_EVAL_MAX_WORKERS` env var still works as an override but is not documented/suggested

### Bug fix: Floating point precision in token bucket
- `0.1 * n` accumulated over multiple sleeps causes IEEE 754 drift (e.g. `0.6 - 0.5 = 0.0999...98`)
- Token comparison `>= 1.0` would fail for `0.9999999999999998`
- Fixed with epsilon: `>= 1.0 - 1e-9`

### Phase 7: Adaptive Rate Limiting & 429 Retry (DONE)
- **Goal**: Automatically detect 429 errors, retry with exponential backoff, and adaptively
  reduce the proactive rate limit to prevent future 429s.

#### rate_limiter.py
- Added `is_rate_limit_error(exc)` — heuristic classifier that checks: class name
  `RateLimitError`, `.status_code == 429`, `.response.status_code == 429`, and
  string fallback ("429" or "rate limit")
- Added `report_throttle()` / `report_success()` to `RateLimiter` ABC (default no-op)
- Added AIMD to `RPSRateLimiter` (gated by `adaptive=True`):
  - Multiplicative decrease: `rps *= 0.5`, floor at 1.0 rps
  - Cooldown: 5s between consecutive decreases to coalesce rapid signals
  - Additive increase: `rps += alpha / current_rps`, capped at initial rate
- Added `call_with_retry(fn, rate_limiter, max_retries, sleep)` — replaces
  `acquire(); fn()` pattern with retry loop for 429 errors
- Added `_EVAL_RETRY_ACTIVE` ContextVar + `eval_retry_context()` context manager +
  `is_eval_retry_active()` query function

#### environment_variables.py
- Changed `MLFLOW_GENAI_EVAL_PREDICT_RATE_LIMIT` type from `float` to `str`, default `"auto"`
  - `"auto"` → adaptive RPSRateLimiter starting at 10 rps
  - `<number>` → fixed-rate RPSRateLimiter (no AIMD)
  - `"0"` → NoOpRateLimiter
- Changed `MLFLOW_GENAI_EVAL_SCORER_RATE_LIMIT` type from `float` to `str`
- Added `MLFLOW_GENAI_EVAL_MAX_RETRIES` (int, default 3)

#### harness.py
- Added `_parse_rate_limit(raw)` → `(rps, adaptive)` to handle auto/number/0 parsing
- Updated `_make_rate_limiter` to accept `adaptive` flag
- Updated `_run_predict()`: wraps predict_fn call in `eval_retry_context()` +
  `call_with_retry()` instead of bare `rate_limiter.acquire(); fn()`
- Updated `_run_score()`: wraps scorer phase in `eval_retry_context()`
- Updated `_compute_eval_scores()`: uses `call_with_retry()` for each scorer
- All submit calls pass `max_retries` from `MLFLOW_GENAI_EVAL_MAX_RETRIES`

#### session_utils.py
- Updated `evaluate_session_level_scorers()` to accept `max_retries` param
- Inner `run_scorer()` uses `eval_retry_context()` + `call_with_retry()`

#### litellm_adapter.py
- `_get_litellm_retry_policy()` checks `is_eval_retry_active()` and sets
  `RateLimitErrorRetries=0` when True, so 429 errors bubble up to our retry layer
  for AIMD adaptation. Other transient retries (500, 503, timeout) are unaffected.

#### Tests
- `test_rate_limiter.py`: 39 tests (was 7) — added is_rate_limit_error parametrized,
  AIMD (throttle halves, floor, cooldown, success restores, adaptive=False no-op),
  call_with_retry (success, retry-on-429, non-429 propagation, exhausted retries,
  throttle+success reporting), eval_retry_context (set/reset, nesting),
  _parse_rate_limit parametrized, _make_rate_limiter adaptive
- `test_evaluation.py`: 81 tests (was 78) — added test_predict_retries_on_429,
  test_scorer_retries_on_429, test_adaptive_rate_reduces_on_429

### Phase 8: Connection Pool Scaling & Benchmark Fixes (DONE)
- **Problem**: Benchmark with 500 requests against Databricks hung with 0% progress.
  Root cause: all HTTP calls (predict, score, trace logging, assessment logging) share
  the same urllib3 connection pool (default 10 per host via `MLFLOW_HTTP_POOL_MAXSIZE`).
  With 20+ predict threads cycling connections continuously, score threads were completely
  starved — not just slow, but unable to make any progress.
- **Root cause detail**: HTTP sessions are cached via `@lru_cache` in
  `mlflow/utils/request_utils.py:_cached_get_request_session()`. The pool_maxsize is
  read at session creation time, not as a cache key. So setting
  `MLFLOW_HTTP_POOL_MAXSIZE` at runtime via `os.environ` has no effect on already-cached
  sessions — they keep using the original pool_maxsize=10. The first HTTP call
  (e.g., `mlflow.set_experiment()`) creates the session before `run()` ever executes.
- **Fix**: After setting the env var, call `_cached_get_request_session.cache_clear()`
  to invalidate the LRU-cached session. The next HTTP call creates a fresh session
  with the larger `pool_maxsize`, which all subsequent calls share.
- **MODIFIED**: `mlflow/genai/evaluation/harness.py`
  - Added connection pool auto-scaling block before creating thread pools
  - Added `_cached_get_request_session.cache_clear()` after setting env var
  - `_pool_size()` uses `rps * 2` multiplier (assuming ~2s avg LLM latency), capped at [10, 500]
  - Backpressure buffer set to `2 * score_workers`
- **MODIFIED**: `mlflow/environment_variables.py`
  - Added `MLFLOW_GENAI_EVAL_RATE_LIMIT_UPPER_MULTIPLIER` (float, default 5.0)
  - AIMD ceiling = `initial_rps * upper_multiplier`, making the max adaptive rate configurable
- **MODIFIED**: `benchmark_evaluate.py`
  - Debug logging writes to `/tmp/mlflow-eval-debug.log` (FileHandler) to avoid tqdm conflicts
  - 30-second heartbeat in pipeline loop via `progress_bar.write()` (in harness)

### Phase 9: Disable HTTP-layer 429 retries during evaluate + import deadlock fix (DONE)
- **Problem 1**: Predict threads hitting 429 on Databricks endpoints were invisible to
  the adaptive rate limiter. MLflow's `_retry_databricks_sdk_call_with_exponential_backoff`
  in `rest_utils.py` silently retried 429s (same problem as litellm for scorers).
- **Fix**: Added generic `_DISABLE_429_RETRY` ContextVar in `rest_utils.py` (no knowledge
  of evaluate). The retry wrapper excludes 429 from retry_codes when the flag is set.
  `eval_retry_context()` in `rate_limiter.py` now sets both `_EVAL_RETRY_ACTIVE` (litellm)
  and `_DISABLE_429_RETRY` (HTTP layer). Dependency direction: `genai.evaluation` → `utils`.
- **Problem 2**: Import lock deadlock — 19 predict threads and 12 scorer threads stuck on
  `importlib._bootstrap._get_module_lock`. Multiple worker threads simultaneously triggered
  lazy imports (`import litellm`, databricks SDK auth modules) and deadlocked on Python's
  per-module import lock.
- **Fix**: Added `_warmup_imports()` in harness that force-imports `litellm` and
  `databricks.sdk.WorkspaceClient` in the main thread before creating thread pools.
  Subsequent `import` statements in worker threads are instant `sys.modules` lookups.
- **MODIFIED**: `mlflow/utils/rest_utils.py`
  - Added `_DISABLE_429_RETRY` ContextVar + `disable_429_retry()` context manager +
    `is_429_retry_disabled()` query function
  - `_retry_databricks_sdk_call_with_exponential_backoff` excludes 429 from retry_codes
    when the flag is set
- **MODIFIED**: `mlflow/genai/evaluation/rate_limiter.py`
  - `eval_retry_context()` now also activates `disable_429_retry()` from rest_utils
- **MODIFIED**: `mlflow/genai/evaluation/harness.py`
  - Added `_warmup_imports()` — imports litellm + databricks SDK in main thread
  - Called before creating thread pools

### Phase 10: Code Review Cleanup (DONE)
- **Style guide fixes** (from `/review changes with the style guide`):
  - Tightened try-catch scope in `call_with_retry()` — moved `report_success()` to `else` clause
  - Parametrized `test_success_climbs_to_multiplier_ceiling` (was inline, now `@pytest.mark.parametrize`)
  - Trimmed redundant docstring on `_dump_eval_thread_stacks`
  - Replaced opaque `lambda: (_ for _ in ()).throw(...)` with named helper functions in tests
- **ALKIS review feedback**:
  - Removed `_dump_eval_thread_stacks`, `_EVAL_THREAD_PREFIXES`, and `MLFLOW_GENAI_EVAL_DUMP_THREAD_STACKS`
  - Moved `_AUTO_PREDICT_RPS` into `_parse_rate_limit` as local `auto_initial_rps`
  - Introduced named constants: `avg_llm_latency_secs` in `_pool_size`, `backpressure_multiplier`
    in `backpressure_buffer` (replacing magic number 2)
  - Moved litellm and databricks.sdk imports to top level with `try/except ImportError` guards,
    eliminating `_warmup_imports()` entirely
  - Fixed dependency direction in litellm adapter: introduced `_DISABLE_RATE_LIMIT_RETRIES`
    ContextVar locally in `litellm_adapter.py` with `disable_litellm_rate_limit_retries()` accessor.
    Removed `is_eval_retry_active()` and `_EVAL_RETRY_ACTIVE` from `rate_limiter.py`.
    `eval_retry_context()` now sets both the litellm flag and `disable_429_retry()`.
  - Simplified scorer rate limit env var docstring to say `auto` = auto-derived

### Phase 11: Decompose `_run_pipeline` into Predict/Score Classes (DONE)
- **Problem**: `_run_pipeline` (164 lines) mixed predict submission, score dispatch,
  and orchestration in one function with two interleaved dicts and a shared semaphore.
- **MODIFIED**: `mlflow/genai/evaluation/harness.py`
  - Added `_PredictSubmitter` class — owns submit thread, predict pool, backpressure
    semaphore, predict futures dict, and timing. Methods: `start()`, `join()`,
    `drain(block=)` → `list[Future]`, `owns(future)`, `on_complete(future)`,
    `release_slot()`, `check_error()`, properties `predict_times`, `pending_count`,
    `limiter`, `shutdown()`.
  - Added `_ScoreSubmitter` class — owns score pool, score futures dict, timing, and
    multi-turn scoring. Methods: `submit(idx)` → `Future`, `owns(future)`,
    `on_complete(future)`, `run_multi_turn(...)`, properties `score_times`, `pending_count`,
    `limiter`, `shutdown()`.
  - Each class creates its own `RateLimiter` and `ThreadPoolExecutor` internally from
    config params (`rps`, `adaptive`, `max_rps_multiplier`, `pool_workers`) — callers
    pass config, not pre-built objects.
  - Extracted `_Heartbeat` class for periodic pipeline logging.
  - Extracted `_get_scorer_rate_config()` and `_get_pool_sizes()` helpers.
  - Rewrote `_run_pipeline` as thin orchestrator: reads env vars, computes config,
    creates predictor/scorer, runs drain→wait→dispatch loop, then multi-turn phase.
    Pool shutdown via `predictor.shutdown()` / `scorer.shutdown()` in finally block.
  - `run()` simplified — no longer manages limiters, pools, retries, or pool shutdown.
  - Code review feedback: renamed `is_mine` → `owns`, changed `drain_into(pending)` →
    `drain(block=)` returning `list[Future]`, changed `submit(idx, pending)` →
    `submit(idx)` returning `Future`, added docstring to `drain`.
  - Moved rate limiter + pool creation from `run()` / `_run_pipeline` into the submitter
    classes. Exposed `limiter` property for heartbeat access and `shutdown()` method.
  - Added docstrings to both `__init__` methods (documenting all parameters) and to
    `_submit_all`, `on_complete`, `release_slot`, and `submit`. Removed unused
    `_ScoreSubmitter.owns()` (dead code — pipeline uses `predictor.owns()` only).
- **Signature preserved**: `_run_pipeline` keeps identical parameters and return type.
  All 121 tests pass unchanged.

## Test Results
- `tests/genai/evaluate/test_rate_limiter.py`: 40 passed (< 5s total)
- `tests/genai/evaluate/test_evaluation.py`: 81 passed (< 49s total)

## PR Stack (via `git stack`)

Recreated the 4-PR stack using `git-stack` (replacing manual `pr/1-*` through `pr/4-*` branches):

| Branch | PR | Title | Base |
|---|---|---|---|
| `stack/rate-limiter` | #20937 | Add rate limiter module and evaluation environment variables | `master` |
| `stack/disable-429` | #20938 | Disable downstream 429 retries during evaluate | `stack/rate-limiter` |
| `stack/harness-helpers` | #20939 | Add harness helpers and supporting infrastructure | `stack/disable-429` |
| `stack/pipeline` | #20940 | Add pipelined predict-score execution | `stack/harness-helpers` |

Previous PRs (#20929, #20930, #20931, #20933) were closed and their branches deleted.

### Phase 12: Code Review Cleanup (stack/pipeline) (DONE)
- **`_Heartbeat`**: Refactored to accept `_PredictSubmitter` and `_ScoreSubmitter` objects
  instead of individual `RateLimiter` instances. `tick()` now takes `(items_predicted, items_scored)`
  instead of a `stats` dict — it pulls `pending_count` and `limiter` from the submitters directly.
- **Removed redundant debug log**: Pipeline loop debug log was duplicating heartbeat info.
- **`wait()` timeout**: Replaced magic `15` with `heartbeat.interval` — the timeout exists to
  wake the loop for heartbeat ticks, so the constant should be defined in one place.
- **`_invoke_scorer()`**: Extracted from nested closure in `_compute_eval_scores` to module-level
  helper function. Used via `lambda` in `call_with_retry` to capture `scorer_func` and `eval_item`.

### Phase 13: Test Assertion Tightening (stack/pipeline) (DONE)
- **`test_predict_rate_limiter`**: Removed scorers, assert exactly 5 acquires (was `>= 5`)
- **`test_scorer_rate_limiter`**: Replaced local `s1`/`s2` with `always_pass`, assert exactly 6
  acquires (was `>= 6`), removed misleading "predict acquires" comment
- **`test_pipelining`**: Removed assertion message per style guide; kept human's fix for
  `current_predicts > 0` guard in signaling scorer
- **`test_backpressure`**: Changed `<=` to `==` — gate blocks scorers so `max_in_flight`
  is deterministically equal to `buffer`
- **`test_no_rate_limit_backward_compat`**: Added `monkeypatch.setenv` to explicitly disable
  both predict and scorer rate limits (was relying on unspecified default behavior)
- **`test_predict_retries_on_429`**: Switched from `nonlocal counter` to list-append approach,
  matching the pattern in `test_scorer_retries_on_429` for consistency
- **`AUTO_INITIAL_RPS`**: Extracted from local variable in `_parse_rate_limit` to module-level
  constant. Tests (`test_parse_rate_limit`, `test_adaptive_rate_reduces_on_429`) now reference
  the constant instead of hardcoded `10.0`.
- **`test_adaptive_rate_reduces_on_429`**: Switched from `nonlocal counter` to list-append
  approach, matching the pattern in the other retry tests.
- **`test_scorer_retries_on_429`**: Changed exception message from `"rate limit exceeded"` to
  `"429 Too Many Requests"` for consistency with predict retry test and to make it clear why
  `is_rate_limit_error()` classifies it as retryable (string fallback matches `"429"`).

### Phase 14: Size Thread Pools for AIMD Headroom (DONE)
- **Problem**: With `auto` rate limiting (default 10 rps), AIMD could ramp up to
  `10 * 5 = 50 rps`, but `_pool_size(10)` created only 20 threads. At ~2s avg latency
  that sustains at most 10 rps — the thread pool was the bottleneck, wasting AIMD gains.
- **Fix**: Pre-size pools with moderate headroom and lower the AIMD ceiling to match.
- **MODIFIED**: `mlflow/environment_variables.py`
  - Changed `MLFLOW_GENAI_EVAL_RATE_LIMIT_UPPER_MULTIPLIER` default from 5.0 to 2.0
  - AIMD ceiling now `10 * 2 = 20 rps` (default case)
- **MODIFIED**: `mlflow/genai/evaluation/harness.py`
  - `_pool_size(rps, max_rps_multiplier=1.0)`: new parameter sizes threads for peak
    rate (`rps * max_rps_multiplier * latency`), so pools can sustain the AIMD ceiling
  - `_get_pool_sizes(predict_rps, scorer_rps, max_rps_multiplier=1.0)`: threads
    multiplier through to `_pool_size`
  - `_run_pipeline`: computes `pool_multiplier = upper_multiplier if predict_adaptive else 1.0`
    so fixed-rate users get no change, adaptive users get properly sized pools
- **MODIFIED**: `tests/genai/evaluate/test_rate_limiter.py`
  - Added `test_pool_size` parametrized (7 cases: None, 0, floor, base, multiplied, medium, cap)
  - Added `test_pool_size_default_multiplier` (backward compat: multiplier defaults to 1.0)
  - Added `test_get_pool_sizes_with_multiplier` (multiplier flows to both pools)
  - Added `test_get_pool_sizes_max_workers_override` (env var still takes precedence)
- **Default behavior after change**: With `auto` = 10 rps, `upper_multiplier` = 2.0:
  predict pool = 40 threads (was 20), score pool = 40 threads (was 20).
  Fixed-rate users and `MAX_WORKERS` override: no change.

### Phase 15: Test Deflake + Progress Bar Fix (DONE)
- **Deflaked `test_backpressure_limits_in_flight_items`**: Removed `timeout=10` from
  `score_gate.wait()` — under system load the timeout could fire, allowing a scorer to
  complete and release a backpressure slot before the test read `max_in_flight`. Wrapped
  test in `try/finally` to always set `score_gate`.
- **Fixed progress bar for multi-turn-only evaluation**: When only session-level scorers
  are provided (no single-turn scorers), the progress bar showed `eval_items + sessions`
  (e.g. 200) instead of just sessions (100).
- **MODIFIED**: `mlflow/genai/evaluation/harness.py`
  - `run()`: Progress bar total is now `(len(eval_items) if single_turn_scorers else 0) + len(session_groups)`
  - `_run_pipeline()`: Guarded predict→score loop on `if single_turn_scorers:`.
    The `else` branch uses the `_PredictSubmitter` to run predictions in parallel
    (setting up traces via clone/link/create) without scoring or progress ticks.
    Multi-turn scoring ticks remain unconditional in `run_multi_turn()`.
- **MODIFIED**: `tests/genai/evaluate/test_evaluation.py`
  - `test_backpressure_limits_in_flight_items`: Removed `score_gate.wait(timeout=10)`,
    added `try/finally` to always set `score_gate`.
