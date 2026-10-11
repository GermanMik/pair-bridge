# Local model benchmarks

PAIR Bridge exposes benchmark data to all clients connected to its MCP server. No cloud service or model judge is required. A client can discover suites, start a run, poll/cancel its job, read persisted metrics and compare completed runs. This does not add endpoints to NVIDIA Personal AI Router's HTTP server.

## Suites

| Suite | Cases | EN / RU | Purpose |
| --- | ---: | --- | --- |
| `smoke` | 6 | 3 / 3 | Basic arithmetic, instruction following, strict JSON |
| `short` | 20 | 10 / 10 | Quick comparison across all five categories |
| `full` | 100 | 50 / 50 | Ten parameterized variants per category and language |

The cases of `smoke` are the first six of `short`; `short` is the first twenty of `full`. `pair_benchmark_suites()` returns version, SHA-256 fingerprint, category/language counts and limitations. Categories: arithmetic, logic, JSON, Python code comprehension, synthetic long-context retrieval (~10k characters). Cases are deterministic and objective. Code is never executed. These narrow tests do not establish general intelligence, generated-code correctness, tool use, maximum usable context, embeddings or vision quality. Changing any prompt, expected answer, order or validator requires a new suite version and invalidates the old fingerprint for routing.

## Consumer workflow

1. Discover `pair_benchmark_suites` and inspect `pair_devices` / `pair_list`.
2. Explicitly load the target if needed using existing PAIR management. Benchmark requires exactly one already-loaded chat instance with a known context. It never loads, unloads, changes settings or substitutes a model.
3. Call `pair_benchmark_run` with the exact configured device/model, suite, repeats (1–5, default 3) and `max_tokens` (minimum 32, default 256). There is no fixed Bridge maximum: fresh explicit model/output limits and the actual loaded context bound the request, without clamping. Unknown ceilings are delegated to the engine, which includes prompt and reasoning in remaining context accounting. Use a larger explicit budget for reasoning models.

```json
{"device":"<configured device>","model":"<exact installed model key>","suite":"short","repeats":3,"max_tokens":256}
```

4. The response contains `job_id`, `run_id`, suite/version and expected sample count. Poll `pair_job_status(job_id)` or `pair_benchmark_results(run_id=...)`. Cancel with `pair_job_cancel(job_id)`. Completed job `structured_output` contains the metrics, never model answers.
5. Compare completed runs using `pair_benchmark_compare(run_ids=[...])`. The response has structured `results` and `table_markdown`. For identical workload settings it compares different models/devices/engines; provenance explains hardware/configuration differences. Incompatible suite fingerprints, context, output budget, repeats or warmup settings cause an error. Failed, cancelled and partial runs cannot be compared.

`pair_benchmark_results` without `profile` returns persisted suite runs. Filters: `device`, `model`, `suite`, `run_id`; pagination: `limit` 1–200 (default 50), `offset` >=0. Response: `status`, `results`, `total`, `limit`, `offset`, `next_offset`. Empty selection: `status=not_measured`, `results=[]`. Each row includes:

- Identity: `run_id`, `job_id` when the worker starts, timestamps, status, suite/version/fingerprint and `workload_id`.
- Provenance: device, model, exact `instance_id`, engine, actual loaded context, whitelisted scalar load settings and generation parameters. Runtime version, hardware and quantization are currently `null` with `unknown_reasons`; no model-name inference. Loading time is `null` because the model is already loaded.
- Quality: `passed`, `total`, `pass_rate` (0–1), `by_category`, `by_language` (`ru`, `en`). Technical failures reduce the overall ratio; zero measured samples gives `null`, not 0%.
- Errors: `total`, `by_type` and `incorrect_answers`. Incorrect content is `mismatch`; invalid JSON/type/schema, non-stop finish and truncation are `format_error`. Technical errors include `timeout`, `transport_error`, `http_error`, `invalid_stream`, `incomplete_stream`, `empty_response`, `output_limit`.
- Speed: `median_latency_ms`, nearest-rank `p95_latency_ms`, `median_ttft_ms`, `median_tokens_per_second`. Each metric is `{value, samples, reason}`. TTFT measures the first nonempty final-content delta, excluding reasoning. Throughput is engine-reported `completion_tokens` / end-to-end request seconds, including prefill and possibly reasoning; it is **not pure decoder speed**. Missing usage yields `null` with a reason; no estimated token counts. Speed excludes technical failures and includes valid wrong answers/format failures.
- Per-case metrics: case ID, category, language, repeat, status, latency, TTFT and usage count; never prompt, expected answer or response text.

One warmup request is excluded, then cases run sequentially for each repeat, at temperature 0. The full device lock is held for the suite; each request has a configurable total deadline (180 seconds by default), queue admission a 30-second deadline. Eight active jobs share existing process capacity. A warmup or transport failure aborts without retry or issuing the next case, because engine computation may still be active. Partial metrics remain queryable. Cancellation closes Bridge's stream but does not prove that engine computation stopped.

## Persistence and routing

`pair_benchmark_run` accepts `request_timeout_seconds` (30–1200, default 180) and `enable_thinking` (null by default, preserving engine defaults). An explicit boolean thinking mode is supported only when Unsloth advertises this request field; unsupported engines fail before warmup. These are request-local controls, never persistent model settings. Both fields are recorded and participate in workload compatibility. Legacy rows mean timeout180 and engine-default thinking. A full run with thinking disabled does not validate a prior default-thinking run.

Metrics are stored in `benchmark-runs.sqlite3` inside the local OS-user `pair-bridge` cache. Completed/failed/cancelled runs survive restart. Nonterminal rows from another Bridge process/session are reported as `interrupted_or_other_process`, not completed. Job IDs retain their existing process/journal recovery semantics; `run_id` is the durable metric lookup key. Separate computers have separate local stores unless they query the same Bridge MCP service.

Malformed history rows are skipped. Results report `invalid_records` for the current page and `history_warning=invalid_records_skipped`; `total` and pagination count stored rows, including malformed ones. Valid routing evidence remains available.

Automatic routing uses only complete, recent (30 days) `short`/`full` results with the current fingerprint. It chooses one compatible workload from the newest eligible result and the newest complete row per device/model within it, without pooling different suite sizes or settings. Quality is profile-specific (`code`, `logic` for analysis, `long_context`) or overall. `smoke` cannot promote a model. If suite measurements are absent, legacy profile summaries remain available. Read failures in optional routing history do not block inference.

Existing `pair_benchmark(device, model, profile, cases)` remains unchanged (3–12 literal substring checks). `pair_benchmark_results(profile, context_length)` retains the routing-summary response; omit `profile` to access suite runs. Combining profile and suite-result filters is rejected.

The feature must be installed/refreshed through the normal plugin workflow before an already-running MCP service can advertise the new tools. Editing this source checkout does not update the installed plugin automatically.
