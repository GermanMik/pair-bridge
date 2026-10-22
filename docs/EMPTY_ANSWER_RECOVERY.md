# Empty-answer recovery

## Runtime and deployment

`pair_capabilities` reports `bridge_version`, an import-time SHA-256 `source_build`, unlimited retry policy, and logging/reconfiguration tools. Compare the running MCP with the tested candidate before attributing its behavior to a release. Existing MCP processes must reconnect after updating source. A source edit alone is not activation.

## Diagnosis

Completed empty generations preserve safe `finish_reason`, token usage, reasoning presence and character count. Reasoning text is not returned as final text or persisted. `length` proves a generation limit was reached, but reasoning-only exhaustion is not asserted without supporting counters/deltas. Refusals, tool responses, malformed completions and interrupted streams are separate errors. Native LM Studio endings retain reported reasons; missing reasons stay `unknown`.

## Recovery states

| Stage | Behavior |
| --- | --- |
| `retrying_empty_answer` | Same request/instance, jittered exponential backoff capped at 30s; no attempt or overall-time limit |
| `waiting_for_configuration` | Empty `length`; no more generations until explicit changed settings are approved |
| `waiting_for_device` | Original router host/instance is unknown; no second generation until the user explicitly pins an exact configured loaded target |
| `waiting_for_resources` | Fresh measured RAM/VRAM would breach the 1 GiB reserve; read-only polling resumes the same request once capacity returns |
| `inference` / `generating` | Generation in progress; cancellation prevents future attempts but engine computation may outlive stream closure |

Every retry refreshes inventory and Bridge capacity. Unknown inventory/capacity for a configured target is exposed as a warning and permits work. A confirmed missing/replaced instance fails without loading or substitution. Router recovery waits for an explicit configured target before any further inference: call `pair_job_reconfigure(job_id, device="EXACT_DEVICE")` only after the user chooses it. The same model key must already be loaded there. Binding can be combined with approved token/thinking changes; it never loads or substitutes a model. Subsequent retries stay pinned to that chosen instance.

Synchronous `pair_ask`, `pair_smart_ask` and `pair_vision_ask` hand empty recovery to a tracked background job. A `recovering` response is pending, not a model answer. Smart asks retain a newly loaded instance on handoff for inspection; recovery never unloads an instance it inherited.

## Explicit settings and budgets

1. Poll `pair_job_status(job_id)` for evidence and recovery advice.
2. Inspect fresh `pair_model_capabilities` and ask about low context and generation budgets separately. Both have a 40000 target; a 40000 context includes input/reasoning and cannot guarantee 40000 output tokens.
3. Apply the user's choice with `pair_job_reconfigure(job_id, max_tokens=40000, enable_thinking=false, remember=true)` where supported. Omit unsupported/unwanted fields. Changing thinking requires a fresh Unsloth advertisement; request settings never reload context.
4. Reconfiguration checks the same instance, known ceilings, and measured prompt tokens from the previous attempt when available. Without a measured prompt count, remaining output stays unknown and the engine validates it.
5. `remember=true` stores only approved settings locally. Future jobs reuse them only with explicit `use_saved_profile=true`; profiles never silently override request-specific choices.

No-op exhausted settings are rejected. Waiting is not a retry-count limit. Use `pair_job_cancel` at any stage to stop future submissions. Ambiguous timeout/interrupted transport never causes an automatic duplicate generation.

## Logs and restart

`pair_logs` and non-overwriting `pair_logs_export` include safe per-attempt evidence and final outcomes, correlated by job ID. They exclude prompts, schema/image inputs, reasoning text, raw engine errors and credentials. Final answers may contain sensitive data. Jobs remain process-local; persisted recovery metadata can identify an interruption, but a restart does not replay an unpersisted prompt.

## Release verification

Run the full `server.py --self-test` in a dependency-complete environment permitting local loopback for MCP integration tests. Regressions cover unlimited transient empties, budget waiting/reconfiguration, cancellation, changed targets, unavailable telemetry, memory recovery, native/OpenAI finish reasons, and log privacy. The user's live-model matrix must use actual advertised IDs, preserve resident instances, record the candidate build/settings/usage/final text, and include short, larger-input and structured requests. A model substitution requires an explicit user choice recorded in task acceptance. Unavailable required models are an explicit release blocker rather than a synthetic PASS.

Native output shape and token counters follow the [LM Studio native chat reference](https://lmstudio.ai/docs/developer/rest/chat). Native `stats.input_tokens`, `stats.total_output_tokens` and `stats.reasoning_output_tokens` are retained as safe usage counts. An unreported finish reason remains unknown; measured exhaustion can still trigger configuration waiting without fabricating `stop` or `length`.
