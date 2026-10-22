# Model result and error logs

PAIR Bridge keeps a separate `outcomes.jsonl` file in the user's local Bridge cache. It records completed chat answers and structured results, the number of `empty_answer` retries, or a sanitized error code and description. Background jobs write a record when they complete, fail, or are cancelled.

The log never includes prompts, schemas, images, embeddings, credentials, authentication responses, or raw engine error bodies. Final answers may still contain sensitive information. The `pair_logs` tool returns recent records locally:

```text
pair_logs(limit=50)
```

To make a separate file for support or analysis, choose a new absolute `.jsonl` path in an existing directory:

```text
pair_logs_export(output_path="C:\\Users\\<user>\\Desktop\\pair-bridge-logs.jsonl", limit=200)
```

Export refuses to overwrite an existing file. It does not upload or send the export. Share it only after reviewing its answer text.

`pair_smart_ask` uses an eligible already-loaded model first. When no suitable model is loaded, it tries the most recently successful model before normal ranking. Explicit model and device choices remain authoritative.

If a chat completion returns no final text (`empty_answer`), recovery has no attempt limit. Synchronous asks return `status=recovering` and a `job_id` after the first empty completion; poll `pair_job_status` and cancel with `pair_job_cancel`. The same loaded instance and prompt are preserved. Exponential backoff includes jitter and is capped at 30 seconds. Each retry rechecks Bridge inventory and capacity; a measured reserve shortage waits for resources and unknown capacity permits work with warnings.

An empty generation ending in `length` waits at `waiting_for_configuration` rather than resending the exhausted budget. Use `pair_job_reconfigure` with explicitly approved `max_tokens` or supported `enable_thinking` settings. Retry eligibility remains unlimited. `remember=true` stores only approved token/thinking values for the exact device/model; `use_saved_profile=true` explicitly reuses that profile. Context reloads are separate authorized operations. HTTP errors, ambiguous timeouts, interrupted streams, refusals, tool calls and invalid structured output are not retried as empty answers.

Attempt records include a correlated job ID, attempt number, sanitized usage, finish reason, reasoning presence/count (never its text), and applied generation settings. `pair_logs` and `pair_logs_export` include these records as well as final outcomes. Logs do not retain a replayable prompt: after MCP restart, interrupted jobs require inspection and an explicit new submission.

See [EMPTY_ANSWER_RECOVERY.md](EMPTY_ANSWER_RECOVERY.md) for the recovery contract and verification.
