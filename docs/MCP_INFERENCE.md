# Model capabilities and bounded inference

PAIR Bridge exposes seven additional tools: `pair_model_capabilities`, `pair_job_list`, `pair_batch_start`, `pair_batch_status`, `pair_batch_cancel`, `pair_embeddings`, `pair_vision_ask`.

## Capabilities

`pair_model_capabilities(device, model)` reads fresh native inventory and returns `checked_at`, engine, exact model key, maximum context, actual loaded contexts and individual capabilities with `status` and `source`. Values are `supported`, `unsupported`, or `unknown`. Explicit engine type confirms chat/embedding classification; Unsloth name hints do not. Only explicit Boolean metadata establishes vision/tool training/JSON Schema support. Inventory metadata does not prove inference will succeed. No model is loaded or inference probe performed.

[LM Studio native inventory](https://lmstudio.ai/docs/developer/rest/list) currently reports vision and tool training. Its JSON Schema endpoint support does not establish the quality or compatibility of every model; missing model metadata remains unknown. Unsloth metadata may omit those fields; do not infer vision from a model name.

## Structured responses

`max_tokens` defaults to 2048 and has a minimum of 32, with no fixed bridge maximum. Requests use fresh explicit `max_output_tokens`, model `max_context_length`, and actual loaded instance context/output ceilings. Cold smart/job loads also check their requested context before preflight/load. Requests above a known ceiling fail without clamping. Missing metadata remains `unknown`, delegated to the engine. `output_budget` reports numeric ceilings and their sources in capabilities, answers and job metadata. The engine validates prompt plus completion (including reasoning) against the remaining context; the reported ceiling is not available-token accounting. Existing loaded contexts are preserved. The 48000-character retained-answer/structured-validation bound remains separate from generation tokens.

`pair_ask`, `pair_smart_ask`, `pair_job_start`, each batch member and `pair_vision_ask` accept either `json_schema` (the schema itself), or the OpenAI `response_format` envelope:

```json
{
  "device": "<configured device>",
  "model": "<installed model key>",
  "prompt": "Return whether the check passed as JSON.",
  "json_schema": {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}},
    "required": ["ok"],
    "additionalProperties": false
  }
}
```

An envelope uses `{"type":"json_schema","json_schema":{"name":"result","strict":true,"schema":{...}}}`; `{"type":"json_object"}` instead requires a JSON object. Supplying both arguments is an error. Schemas use draft 2020-12, at most 32 KiB for the envelope and 24 nesting levels; references (`$ref`, `$dynamicRef`, `$recursiveRef`) are deliberately unsupported. Inline definitions; nothing is resolved over the network. JSON Schema `format` annotations are not additional semantic validation.

Schema input is checked before model mutations. Explicit unsupported JSON Schema metadata rejects that request; unknown support permits an explicit attempt. The request uses the engine's [OpenAI-compatible JSON Schema API](https://lmstudio.ai/docs/developer/openai-compat/structured-output). Returned text must finish with `stop`, parse as strict finite JSON without duplicate keys, and satisfy the schema locally. Success adds `structured_output` and `structured_validation=passed`. Invalid, truncated or mismatched output fails without automatic retries or model substitutions; newly owned instances are retained for inspection on failure. Structured jobs use OpenAI SSE rather than native LM Studio chat, and require a finish reason plus `[DONE]`. Job status carries parsed JSON only after completion; schemas, prompts, parsed output and answers are never journalled.

## Job listing and batches

`pair_job_list(device?, include_terminal=false, limit=64)` returns metadata only, per-device queued/running job IDs, active count and queue wait seconds. No answers, prompts, partial text or structured output. Scope is the current MCP process; synchronous calls, other Bridge processes and external clients are not visible. Listing order is creation order; lock admission is not guaranteed FIFO. An empty list does not establish device idleness. Cross-process device locks still serialize requests, with the existing 30-second queue timeout.

`pair_batch_start(requests)` accepts 1–8 individual objects with `device`, `model`, `prompt`, and optional `context_length`, `max_tokens`, `unload_after`, `response_format`/`json_schema`. It validates the entire input and atomically reserves process capacity (eight active jobs) before starting any worker. Installed inventory and actual loading are checked independently inside each worker; a failed member does not cancel others. `pair_batch_status(batch_id)` returns each outcome; `pair_batch_cancel(batch_id)` requests cancellation of live members without changing terminal outcomes. All workers start before the batch returns. Use distinct physical PCs for parallel inference; one device remains serialized.

Batch IDs live in this process. Up to 64 recent groups and 512 recent jobs are retained in memory; an old job can fall back to prompt-free journal metadata, without its answer. Recovery never re-executes a task. Stream cancellation does not prove engine computation stopped. No automatic retry, download, alias bypass or change to existing ownership cleanup.

## Embeddings

`pair_embeddings(device, model, input, expected_dimensions?)` accepts a string or 1–32 strings, nonblank, at most 8192 characters each and 48000 total. These are character bounds, not tokenizer counts; the engine can impose tighter token limits. Fresh engine metadata must confirm an embedding model and exactly one instance must already be loaded. Use `pair_load` explicitly when needed; this tool never JIT-loads a missing instance, downloads weights or changes configuration.

The [embedding request](https://lmstudio.ai/docs/developer/openai-compat/embeddings) uses the exact instance (Unsloth uses its accepted model key), `encoding_format=float`, then checks every input has exactly one uniquely indexed vector, all components are finite numbers, and dimensions agree (1–16384). `expected_dimensions` validates; it never resizes vectors. Results contain vectors in input order, dimensions, count, provenance and usage. Inputs/vectors are not persisted. No router fallback. A reported embedding model does not establish the engine endpoint is implemented: an HTTP error is returned without retry.

## Vision

`pair_vision_ask(device, model, prompt, images, max_tokens=2048, response_format?, json_schema?)` requires an already loaded chat instance and fresh explicit `vision=true` metadata. Unknown support is rejected. `images` contains 1–4 inline `data:image/png;base64,...`, `data:image/jpeg;base64,...` or `data:image/webp;base64,...` strings, at most 4 MiB decoded per image and 12 MiB total. Base64 and image headers/containers are checked locally (PNG dimensions <=8192 per side); complete pixel decoding remains the engine's responsibility.

The tool sends OpenAI multimodal content (`text` and `image_url` parts) directly to the configured engine. It never reads filesystem paths or fetches remote image URLs. Image contents are not stored in diagnostics, journals or memory. Unsupported formats, unknown capabilities, ambiguous/unloaded instances and engine errors are reported without retry or fallback. Optional structured output uses the same local checks as text requests.

Benchmark improvements are deferred; existing benchmark checks and scoring are unchanged.
