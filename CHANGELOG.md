# Changelog

## 0.8.4

- Preserve per-attempt finish/usage/reasoning-presence evidence, distinguish tool/refusal/transport failures, and expose the running version and source build.
- Hand empty synchronous answers to cancellable jobs; retain unlimited retries, wait for explicit budget/thinking changes after exhaustion, and recheck target and memory before retries.
- Add `pair_job_reconfigure` with explicit local recovery profiles and prompt/context budget validation; export safe attempt evidence with final outcomes.

- Document NVIDIA PAIR 1.0 engine-specific OpenAI endpoints, including llama.cpp, and clarify that llama.cpp model lifecycle remains managed by PAIR.
- Retry confirmed empty final answers without a retry-count limit on the same request and model instance; expose retry progress and keep cancellation for background jobs.
- Add local answer/error logs with safe export, prefer resident models and then the latest successful model, and warn when Bridge cannot verify a cold-load memory estimate or fresh capacity.
- Require a measured 1 GiB reserve for system RAM and NVIDIA VRAM during cold-load preflight; keep model work available with a warning when capacity telemetry is unknown.

## 0.8.3

- Add advertised per-request Unsloth `enable_thinking` control to direct asks, smart asks, jobs and batches; omitted values preserve engine defaults.
- Report separate confirmation requests to increase generation `max_tokens` and loaded/planned `context_length` to 40000, respecting known model ceilings and requiring confirmation before changes or reloads.
- Validate thinking transport, reasoning-only empty streams, independent budget thresholds, and batch forwarding; all 8 targeted regression tests pass.

## 0.8.2

- Add nested, versioned bilingual smoke (6), short (20) and full (100) benchmark suites, background runs, metrics-only history, comparisons and routing eligibility for compatible completed short/full runs.
- Preserve exact loaded-instance targeting, cancellation, workload-aware comparisons, objective validators and no prompt/answer persistence; retain verified benchmark request and history fixes.
- Bound total model loading to 180 seconds by default, configurable per device from 1 to 3600 seconds; deadlines cover headers and body and release Bridge locks on timeout.
- Keep model weights installed through the engine: Bridge loads existing files into RAM/VRAM and rejects disabled download entry points before network or configuration access.
- On ambiguous load timeout, inspect engine state before retrying; the engine may continue loading.
## 0.8.0

- Add fresh per-model capability and context discovery, including explicit `supported`, `unsupported`, and `unknown` states.
- Add structured JSON output with local schema validation, plus process-local job and queue visibility.
- Add atomic multi-job batches with independent outcomes, and bounded embeddings and inline-image vision tools for exact already-loaded model instances.
- Remove the fixed 8192 output-token ceiling. Requests honor fresh model and loaded-instance limits without silent clamping; unknown limits remain delegated to the engine, which accounts for prompt and reasoning tokens.
- Expand English and Russian guides and plugin instructions for the new tools and output-budget behavior.
- Defer benchmark expansion; existing benchmark behavior is unchanged.
- Validation: 116 tests passed in the isolated release checkout.

## 0.7.4

- Advertise cross-PC concurrency through `pair_capabilities` and MCP initialization instructions.
- Allow background jobs on distinct configured PCs to run concurrently while preserving per-device serialization.
- Document the start-all-before-poll workflow for compatible MCP clients in English and Russian.
- Preserve `pair_smart_ask(device=...)` when selecting a model through memory preflight.
- Detect deferred Unsloth load errors and classify in-band out-of-memory failures before applying the configured unload allowlist.
- Report confirmed allowlisted unloads cumulatively across rejected smart-selection candidates, with device provenance.
- Recognize Unsloth's native GPU-fit shortage message in deferred successful-HTTP load responses.
- Preserve confirmed allowlisted unload records when a load retry or job preflight fails, including prompt-free job recovery journals.
- Require exact LM Studio load-response instance ownership before background job cleanup.
- Include confirmed releases in MCP-visible errors throughout synchronous preflight, load and verification failures.
- Recognize native Unsloth free-GPU-memory shortage guards during training and native audio placement, preserving noncapacity refusals.
- Include device provenance in confirmed-release error text so identical IDs on different PCs remain distinguishable.
- Preserve external Unsloth instances reused by already_loaded responses; reject unconfirmed load statuses before ownership.
- Attribute every smart-ask postselection release to its exact selected device before merging records across PCs.
- Omit unsolicited Unsloth quantization overrides, preserving native cold-load defaults and resident precision inheritance.
- Validation: 88 tests passed in the release checkout; 90 passed in the local checkout with existing desktop authentication.

## 0.7.2

- Add configured Unsloth Studio devices for model inventory, memory estimates, load/unload, direct chat, background jobs, and the Hermes gateway.
- Require Unsloth models to be loaded through PAIR Bridge before direct chat so memory preflight and the configured unload allowlist apply.
- Keep Unsloth memory estimates independent of the LM Studio `lms` CLI.
- Expand English and Russian setup, safety, and tool documentation for Unsloth Studio.

### Local paired-node visibility fix
- Automatically discover native NVIDIA PAIR devices over mDNS, refresh model inventories and use the paired roster as fallback alongside configured engines. Keep native IDs read-only and unknown/offline states explicit. See docs/CLUSTER_INVENTORY.md.
