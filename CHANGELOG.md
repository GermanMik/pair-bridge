# Changelog

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
