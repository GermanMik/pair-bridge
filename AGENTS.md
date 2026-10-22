# PAIR Bridge project rules

- Preserve unlimited recovery attempts for completed empty model answers. Budget exhaustion waits for explicit reconfiguration; it is not a retry-count limit.
- Recovery must preserve the original prompt and loaded instance. Never replace a model, reload context, or replay an ambiguous timed-out request automatically.
- Check device/instance and RAM/VRAM through Bridge before each retry. Unknown telemetry permits work with a warning; a measured shortage waits to preserve a 1 GiB reserve. These checks cannot guarantee memory availability throughout an external engine's generation.
- Keep attempt diagnostics safe: finish reason, token counts, reasoning presence/count, approved parameters and job identity. Never persist prompts, reasoning text, credentials or raw engine error bodies. Model answer logs are explicitly local and can contain sensitive text.
- Source changes do not activate an existing MCP process. Verify the running `bridge_version` and `source_build` after reconnecting.
- Full unittest, the user's selected live-model matrix and a fresh independent verifier gate a release. An unavailable device is `UNKNOWN`, never a synthetic success. Task evidence lives under `.agent/tasks/`.
- Keep NVIDIA PAIR vendor code and engine lifecycle separate from Bridge updates. Preserve the existing loaded-model selection and explicit context/output-budget confirmation contracts.
