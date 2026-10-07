# Changelog

## 0.7.3

- Advertise cross-PC concurrency through `pair_capabilities` and MCP initialization instructions.
- Allow background jobs on distinct configured PCs to run concurrently while preserving per-device serialization.
- Document the start-all-before-poll workflow for compatible MCP clients in English and Russian.

## 0.7.2

- Add configured Unsloth Studio devices for model inventory, memory estimates, load/unload, direct chat, background jobs, and the Hermes gateway.
- Require Unsloth models to be loaded through PAIR Bridge before direct chat so memory preflight and the configured unload allowlist apply.
- Keep Unsloth memory estimates independent of the LM Studio `lms` CLI.
- Expand English and Russian setup, safety, and tool documentation for Unsloth Studio.
