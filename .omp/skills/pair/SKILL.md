---
name: pair
description: Inspect and consult installed PAIR, LM Studio, and Unsloth Studio models through the PAIR Bridge MCP server in Oh My Pi.
---

# PAIR in Oh My Pi

Use `pair_devices` and `pair_list` for a bare inspection. For a task, call
`pair_smart_ask` with a bounded prompt and optional exact device/model. It uses
live installed inventory; it does not download weights. `pair_compare` asks two
explicit targets sequentially. `pair_diagnose` is read-only.

Preserve pre-existing loaded instances. The smart tool only attempts to unload
an instance that its own call loaded. A timeout can leave an upstream request
running; inspect state before retrying. Local model answers have no tools and
must be verified before acting on them.

For Unsloth, load models through PAIR Bridge so memory preflight runs. Unsloth
may not estimate every Transformers model; report unknown capacity accurately.
Configured `auto_unload_models` entries authorize exact model unloads after a
confirmed shortage or a clear out-of-memory response.

`pair_decide` uses external TypeSafe AI Jev and must never be an implicit
fallback. Ask for explicit permission to send the specific state externally.
