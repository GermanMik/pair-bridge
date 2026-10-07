<div align="center">

# PAIR Bridge — Local Models for Compatible Tools

### One bridge. Many tools. Your local models.

**An open-source MCP bridge for compatible tools, NVIDIA PAIR, LM Studio, and Unsloth Studio.**<br>
Discover models across devices · ask for a second opinion · compare answers · manage memory deliberately.

[Get started](#get-started) · [See how MCP works](#what-does-mcp-actually-do) · [Explore the tools](#tool-reference) · [Русский](README.ru.md)

![Codex, Oh My Pi, and other compatible tools connect through PAIR Bridge to local models and devices](docs/assets/pair-bridge-architecture.svg)

[![Tests](https://github.com/GermanMik/pair-bridge/actions/workflows/test.yml/badge.svg)](https://github.com/GermanMik/pair-bridge/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-mint.svg)](LICENSE)
![MCP](https://img.shields.io/badge/MCP-bridge-6C63FF)
![Local-first](https://img.shields.io/badge/models-local--first-5A9E50)

</div>

**PAIR Bridge lets compatible MCP clients use installed local models across configured devices. Codex and Oh My Pi are supported examples. Requests can route through PAIR or target configured LM Studio and Unsloth Studio devices, including machines reachable over SSH/Tailscale.**

| Discover | Ask & compare | Manage safely |
| --- | --- | --- |
| See which devices are online and which models are installed or loaded. | Request a bounded second opinion or compare two model answers. | Estimate memory at the intended context; load or unload exact instances. |

Type `/pair` and choose the **pair** skill, or mention `$pair`. Astra, Sol, and other tool-capable Codex models can then use these tools:

> `/pair find an installed model on pc, estimate memory for 8192 context tokens, and ask it to review this function`

**Installed models first.** The bridge checks live inventory before asking or loading, so a missing model such as `gpt-oss-20b` is reported instead of being called blindly. Downloading new weights requires a separate, explicit plan and action.

Independent community project. Not affiliated with or endorsed by OpenAI or NVIDIA.

## Concurrent work across PCs

Codex, Claude and other compatible MCP clients can run separate model tasks simultaneously on different configured PCs. `pair_capabilities` exposes this as machine-readable data, and `pair_devices` includes the same capabilities. Each PC can receive its own individual prompt and model choice; a shared task is not required. Discover exact device/model IDs first.

1. Inspect `pair_capabilities`, `pair_devices`, and `pair_list(device=...)` for each target.
2. Start `pair_job_start(device="pc_a", model="<installed key on pc_a>", prompt="task A")`.
3. Start `pair_job_start(device="pc_b", model="<installed key on pc_b>", prompt="task B")` **before waiting for A**.
4. Keep both job IDs and poll `pair_job_status` independently in the same MCP session. Cancel only the intended job with `pair_job_cancel`.

Start calls may be sequential: jobs execute in background threads and overlap across devices. Up to eight active jobs per bridge process; operations on one device ID serialize across this OS user's bridge processes with a 30-second queue timeout. Use distinct physical PCs, not multiple aliases for one host. Router calls without an explicit device cannot guarantee different hosts. `pair_compare` remains sequential. This runs separate tasks, not one model distributed across PCs. A failed job does not cancel other jobs. Cancellation closes the bridge stream but engine computation may continue; inspect before retrying. After restart, journal metadata supports recovery, not recovery of the answer text.

For any MCP client that accepts stdio servers, use `uv` with arguments `run`, `--locked`, `--script`, and the **absolute path** to `plugins/pair-bridge/scripts/server.py`. A common MCP configuration shape is:

```json
{
  "mcpServers": {
    "pair-bridge": {
      "command": "uv",
      "args": ["run", "--locked", "--script", "C:/Develop/PAIR/pair-bridge-pr/plugins/pair-bridge/scripts/server.py"]
    }
  }
}
```

Adapt the path and enclosing configuration to your client. The server reads the existing `.pair-bridge.json` device configuration; no second PAIR broker is needed. Restart the MCP connection after updating server code so initialization instructions and tool schemas refresh. Already running plugin processes keep their old code until restarted.

## What does MCP actually do?

**MCP means Model Context Protocol.** It is the interface through which compatible clients discover and call tools. In this project, a small MCP server runs on your computer and exposes tools for discovery, model lifecycle, and inference.

Think of the workflow as four jobs:

| Component | Its job | Example |
| --- | --- | --- |
| **Compatible client** | Understand your task and use the returned answer. | “I need a second opinion on this function.” |
| **MCP bridge** | Expose discovery, lifecycle, and inference as tools. | Inspect a device, load a model, then call it. |
| **PAIR** | Route requests when no device is selected. | Send the request to a connected model server. |
| **Local model** | Generate an answer. | Return review comments to Codex. |

**MCP is the tool connection; PAIR is the model router.** The bridge does not turn a local model into the main Codex model. Codex continues coordinating your task, and the consulted model returns text for Codex to assess.

The diagram shows both supported paths: PAIR can route a request, or the bridge can target a configured local model engine directly.

## A real example

> **You:** “Use `pair_list`, then `pair_ask` to review this function for edge cases.”
>
> **Codex:** Reads the current model list, selects an appropriate chat model, and sends the relevant code through the bridge.
>
> **Local model:** Returns its review.
>
> **Codex:** Checks the suggestions against your code and explains which changes are worth making.

The consulted model receives the text passed to the tool. This bridge gives it no shell tools or direct access to your files.

## Get started

### 1 · Prepare your local models

You need:

- **Codex** with plugin marketplace support.
- **[uv](https://docs.astral.sh/uv/getting-started/installation/)** installed and available on your PATH.
- **PAIR running**, connected to at least one working chat model, for routed requests.
- **LM Studio 0.4+ or Unsloth Studio** with its API enabled on each device you want Codex to manage.
- **The LM Studio `lms` CLI** for LM Studio memory estimates. Unsloth uses its memory-estimate API when the selected model format is supported.
- **OpenSSH** and a working SSH config alias for remote loopback-only engine endpoints.

After installing uv, restart Codex so it can discover it. Confirm uv is available with `uv --version`.

### 2 · Install the plugin

Run these commands in a terminal:

```sh
codex plugin marketplace add GermanMik/pair-bridge
codex plugin add pair-bridge@pair-bridge
```

This adds our GitHub marketplace to Codex. It is a community catalog, separate from OpenAI's universal plugin directory. [About plugin marketplaces](https://developers.openai.com/plugins/build/plugins).

The first tool launch downloads the locked dependencies and, if needed, a compatible Python runtime. Subsequent launches reuse the cache.

### 3 · Start a new Codex task

Try the skill first:

> `/pair show my devices and available models`

Then ask it to operate on a device:

> `/pair load a suitable installed chat model on pc, ask it to check this function, then unload the instance you started`

You should get a model answer that Codex can use in the conversation. The model list alone does not confirm that every listed model can load successfully.

## Prompts to try

| Goal | Ask Codex |
| --- | --- |
| Inspect everything | “`/pair show configured devices and the PAIR routing catalog.`” |
| Review on one device | “`/pair use pc to review this code for bugs.`” |
| Get another approach | “`/pair ask a suitable available model for an alternative, then evaluate its answer.`” |
| Compare devices | “`/pair ask one model on mac and one on pc, then compare their answers.`” |

Model names differ between installations. Let `/pair` inspect the live catalog before it selects one.

## Connect your PAIR router

The default address is **`http://127.0.0.1:1234/v1`**. This is the PAIR proxy in the tested setup. Check the endpoint displayed by your PAIR installation if it uses a different port.

To change it, create **`.pair-bridge.json` in your home directory**:

```json
{
  "base_url": "http://127.0.0.1:1234/v1"
}
```

| System | Configuration file |
| --- | --- |
| macOS / Linux | `~/.pair-bridge.json` |
| Windows | `%USERPROFILE%\.pair-bridge.json` |

Use your **PAIR router's endpoint**. An individual model-engine endpoint only provides that server's models. `127.0.0.1` refers to the computer running the bridge.

<details>
<summary><strong>Environment variables and authentication</strong></summary>

`PAIR_BASE_URL` overrides the configuration file. Set it in the environment inherited by Codex.

If your endpoint requires a bearer token, set `PAIR_API_KEY` in that environment. Credentials must not be embedded in the URL or committed to the repository.

</details>

## Configure managed devices

Add LM Studio or Unsloth Studio API origins to the same configuration file. These are separate from the PAIR proxy URL:

```json
{
  "base_url": "http://127.0.0.1:1234/v1",
  "devices": [
    {"id": "mac", "engine": "lmstudio", "base_url": "http://127.0.0.1:1235"},
    {"id": "pc", "engine": "lmstudio", "base_url": "http://127.0.0.1:1235", "ssh_host": "my-pc",
     "auto_unload_models": ["publisher/model-to-release"]},
    {"id": "unsloth", "engine": "unsloth", "base_url": "http://127.0.0.1:8888",
     "api_key_env": "UNSLOTH_API_KEY", "auto_unload_models": ["publisher/model-to-release"]}
  ]
}
```

Use each engine's API origin; ports above are examples. Unsloth Studio's default port is `8888`. Set `api_key_env` only when its API requires a bearer token. `ssh_host` must be an existing OpenSSH config alias. The bridge opens a temporary loopback-only tunnel with host-key checking, runs no remote shell command, and closes the tunnel after the request. A direct HTTPS origin is also supported.

Each device may set `max_loaded_bytes` to a positive memory-estimate budget. Before a cold LM Studio load Bridge uses `lms`; for Unsloth it calls `/api/inference/estimate-memory`. A complete estimate is required for the configured budget check. Unsloth currently returns an unavailable estimate for model formats it cannot size, including many Transformers models. Bridge reports that as unknown. With a configured budget, unknown estimates block loading; without one, Bridge still checks fresh local or SSH RAM/NVIDIA VRAM telemetry when it can compare a known estimate.

By default, `auto_unload_models` is empty and Bridge does not free other loaded models. Add exact model keys to that per-device list to authorize unloading them when preflight proves a shortage. If an engine reports a clear out-of-memory error while the estimate is unknown, Bridge can also unload only listed models, verify each unload, and retry the load. The target model is never unloaded to make room for itself. Every load result reports instances released by this setting. `pair_smart_ask` ranks installed models and chooses the first one that passes memory preflight; an explicitly requested model is never replaced. Its response includes rejected candidates and reasons. Calls wait up to 30 seconds in a per-device queue, and `/pair diagnose` shows queued/loading/inference stages while active. [LM Studio Idle TTL and Auto-Evict](https://lmstudio.ai/docs/developer/core/ttl-and-auto-evict) apply to LM Studio JIT loads according to server settings; Bridge does not change those settings.

Use `pair_memory_plan(device, model, context_length)` to inspect estimates without loading. The planned context matters: on Alfred, Qwen3.8 27B was estimated at 19.24 GiB for 8,192 tokens and 26.03 GiB for 65,536 tokens. Unsloth reports a loaded model's actual context when its API provides it. `size_bytes` describes model weights on disk, not total RAM/VRAM. `pair_devices` reports fresh available RAM, NVIDIA VRAM where present, and free disk space. [LM Studio CLI](https://lmstudio.ai/docs/cli/local-models/load), [LM Studio inventory](https://lmstudio.ai/docs/developer/rest/list), [Unsloth Studio API source](https://github.com/unslothai/unsloth/tree/main/studio).

### Plan memory before loading

For an installed but unloaded model, ask Codex to call `pair_memory_plan` with its exact device, model key and intended context:

```json
{"device":"pc","model":"qwen/qwen3.8-27b","context_length":8192}
```

The result reports the candidate estimate, estimates for already loaded instances at their actual contexts, `required_with_headroom_bytes`, and a fresh device-capacity sample. A cold load is blocked if the estimated additional need plus 10% exceeds measured available RAM or NVIDIA VRAM. You can also set `"max_loaded_bytes": 34359738368` (32 GiB) as an independent total-load policy. Unknown estimates or capacity are shown as unknown; a configured budget still fails closed when estimation is unavailable. Existing loaded models are reused without changing their context.

### Download a new model explicitly

Downloading is separate from asking or loading. Ask Codex to call `pair_download_plan` with an exact LM Studio catalog ID or a Hugging Face repository URL, a device, and a destination to review. For a Hugging Face GGUF repository, pass `quantization` when needed. For a catalog ID or ambiguous/unreachable metadata, also supply a reviewed `estimated_size_bytes`. The plan has a one-use ID valid for ten minutes; `pair_download(plan_id, confirm_model)` requires the exact model ID again, and `pair_download_status` follows the job. No download starts while creating the plan.

The plan can verify the size of **one unambiguous GGUF file** and its repository revision; it rechecks both before starting. That file size may differ from the complete LM Studio download. Bridge checks free space with 10% headroom only when the target device is local. The destination is supplied for review: LM Studio does not confirm that it is its configured model folder. For remote devices, verify both the actual folder and free space on that device before starting. [LM Studio download API](https://lmstudio.ai/docs/developer/rest/download).

Devices are configured explicitly. PAIR peer discovery does not grant model-management access. Ollama lifecycle management, model deletion, engine installation, Unsloth model downloads, and PAIR cluster administration are not implemented. The separate `pair_download` tool is LM Studio-only, requires a one-use `pair_download_plan` and the exact model ID repeated in `confirm_model`; it is never used by `pair_ask` or `pair_smart_ask`.

For an authenticated device, set `api_key_env` to the name of an environment variable containing its token and pass that variable to the MCP process through `.mcp.json` `env_vars`. Keep tokens out of configuration committed to Git and out of prompts.

## What stays local?

The bridge runs on your computer and sends requests to **your configured PAIR endpoint**. PAIR can route them to your connected model servers. The project maintainer receives no requests through this plugin.

**The complete Codex conversation is not necessarily local.** Model answers return to Codex and follow your Codex/OpenAI data settings. PAIR and model servers may also keep their own logs. [Read the privacy note](PRIVACY.md).

## Hermes Agent

PAIR Bridge includes an OpenAI-compatible gateway for Hermes. It lists PAIR router models and chat models from the `devices` configured in `~/.pair-bridge.json`. Use `device/<device-id>/<model-key>` to target a specific computer; router models keep their original PAIR IDs.

To start the gateway and register its provider automatically, run one command from the cloned repository root:

```sh
python3 plugins/pair-bridge/scripts/install_hermes.py
```

The installer writes settings through the Hermes CLI, leaves your current main model unchanged, and starts the gateway in the background. Then select PAIR Bridge with `hermes model` or `/model`. Hermes, `uv`, and an existing PAIR Bridge config must already be installed.

Manual gateway launch (if you are not using the installer):

```sh
uv run --script ./plugins/pair-bridge/scripts/hermes_proxy.py
```

By default, the gateway listens only on `127.0.0.1:8765`. Add a provider to `~/.hermes/config.yaml`, keeping any existing entries:

```yaml
providers:
  pair-bridge:
    api: http://127.0.0.1:8765/v1
    api_key: local
    transport: openai_chat
model:
  provider: pair-bridge
  default: device/pc/qwen/qwen3-8b
  base_url: http://127.0.0.1:8765/v1
  api_mode: chat_completions
```

Replace `pc` and the model key with IDs from your device configuration and `GET http://127.0.0.1:8765/v1/models`. Choose models with `hermes model` or `/model`. Device-qualified models go directly to that configured engine; unqualified IDs go through the PAIR router.

Device models must be installed in their configured engine, with its API reachable from the gateway host. Unsloth device requests require the model to be loaded through PAIR Bridge first, so memory preflight runs before inference. The gateway accepts text chat-completion requests and emits OpenAI-compatible SSE when Hermes requests streaming. It binds to loopback by default. For network access, set `HERMES_PAIR_HOST` and `HERMES_PAIR_API_KEY` and expose the port only on a trusted network. Set `PAIR_BASE_URL` and `PAIR_API_KEY` for the router, as with the MCP server.

## Troubleshooting

| What you see | What to check |
| --- | --- |
| Tools do not appear | Open a new task; verify the plugin is enabled and `uv` is on Codex's PATH. |
| Cannot reach PAIR | Start PAIR and check its endpoint against the configuration file. |
| A configured device is unreachable | Verify its configured engine API, port, and direct HTTPS or SSH connection. |
| A model is listed but fails | Inspect PAIR's job details and model-server logs. Catalog entries are not health checks. |
| A named model is not installed | Run `/pair` inventory and choose an exact installed key; the bridge does not download missing weights. |
| Memory preflight is unknown or blocks a load | For LM Studio, check `lms`; for Unsloth, check whether Studio can estimate the model format. Use the intended context and review `max_loaded_bytes`. |
| Download plan cannot verify size or destination | Downloads are LM Studio-only. Use an exact Hugging Face GGUF repository and quantization when available; otherwise provide a reviewed size estimate. Check the actual storage folder and remote free space yourself. |
| HTTP 400 or 500 | Check the exact model ID, model loading, memory availability, and server errors. |
| Another request is running | Wait for the current bridge call to finish. |
| Timeout | Check PAIR before retrying: the model job may still be running. |
| No final text | The model may have spent its output budget on reasoning; inspect the result before choosing a larger budget. |

## Slash command and MCP tools

`/pair` activates the skill that teaches Codex how to plan safe model operations. The names below are MCP tools used by that skill. They are not terminal commands.

Without `device`, `pair_list` and `pair_ask` use the PAIR router. With `device`, they target that configured engine directly. Direct inference requires exactly one loaded instance of the selected model key.

**`pair_ask`** — requires `model` (an exact ID from `pair_list`) and `prompt` (your question). Optional `max_tokens` defaults to `2048`.

Example arguments for `pair_ask` (replace the model ID):

```json
{
  "model": "<exact ID from pair_list>",
  "prompt": "Review this function for edge cases: ...",
  "max_tokens": 2048
}
```

## Tool reference

| Tool | Inputs | Returns |
| --- | --- | --- |
| `pair_devices` | None | Reachability, installed/loaded inventory, timestamped free RAM, NVIDIA VRAM and disk capacity for every configured device. |
| `pair_list` | Optional `device` | PAIR routing catalog, or native inventory for one device. If the router catalog is empty, it returns installed models from reachable configured devices with explicit device/load provenance. |
| `pair_load` | `device`, `model`, optional `context_length` | Reused or newly loaded instance and its exact instance ID. |
| `pair_memory_plan` | `device`, installed `model`, optional `context_length` | Read-only CLI estimate for a cold load, including already loaded instances at their actual contexts. |
| `pair_unload` | `device`, `instance_id` | Confirmation that one exact instance is no longer observed. |
| `pair_ask` | `model`, `prompt`, optional `device`, `max_tokens` | Answer, selected model, completion status, timing and usage when available. |
| `pair_smart_ask` | `prompt`; optional `model`, `device`, `task_hint` (`general`, `code`, `fast`, `long_context`, `analysis`), `context_length`, `max_tokens`, `unload_after` | Chooses from live installed device inventories, loads if needed, asks once and reports cleanup. No download or silent fallback. |
| `pair_benchmark` / `pair_benchmark_results` | Exact loaded device/model, profile and 3–12 cases with `prompt` and literal `expected_contains` | Measures pass count and latency. Recent results influence smart routing; only metrics are saved. Use representative, reviewed cases. |
| `pair_job_start` / `pair_job_status` / `pair_job_cancel` / `pair_job_recover` | Exact device/model and prompt to start; job ID thereafter | Bounded background request with loading, prompt, response and cleanup stages, partial text, cancellation and journal-backed recovery metadata. |
| `pair_compare` | `prompt`, two exact device/model pairs | Two sequential results with provenance and textual disagreements; Codex verifies claims against source. |
| `pair_diagnose` | None | Router/device health and recent local request stages, durations, and sanitized failure reasons; no prompts or tokens. `request_journal_status` reports `ok`, `not_created`, `permission_denied`, or `unavailable`; journal access errors do not abort diagnosis. |
| `pair_download_plan` / `pair_download` / `pair_download_status` | Exact model and destination; optional estimate for unknown sizes; one-use plan ID, repeated `confirm_model`; job ID | A Hugging Face repository link with an unambiguous GGUF file can provide independently checked file size/revision. Free space is checked with 10% headroom when the configured `models_path` matches the destination. Otherwise the actual storage path remains unverified. Catalog IDs require a caller-supplied estimate. LM Studio reports the job total only after starting. |
| `pair_decide` / `pair_score` | State, Choice options or ordered Score levels, `allow_external=true` | Optional typed evaluation from **external** TypeSafe AI Jev; requires `TYPESAFE_API_KEY`. |

The smart path requires at least one explicitly configured, online device. It preserves pre-existing loaded instances. An instance loaded for a successful smart request is unloaded by default; an inference error or timeout leaves it loaded for inspection. Other applications can use the same engine, so Bridge cannot guarantee an instance is idle outside its own calls. Set `unload_after=false` when sharing a model with other clients.

Oh My Pi users can use the same MCP server and a native `/pair` command. See the [OMP setup guide](docs/OMP.md). Jev is a separate cloud decision service, never a local chat fallback; the bridge sends no state to it without an explicit `allow_external=true` call. [TypeSafe API reference](https://docs.typesafe.ai/api).

<details>
<summary><strong>Limits and request behavior</strong></summary>

- Exact model IDs only; the catalog is refreshed before inference.
- Likely embedding and draft models are rejected for chat. Type hints are inferred from names.
- Load, unload, and inference operations are queued per configured device across this user's bridge processes. Router calls use a separate lock because the destination is unknown. Other applications are outside these limits.
- No automatic retries, fallback models, or downloads during a model request. The engine may load an already installed model and consume GPU/RAM.
- Default output budget: 2,048 tokens; allowed range: 32–8,192. Input: up to 48,000 characters. Model context limits still apply.
- Request timeout: 180 seconds. Cancellation or timeout does not guarantee cancellation of the upstream model job.
- Empty final answers are errors. Answers stopped by the output budget are marked as truncated.
- Unloading affects the exact configured model instance and can disrupt another application that uses it. The skill tracks task-owned loads and avoids unloading unrelated instances.
- The bridge does not change persistent engine settings or repair model catalogs. Explicit loads can set context length.

</details>

## Migrating from `codex-pair-bridge`

The marketplace and plugin IDs are now `pair-bridge`. Existing `~/.codex-pair-bridge.json` configuration remains readable; the new `~/.pair-bridge.json` takes precedence if both exist. The local diagnostic cache keeps its old directory name to preserve request history. For an existing installation, run:

```sh
codex plugin remove codex-pair-bridge@codex-pair-bridge
codex plugin marketplace remove codex-pair-bridge
codex plugin marketplace add GermanMik/pair-bridge
codex plugin add pair-bridge@pair-bridge
```

## Update the plugin

Ask Codex: “`/pair update the plugin on this computer`”. The skill checks the installed plugin and marketplace source, updates it with Codex's supported commands, and reports the installed version. For a Git marketplace, the commands are:

```sh
codex plugin marketplace upgrade pair-bridge
codex plugin add pair-bridge@pair-bridge
```

This updates the computer running Codex; the bridge does not connect to configured LM Studio devices to update them. For a local source, update the source checkout through its normal workflow. Open a new task after updating so Codex discovers the `/pair` skill and current MCP tools. Version 0.7.0 adds a one-command Hermes setup and OpenAI-compatible PAIR gateway with device-qualified model IDs. Version 0.6.2 exposes installed device models when the PAIR router catalog is empty. Version 0.6.1 fixed free-memory sampling when Bridge runs locally on Windows. Version 0.6.0 added measured device capacity, local benchmark-based routing, and cancellable background requests with recovery metadata.

## For contributors

See the [Contributor Guide](CONTRIBUTING.md) for setup, tests, project structure, and first contributions.

```sh
cd plugins/pair-bridge
uv run --locked --script ./scripts/server.py --self-test
```

The automated tests cover MCP initialization, argument validation, configuration, SSH tunnel planning, device inventory, lifecycle operations, errors, timeouts, locking and response parsing. CI runs on **macOS, Windows and Linux**. Real routed and device-targeted requests were tested on macOS and a remote Windows node for the earlier release; new workflows still require device-specific acceptance runs.

Dependencies are locked in `scripts/server.py.lock`. Update intentionally with `uv lock --script scripts/server.py`, then rerun tests.

<details>
<summary><strong>Uninstall</strong></summary>

```sh
codex plugin remove pair-bridge@pair-bridge
codex plugin marketplace remove pair-bridge
```

</details>

---

[Report an issue](https://github.com/GermanMik/pair-bridge/issues) · [Privacy](PRIVACY.md) · [Security](SECURITY.md) · [MIT license](LICENSE)
