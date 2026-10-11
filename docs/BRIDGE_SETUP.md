# Portable Bridge setup and recovery

This setup changes only PAIR Bridge. It does not require modifying or replacing NVIDIA PAIR binaries. Install the Bridge version matching your package and its locked dependencies; restart the MCP process after changing Bridge code or configuration.

## Explicit engine access

Edit the existing user-home `.pair-bridge.json`, preserving other keys and devices. Use the API origin shown by your engine. Ports below are examples, not discovery defaults. Do not point a direct device at the multi-host PAIR ingress: its catalog can contain other computers' models.

```json
{"devices":[{"id":"local-lmstudio","engine":"lmstudio","base_url":"http://127.0.0.1:1234"}]}
```

For remote Unsloth, enable its OpenAI-compatible server and explicitly configure its advertised origin. Keyless LAN inference requires **Chat and inference**; **Full access** is limited to localhost. A device configured with `api_key_env` still requires that environment variable even when the engine has keyless access. Remove that field only when intentionally configuring keyless access. For authenticated access, use `api_key_env: "UNSLOTH_STUDIO_AUTH_TOKEN"` and set that variable before starting Codex; the packaged MCP configuration forwards this named variable. Custom variable names require adding the same name to the MCP `env_vars` list. Restart the MCP process after changing its environment. Never put the token in the config or chat.

For local Unsloth only, Bridge can exchange an **existing**, explicitly selected desktop credential for a short-lived token:

```json
{"devices":[{"id":"local-unsloth","engine":"unsloth","base_url":"http://127.0.0.1:8888","desktop_secret_file":"<absolute path to existing desktop credential>"}]}
```

Replace the placeholder with a real absolute path from your installation. Bridge does not search for, generate or copy credentials. This option rejects remote origins, SSH tunnels and combinations with `api_key_env`. The token stays in client memory, reaches asynchronous jobs and is not saved to config or journals. Redirects are not followed.

## Native device discovery

Keep NVIDIA PAIR running and computers awake on a reachable local network. Bridge uses a dedicated IPv4 mDNS QU query socket for `_nvpair-node._tcp.local.` and refreshes discovery. Tests using the real Zeroconf decoder/cache cover standard announcements and stock PAIR announcements with dotted instance labels and duplicated `.local` in the SRV host. These announcements do not require a native publisher patch to be read by Bridge.

The existing native roster remains a fallback. A publisher that sends no usable response cannot be repaired by Bridge; report this honestly and use the manufacturer's supported recovery. Remote model inventory requires an existing exact native peer certificate pin. Newly seen devices are not automatically trusted, and Bridge does not scan arbitrary ports. Native `pair-<uuid>` IDs provide read-only inventory; configure an explicit direct engine device to run jobs or manage models.

## Diagnosis

Run `pair_diagnose()` and `pair_devices()`. HTTP401/403 or missing authentication is reported as `online: null` with an authentication hint. It is not proof that the computer is offline. Connection failure has a separate status. Errors and discovery summaries exclude endpoint addresses, credentials and certificate bodies. Empty mDNS results alone do not identify a firewall fault. Check engine API availability, authentication mode, native pairing and wake state before changing network settings.

`pair_list()` can return configured/native inventories when the router catalog has HTTP502/503, with sanitized router status. This read-only fallback does not retry failed inference or silently select another model. Request budgets retain the explicit value, including 16384; no new 8192 clamp is introduced.

Project-specific operational scripts and previous repair evidence are not a general installer. This change was tested with isolated stdio/HTTP/DNS fixtures and regression tests on Windows; macOS/Linux clean-install acceptance is not claimed. Distribution and publication are separate steps.


## Model loading and device contention

Model loading has a total network deadline of 180 seconds, covering response headers and body even when progress arrives continuously. Set `load_timeout_seconds` on the explicit device to an integer 1..3600 when a model legitimately needs longer. This is separate from the HTTP read timeout and does not limit output tokens. The authenticated load connection preserves headers and refuses redirects; load responses are capped at 1 MiB.

A total deadline reports `load_timeout`, closes the Bridge request and releases its device lock. The engine may continue loading: inspect inventory before any new load or inference. Bridge does not automatically retry or unload a model after ambiguous timeout. A competing request reports `device_busy` after the queue wait. `pair_job_list` observes background jobs in its own process; synchronous smart requests and other processes can still own a lock. Existing hung requests in older MCP processes require their normal cancellation/restart; an updated source file cannot interrupt old running code. Never delete a live lock or bypass it with another device alias.


Bridge never downloads model files. Downloading obtains weights from a repository; loading starts already installed weights in RAM/VRAM. Download plan/start/status MCP tools are removed; compatibility entry points reject without network calls. Install files through the engine outside Bridge before asking Bridge to load them.
