# Native PAIR paired-node inventory

`pair_devices()` retains configured direct LM Studio/Unsloth devices and additionally reports native NVIDIA PAIR paired nodes by physical UUID. `pair_list()` exposes native entries separately in `cluster_models` and `cluster_inventory`; existing routing IDs and selection rules are preserved. `pair_list(device="pair-<uuid>")` returns the native node inventory.

A single process-local mDNS browser automatically observes `_nvpair-node._tcp.local.` on IPv4 interfaces. New and changed devices are picked up from add/update events, removed discoveries disappear, and current advertised ni/em ports supersede roster defaults. The roster remains a fallback. Model APIs are refreshed on every catalog inspection; no per-device manual configuration is required for this inventory. Unpaired peers remain visible; remote model inventory requires an existing exact trusted pin and never falls back to plaintext HTTP. No automatic trust changes or arbitrary port scanning occur. Native IDs cannot be used for Bridge loading, unloading, jobs or direct inference. Models have explicit management_supported=false and inference_supported=false. Native PAIR currently publishes IPv4 services; IPv6-only discovery is unsupported. Unknown model type/context remain unknown; loaded state requires per-engine attribution. Offline nodes remain visible. Configured direct engines remain authoritative for their own device inventory.

The fallback roster is read from the existing NVIDIA PAIR app cluster directory; `PAIR_CLUSTER_DIR` can supply an absolute override. Windows uses LOCALAPPDATA/Nvidia Corporation/Personal AI Router/cluster. Native node-info and model service defaults are 14318 and 14322. Custom ports require a cluster-directory `bridge-inventory-ports.json` object with integer `node_info` and `models` values. No port scanning or new broker is performed. A stale roster address or changed service port is reported as unavailable, not silently guessed.

Remote model inventory authenticates with the existing node.crt/node.key and the trusted peer certificate. The exact server DER pin is checked before sending HTTP. Certificates/keys and endpoint addresses are excluded from returned inventory records. Native metadata and response reads are capped at 1 MiB, each connection has a six-second timeout, and new probes stop after an eighteen-second inventory deadline. This bounds new probes; it is not a hard whole-body deadline against a continuously trickling response.


## Router and network recovery

Read-only `pair_list()` continues with configured and native inventories when the router catalog returns an error. It reports sanitized router status and HTTP code; this does not retry inference or change routing. Configured device endpoints must address their own engine, not the multi-host NVIDIA ingress. Authentication-required engines are reported with unknown online state rather than incorrectly offline.

Zeroconf and its browser are created in a private worker so synchronous MCP requests cannot block their event loop. A dedicated ephemeral query socket sends multicast DNS-SD queries with standard QU replies, avoiding shared UDP5353 contention with native scanner and other MCP processes. Discovery is refreshed through queries, not passive unsolicited multicast announcements. Failed initialization retries after a 30-second cooldown; an active browser is refreshed at that interval to enumerate changed interfaces. Up to four validated advertised addresses are tried within the existing inventory deadline, with UUID identity checks before model access. Malformed roster addresses remain visible with metadata diagnostics.



Stock dotted-instance and duplicate-local SRV compatibility is covered by real DNS decoder/cache fixtures. Bridge requires no native binary patch for these announcements. See [portable setup](BRIDGE_SETUP.md) for explicit authentication and recovery.
