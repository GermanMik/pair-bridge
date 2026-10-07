# Contributing to PAIR Bridge

Thanks for helping improve PAIR Bridge. Contributions can include focused bug fixes, tests, documentation, and reports from real PAIR or LM Studio setups.

- [Open issues](https://github.com/GermanMik/pair-bridge/issues)
- [Good first issues](https://github.com/GermanMik/pair-bridge/labels/good%20first%20issue)
- [Report a bug](https://github.com/GermanMik/pair-bridge/issues/new?template=bug_report.yml)
- [Request an improvement](https://github.com/GermanMik/pair-bridge/issues/new?template=feature_request.yml)

## Before you start

Search existing issues and pull requests first. For a substantial change, open an issue to agree on the behavior before investing in implementation. Small fixes and documentation corrections can go directly into a pull request.

PAIR Bridge is a local-first MCP server for compatible clients. Keep changes within the behavior documented in the README; do not imply support for an engine, client, or remote-management path that has not been implemented and verified.

## Development setup

Requirements: Python 3.11–3.14 and [uv](https://docs.astral.sh/uv/getting-started/installation/).

```sh
git clone https://github.com/GermanMik/pair-bridge.git
cd pair-bridge/plugins/pair-bridge
uv --version
uv run --locked --script ./scripts/server.py --self-test
```

The self-test is the same check used by CI. It exercises the MCP protocol and the bridge's unit tests without requiring a live PAIR router or model device. CI runs on Linux, macOS, and Windows; a passing local run does not replace device-specific checks for changes to real LM Studio or PAIR behavior.

## Where to make a change

The plugin implementation and its tests live in `plugins/pair-bridge/scripts/`:

- `server.py` — MCP tools, routing, validation, and request flow.
- `management.py` — configured LM Studio devices and model lifecycle operations.
- `telemetry.py`, `benchmarks.py`, `jobs.py`, `diagnostics.py` — resource samples, measured routing, background jobs, and sanitized request records.
- `download_review.py` — explicit model-download planning and safeguards.
- `jev.py` — Jev integration.
- `test_*.py` — regression and protocol tests.

User-facing setup and behavior belong in `README.md` and `README.ru.md`; operational details belong in `docs/`.

## Make a safe, reviewable change

- Keep each pull request focused and explain the user-visible behavior it changes.
- Add or update a regression test for behavior changes, including an error-path test where relevant.
- Preserve explicit user control: asking a model must not silently download weights or replace an explicitly named model. Automatic selection must pass memory preflight. Unload another instance only when its exact model key is in that device's `auto_unload_models` allowlist and preflight reports insufficient memory.
- Never commit API keys, device addresses that identify private networks, real prompts, personal logs, or machine-specific configuration. Use synthetic examples and redact diagnostics.
- If dependencies change, update the inline script metadata and lock file intentionally; CI uses `uv run --locked`.
- Update the English and Russian documentation when setup, safety behavior, or user-visible tools change.

## Pull request checklist

- [ ] The change has a clear scope and explains why it is needed.
- [ ] Tests cover the changed behavior and relevant failure cases.
- [ ] `uv run --locked --script ./scripts/server.py --self-test` passes.
- [ ] Documentation is updated in both languages when applicable.
- [ ] No credentials, private prompts, or unrelated machine data are included.
- [ ] Any real-device testing is described with the tested platform and versions; untested paths are called out.

A pull request does not need to claim that every device combination was tested. State exactly what you verified.

## Issues and first contributions

The [good first issue label](https://github.com/GermanMik/pair-bridge/labels/good%20first%20issue) is for self-contained tasks with a clear expected result and no prerequisite design decision. If there are no open issues with that label, propose a small task using the feature-request form or ask maintainers to split a larger issue. When reporting a bug, include the client, operating system, PAIR/LM Studio versions, relevant model/device state, steps to reproduce, expected and actual result, and sanitized error text. Never attach tokens or private prompts.

PAIR Bridge is an independent community project and is not affiliated with or endorsed by NVIDIA or OpenAI.
