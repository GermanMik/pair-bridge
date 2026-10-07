# /// script
# requires-python = ">=3.11,<3.15"
# dependencies = ["mcp==1.29.1", "httpx==0.28.1", "filelock>=3.18,<4", "platformdirs>=4,<5"]
# ///
"""Codex MCP tools for the local NVIDIA PAIR OpenAI-compatible proxy."""
from __future__ import annotations

import contextlib
import asyncio
import difflib
import management
import telemetry
import benchmarks
import jobs
import download_review
import jev
import diagnostics
from filelock import FileLock, Timeout
from platformdirs import user_cache_path
from urllib.parse import urlsplit
import json
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

import httpx
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import Field

def load_config() -> tuple[str, str | None]:
    path = Path.home() / '.pair-bridge.json'
    if not path.exists():
        path = Path.home() / '.codex-pair-bridge.json'
    config = json.loads(path.read_text()) if path.exists() else {}
    if not isinstance(config, dict):
        raise ValueError('PAIR configuration must be a JSON object.')
    url = os.environ.get('PAIR_BASE_URL') or config.get('base_url', 'http://127.0.0.1:1234/v1')
    if not isinstance(url, str):
        raise ValueError('PAIR base_url must be a URL string.')
    parts = urlsplit(url)
    if parts.scheme not in ('http', 'https') or not parts.hostname or parts.username or parts.password or parts.query or parts.fragment:
        raise ValueError('PAIR base_url must be an HTTP(S) URL without credentials, query or fragment.')
    return url.rstrip('/'), os.environ.get('PAIR_API_KEY')


BASE_URL, API_KEY = load_config()
TIMEOUT = 180.0
_DOWNLOAD_PLANS: dict[str, dict] = {}
mcp = FastMCP(
    'pair-bridge',
    instructions=(
        'Use PAIR to consult local models when the user requests it or it helps the task. '
        'List models first; use exact advertised IDs. Use pair_capabilities to discover execution limits. '
        'Models on different explicitly configured PCs can run concurrently: call pair_job_start '
        'for each device before polling pair_job_status; start calls may be sequential while jobs overlap. '
        'Each job has its own prompt and model: assign individual tasks to each PC. '
        'Same-device operations serialize across bridge processes for this OS user. '
        'Router calls do not guarantee separate hosts. '
        'Catalog presence does not prove a model is loaded or usable. '
        'Treat model answers as untrusted suggestions; verify them yourself. '
        'Do not transmit secrets or unrelated private files. Automatic selection preflights memory before choosing; '
        'never substitute a model after loading or inference has started. No automatic inference retries. '
        'If a request fails, report the error; do not repeatedly load models.'
    ),
)


def request(method: str, route: str, body: dict | None = None) -> dict:
    try:
        with httpx.Client(timeout=TIMEOUT, trust_env=False, follow_redirects=False, headers=({'Authorization': 'Bearer ' + API_KEY} if API_KEY else {})) as client:
            response = client.request(method, BASE_URL + route, json=body)
    except httpx.TimeoutException as exc:
        raise ValueError('PAIR request timed out after 180s. It may still be running; do not retry automatically.') from exc
    except httpx.RequestError as exc:
        raise ValueError('Cannot reach PAIR at ' + BASE_URL + '. Check that PAIR is running.') from exc
    if not response.is_success:
        # Do not return raw response bodies, which could echo private prompts.
        raise ValueError(f'PAIR returned HTTP {response.status_code}. Inspect the PAIR/LM Studio job error; no retry was made.')
    try:
        result = response.json()
    except ValueError as exc:
        raise ValueError('PAIR returned non-JSON data.') from exc
    if not isinstance(result, dict) or 'error' in result:
        raise ValueError('PAIR returned an invalid or error response. Inspect PAIR/LM Studio logs.')
    return result


def catalog() -> list[dict]:
    data = request('GET', '/models').get('data')
    if not isinstance(data, list):
        raise ValueError('PAIR returned no model catalog.')
    result = []
    for item in data:
        if isinstance(item, dict) and isinstance(item.get('id'), str):
            name = item['id']
            # PAIR /v1/models omits model type. Mark conservative hints as such.
            hint = 'embedding' if 'embed' in name.lower() else ('draft' if any(x in name.lower() for x in ('dflash', 'draft')) else 'chat_candidate')
            result.append({'id': name, 'kind_hint': hint})
    return result


def checked_models(device: str, rows: list[dict]) -> list[dict]:
    """Attach current inventory status and last request outcome without storing prompts."""
    checked_at = datetime.now(timezone.utc).isoformat()
    recent = diagnostics.recent(100)
    enriched = []
    for item in rows:
        last = next((r for r in reversed(recent) if r.get('device') == device and
                     r.get('model') == item['key']), None)
        enriched.append(dict(item, availability='online',
                             load_state='loaded' if item['loaded_instances'] else 'installed_unloaded',
                             checked_at=checked_at, last_request_status=last.get('status') if last else 'not_checked',
                             last_request_reason=last.get('reason') if last else None))
    return enriched


def resource_summary(rows: list[dict], cap: int | None) -> dict:
    return {'max_loaded_bytes': cap,
            'loaded_model_weight_bytes': sum(m.get('size_bytes', 0) for m in rows if m['loaded_instances']),
            'note': 'Disk weight sizes are not memory estimates. Cold-load preflight uses the configured engine estimate and fresh RAM/VRAM telemetry when available.'}


def completion(data: dict, requested_model: str, device: str | None, started: float) -> dict:
    choices = data.get('choices')
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ValueError('Model returned no completion choices.')
    choice = choices[0]
    message = choice.get('message') or {}
    content = message.get('content') if isinstance(message, dict) else None
    if isinstance(content, list):
        content = '\n'.join(p['text'] for p in content if isinstance(p, dict) and isinstance(p.get('text'), str))
    if not isinstance(content, str) or not content.strip():
        raise ValueError('Model returned no final text (possibly exhausted its reasoning token budget). No automatic retry was made.')
    return {
        'device': device, 'route': 'direct_engine' if device else 'pair_router',
        'requested_model': requested_model, 'reported_model': data.get('model'),
        'answer': content, 'finish_reason': choice.get('finish_reason'),
        'truncated': choice.get('finish_reason') == 'length',
        'elapsed_seconds': round(time.monotonic() - started, 2), 'usage': data.get('usage'),
    }


def rank_models(inventory: list[dict], model: str | None = None, device: str | None = None,
                context_length: int = 8192, task_hint: str = 'general',
                max_load_bytes: int | None = None) -> list[tuple[str, dict]]:
    """Return installed LLMs in deterministic preference order."""
    candidates = []
    for row in inventory:
        if not row.get('online') or (device is not None and row.get('device') != device):
            continue
        for item in row.get('models', []):
            if item.get('type') != 'llm' or (model is not None and item.get('key') != model):
                continue
            limit = item.get('max_context_length')
            if isinstance(limit, int) and limit < context_length:
                continue
            candidates.append((row['device'], item))
    if not candidates:
        raise ValueError('No suitable installed chat model on an online configured device; no download was made')
    scores = benchmarks.summaries(task_hint, context_length) if model is None else {}
    qualified = any(score['pass_rate'] >= .6 for score in scores.values())
    def rank(row):
        target, item = row
        loaded = bool(item['loaded_instances'])
        size = item.get('size_bytes') if isinstance(item.get('size_bytes'), int) else 1 << 62
        capacity = item.get('max_context_length') if isinstance(item.get('max_context_length'), int) else 0
        code_hint = any(term in item['key'].lower() for term in ('code', 'coder', 'devstral'))
        score = scores.get((target, item['key']))
        if qualified:
            if score and score['pass_rate'] >= .6:
                primary = (0, score['median_latency_ms'], -score['pass_rate']) if task_hint == 'fast' else (0, -score['pass_rate'], score['median_latency_ms'])
            else:
                primary = (1, 0, 0)
        else:
            primary = (0, 0, 0)
        if task_hint == 'code':
            return (*primary, not code_hint, not loaded, -capacity, size, target, item['key'])
        if task_hint == 'fast':
            return (*primary, not loaded, size, target, item['key'])
        if task_hint == 'long_context':
            return (*primary, -capacity, not loaded, size, target, item['key'])
        if task_hint == 'analysis':
            return (*primary, not loaded, -capacity, -size, target, item['key'])
        return (*primary, not loaded, size, target, item['key'])
    return sorted(candidates, key=rank)


def select_model(inventory: list[dict], model: str | None = None, device: str | None = None,
                 context_length: int = 8192, task_hint: str = 'general',
                 max_load_bytes: int | None = None) -> tuple[str, dict]:
    """Keep the legacy one-result selector for callers that do not preflight memory."""
    return rank_models(inventory, model, device, context_length, task_hint, max_load_bytes)[0]


def _validate_smart_candidate(item: dict, context_length: int) -> None:
    if item.get('type') != 'llm':
        raise ValueError('Candidate is no longer a chat LLM')
    limit = item.get('max_context_length')
    if isinstance(limit, int) and limit < context_length:
        raise ValueError('Candidate context is below the requested context')
    instances = item.get('loaded_instances', [])
    if len(instances) > 1:
        raise ValueError('Multiple instances of the candidate are loaded')
    if instances:
        config = instances[0].get('config') or {}
        actual = config.get('context_length')
        if isinstance(actual, int) and actual < context_length:
            raise ValueError('Loaded candidate context is below the requested context')


def select_model_for_memory(inventory: list[dict], context_length: int, task_hint: str,
                            max_load_bytes: int | None = None,
                            device: str | None = None) -> tuple[str, dict, dict, list[dict]]:
    """Choose the highest-ranked model that passes preflight, then try configured evictions."""
    ranked = rank_models(inventory, device=device, context_length=context_length, task_hint=task_hint,
                         max_load_bytes=max_load_bytes)
    errors = []
    configs = management.devices()
    all_unloaded = []
    for target, item in ranked:
        device_row = next(row for row in inventory if row.get('device') == target)
        try:
            _validate_smart_candidate(item, context_length)
            if item.get('loaded_instances'):
                return target, item, {'status': 'already_loaded'}, [], errors
            config = configs[target]
            configured_cap = config.get('max_loaded_bytes')
            cap = min(configured_cap, max_load_bytes) if configured_cap and max_load_bytes else (configured_cap or max_load_bytes)
            memory = management.memory_preflight(target, device_row['models'], item, context_length,
                                                 cap, telemetry.sample(config))
            return target, item, memory, [], errors
        except ValueError as exc:
            errors.append(f"{target}/{item['key']}: {exc}")

    for target, item in ranked:
        config = configs[target]
        if not config.get('auto_unload_models'):
            continue
        try:
            _validate_smart_candidate(item, context_length)
            configured_cap = config.get('max_loaded_bytes')
            cap = min(configured_cap, max_load_bytes) if configured_cap and max_load_bytes else (configured_cap or max_load_bytes)
            with inference_lock(target, wait_seconds=30), management.client(target) as c:
                prepared = management.preflight_for_load(
                    c, target, item['key'], context_length, config, cap,
                    telemetry.sample(config), lambda: telemetry.sample(config))
            all_unloaded.extend(dict(row, device=target) for row in prepared['auto_unloaded_instances'])
            _validate_smart_candidate(prepared['candidate'], context_length)
            return target, prepared['candidate'], prepared['memory'], all_unloaded, errors
        except ValueError as exc:
            all_unloaded.extend(dict(row, device=target) for row in getattr(exc, 'auto_unloaded_instances', []))
            errors.append(f"{target}/{item['key']}: {exc}")

    if not ranked:
        raise ValueError('No suitable installed chat model on an online configured device; no download was made')
    details = '; '.join(errors[:4])
    if all_unloaded:
        details += '; already released configured instances: ' + management.describe_unloaded_instances(all_unloaded)
    error = ValueError('No installed chat model passes current memory preflight' + (f': {details}' if details else ''))
    error.auto_unloaded_instances = all_unloaded
    raise error


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False))
@diagnostics.traced
def pair_benchmark(device: str, model: str,
                   profile: Annotated[str, Field(pattern='^(general|code|fast|long_context|analysis)$')],
                   cases: list[dict[str, str]]) -> dict:
    """Run 3–12 objective substring checks on one already loaded model; save only metrics."""
    if not 3 <= len(cases) <= 12 or any(
        not isinstance(case, dict) or not isinstance(case.get('prompt'), str) or
        not isinstance(case.get('expected_contains'), str) or
        not 1 <= len(case['prompt']) <= 4000 or not 1 <= len(case['expected_contains']) <= 200
        for case in cases
    ):
        raise ValueError('Provide 3–12 cases with prompt and expected_contains within size limits')
    diagnostics.stage('queue', device=device, model=model)
    with inference_lock(device, wait_seconds=30), management.client(device) as c:
        selected = management.find_model(c, model, device)
        if selected.get('type') != 'llm' or len(selected['loaded_instances']) != 1:
            raise ValueError('Benchmark requires one already loaded chat instance; no model was loaded')
        instance = selected['loaded_instances'][0]
        config = instance.get('config') or {}
        context_length = config.get('context_length')
        if not isinstance(context_length, int):
            raise ValueError('Loaded context is unknown; benchmark cannot be attributed')
        outcomes = []
        for index, case in enumerate(cases):
            diagnostics.stage('inference')
            started = time.monotonic()
            try:
                data = management.request(c, 'POST', '/v1/chat/completions',
                                          {'model': management.chat_model_id(device, model, instance['id']),
                                           'messages': [{'role': 'user', 'content': case['prompt']}],
                                           'max_tokens': 2048, 'temperature': 0, 'stream': False})
                answer = completion(data, model, device, started)['answer']
                passed = case['expected_contains'].casefold() in answer.casefold()
                status = 'pass' if passed else 'mismatch'
            except ValueError as exc:
                passed, status = False, diagnostics.reason(exc)
            outcomes.append({'case': index + 1, 'passed': passed, 'status': status,
                             'latency_ms': round((time.monotonic() - started) * 1000)})
        summary = benchmarks.save(device, model, profile, context_length, outcomes)
    return {'device': device, 'model': model, 'profile': profile, 'summary': summary,
            'cases': outcomes, 'notice': 'Exact substring checks measure this suite only; prompts and answers were not saved.'}


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False))
def pair_benchmark_results(profile: Annotated[str, Field(pattern='^(general|code|fast|long_context|analysis)$')],
                           context_length: Annotated[int, Field(ge=512, le=262144)] = 8192) -> dict:
    """Read recent benchmark metrics used by automatic model selection."""
    scores = benchmarks.summaries(profile, context_length)
    return {'profile': profile, 'minimum_context_length': context_length,
            'results': [dict(device=d, model=m, **value) for (d, m), value in sorted(scores.items())]}


@contextlib.contextmanager
def inference_lock(device: str | None = None, wait_seconds: float = 0):
    # Per-device queues allow independent configured devices to run concurrently.
    # Router calls have a separate lock because their final host is unknown.
    if device is not None and not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', device):
        raise ValueError('Invalid device ID for lock')
    folder = user_cache_path('codex-pair-bridge', appauthor=False)
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock = FileLock(str(folder / ('router.lock' if device is None else 'device-' + device + '.lock')),
                    timeout=wait_seconds)
    try:
        lock.acquire()
    except Timeout as exc:
        raise ValueError('Another Codex PAIR request is running on this target. Wait for it to finish before calling again.') from exc
    try:
        yield
    finally:
        lock.release()



@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False))
def pair_list(device: str | None = None) -> dict:
    """List the current model IDs advertised by PAIR across its connected computers.

    With device, return installed models and loaded instances from that device.
    Without device, return the PAIR routing catalog. If that catalog is empty,
    return installed models from configured online devices as an explicit fallback.
    kind_hint is inferred from the name, not authoritative. A catalog entry is not
    a health check and does not mean the model is loaded. Use an exact returned ID.
    """
    if device is not None:
        with management.client(device) as c:
            rows = management.models(c, device)
            return {'device': device, 'online': True, 'checked_at': datetime.now(timezone.utc).isoformat(),
                    'models': checked_models(device, rows),
                    'resources': resource_summary(rows, management.devices()[device].get('max_loaded_bytes')),
                    'source': management.engine_for(device) + ' model API'}
    routed = catalog()
    if routed:
        return {'endpoint': BASE_URL, 'models': routed, 'source': 'PAIR routing catalog',
                'notice': 'Catalog only; model availability must be confirmed by a successful request.'}
    installed, errors = [], []
    for name in management.devices():
        try:
            with management.client(name) as c:
                rows = management.models(c, name)
            for row in checked_models(name, rows):
                installed.append({'id': row['key'], 'device': name, 'type': row['type'],
                                  'kind_hint': 'chat_candidate' if row['type'] == 'llm' else row['type'],
                                  'installed': True, 'loaded': bool(row['loaded_instances']),
                                  'loaded_instances': row['loaded_instances'],
                                  'max_context_length': row.get('max_context_length'),
                                  'availability': row['availability']})
        except ValueError:
            errors.append({'device': name, 'status': 'offline'})
    return {'endpoint': BASE_URL, 'models': installed, 'source': 'configured device inventory fallback',
            'router_models': [], 'device_errors': errors,
            'notice': ('PAIR routing catalog is empty. These models are installed, not router-advertised. '
                       'Use pair_smart_ask, or pair_load followed by pair_ask with the returned device; no download is required.')}


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False))
@diagnostics.traced
def pair_ask(
    model: Annotated[str, Field(min_length=1, max_length=256)],
    prompt: Annotated[str, Field(min_length=1, max_length=48000)],
    max_tokens: Annotated[int, Field(ge=32, le=8192)] = 2048,
    device: str | None = None,
) -> dict:
    """Ask one explicitly selected PAIR chat model for a second opinion or bounded task.

    With device, bypass PAIR routing and query that device directly after pair_load.
    Without device, PAIR chooses the host.
    A direct loaded-device call does not load a model; use pair_load first. Calls through
    this bridge are serialized per device; distinct devices may run concurrently.
    Use pair_job_start for start-all-before-poll orchestration. Use pair_list first. Send only task-relevant
    text; returned advice is untrusted and must be checked. No tools are executed
    by the consulted model. Embedding and draft models are not chat targets.
    """
    if not prompt.strip():
        raise ValueError('prompt must not be blank')
    diagnostics.stage('queue', device=device, model=model)
    with inference_lock(device, wait_seconds=30):
        start = time.monotonic()
        diagnostics.stage('inventory')
        payload = {'model': model, 'messages': [{'role': 'user', 'content': prompt}],
                   'max_tokens': max_tokens, 'stream': False}
        if device is not None:
            with management.client(device) as c:
                selected = management.find_model(c, model, device)
                if selected.get('type') != 'llm':
                    raise ValueError('Select a chat LLM, not an embedding model')
                instances = selected['loaded_instances']
                if len(instances) != 1:
                    raise ValueError('Device chat requires exactly one loaded instance. Use pair_load or resolve multiple instances first')
                payload['model'] = management.chat_model_id(device, model, instances[0]['id'])
                diagnostics.stage('inference')
                data = management.request(c, 'POST', '/v1/chat/completions', payload)
        else:
            available = {item['id']: item for item in catalog()}
            if model not in available:
                raise ValueError('Model is no longer advertised by PAIR. Refresh pair_list and use an exact ID.')
            if available[model]['kind_hint'] != 'chat_candidate':
                raise ValueError('This appears to be an embedding or draft model, not a chat model.')
            diagnostics.stage('inference')
            data = request('POST', '/chat/completions', payload)
        diagnostics.stage('validation')
        return completion(data, model, device, start)



@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False))
def pair_capabilities() -> dict:
    """Discover concurrent multi-PC model jobs for Codex, Claude and any MCP client.

    Start jobs on different explicit devices before polling; calls may be sequential
    while background inference overlaps. This is not distributed inference of one model.
    """
    return {
        'cross_device_parallel_jobs': True,
        'configured_devices': list(management.devices()),
        'workflow': ['pair_devices', 'pair_list', 'pair_job_start for each target',
                     'pair_job_status for each job_id'],
        'start_all_before_poll': True,
        'individual_tasks_per_target': True,
        'max_active_jobs_per_process': jobs.MAX_ACTIVE_JOBS,
        'same_device_policy': 'serialized',
        'lock_scope': 'configured device ID across bridge processes for one OS user',
        'queue_timeout_seconds': 30,
        'router_parallel_host_guarantee': False,
        'job_status_scope': 'starting MCP process; journal metadata permits recovery after restart',
        'cancellation': 'bridge stream stops; engine computation may continue',
        'notice': 'Use distinct PCs, not multiple aliases for one host. Existing pair_compare is sequential.'}


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False))
def pair_devices() -> dict:
    """Inspect every configured device, including installed and loaded local models.

    Reports each unreachable device separately. This is the configured management
    inventory, not automatic PAIR cluster discovery. No model is loaded by this call.
    """
    result = []
    for name, config in management.devices().items():
        try:
            with management.client(name) as c:
                models = management.models(c, name)
                result.append({'device': name, 'online': True,
                               'checked_at': datetime.now(timezone.utc).isoformat(),
                               'models': checked_models(name, models),
                               'resources': resource_summary(models, config.get('max_loaded_bytes')),
                               'capacity': telemetry.sample(config)})
        except ValueError as exc:
            result.append({'device': name, 'online': False,
                           'checked_at': datetime.now(timezone.utc).isoformat(),
                           'check_status': 'unreachable', 'error': str(exc)})
    return {'devices': result, 'capabilities': pair_capabilities(),
            'notice': 'Management covers explicitly configured LM Studio and Unsloth devices only. PAIR routing catalog remains pair_list().'}


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False))
@diagnostics.traced
def pair_load(device: str, model: str,
              context_length: Annotated[int, Field(ge=512, le=262144)] = 8192) -> dict:
    """Load an installed model on one device into RAM/VRAM; never download weights.

    Reuses existing loaded instances without changing their configuration. May
    consume substantial memory or trigger engine auto-eviction. Verify the result.
    """
    diagnostics.stage('queue', device=device, model=model)
    with inference_lock(device, wait_seconds=30), management.client(device) as c:
        auto_unloaded = []
        with management.report_releases_on_error(lambda: auto_unloaded):
            diagnostics.stage('inventory')
            before_rows = management.models(c, device)
            matches = [m for m in before_rows if m['key'] == model]
            if len(matches) != 1:
                raise ValueError('Model is not installed on this device. Refresh pair_list(device=...)')
            selected = matches[0]
            if selected['loaded_instances']:
                return {'device': device, 'status': 'already_loaded', 'model': selected}
            maximum = selected.get('max_context_length')
            if selected.get('type') == 'llm' and isinstance(maximum, int) and context_length > maximum:
                raise ValueError('Requested context exceeds this model maximum')
            diagnostics.stage('preflight')
            device_config = management.devices()[device]
            prepared = management.preflight_for_load(
                c, device, model, context_length, device_config,
                device_config.get('max_loaded_bytes'), telemetry.sample(device_config),
                lambda: telemetry.sample(device_config))
            memory = prepared['memory']
            selected = prepared['candidate']
            if selected['loaded_instances']:
                return {'device': device, 'status': 'already_loaded', 'model': selected,
                        'auto_unloaded_instances': prepared['auto_unloaded_instances']}
            diagnostics.stage('load')
            try:
                load_attempt = management.load_with_auto_unload(
                    c, device, model, context_length, selected.get('type'), device_config)
                load_result = load_attempt['result']
            except ValueError as exc:
                released = [*prepared['auto_unloaded_instances'], *getattr(exc, 'auto_unloaded_instances', [])]
                exc.auto_unloaded_instances = released
                details = management.describe_unloaded_instances(released)
                if details:
                    error = ValueError(f'{exc}; configured auto_unload_models released: {details}')
                    error.auto_unloaded_instances = released
                    raise error from exc
                raise
            auto_unloaded = [*prepared['auto_unloaded_instances'], *load_attempt['auto_unloaded_instances']]
            newly_loaded = management.load_creates_owned_instance(device, load_result)
            diagnostics.stage('verification')
            after = management.find_model(c, model, device)
            after_rows = management.models(c, device)
            before_ids = {i['id'] for m in before_rows for i in m['loaded_instances']}
            after_ids = {i['id'] for m in after_rows for i in m['loaded_instances']}
            return {'device': device, 'status': ('loaded' if newly_loaded else 'already_loaded') if after['loaded_instances'] else 'not_confirmed',
                    'model': after, 'load_time_seconds': load_result.get('load_time_seconds'),
                    'engine_evicted_instances': sorted(before_ids - after_ids -
                                                       {row['instance_id'] for row in auto_unloaded}),
                    'auto_unloaded_instances': auto_unloaded, 'memory_preflight': memory}


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False))
def pair_memory_plan(device: str, model: str,
                     context_length: Annotated[int, Field(ge=512, le=262144)] = 8192) -> dict:
    """Estimate memory on the configured engine without loading the installed model."""
    with inference_lock(device, wait_seconds=30), management.client(device) as c:
        rows = management.models(c, device)
        selected = next((m for m in rows if m['key'] == model), None)
        if selected is None:
            raise ValueError('Model is not installed on this device; no download was made')
        maximum = selected.get('max_context_length')
        if isinstance(maximum, int) and context_length > maximum:
            raise ValueError('Requested context exceeds this model maximum')
        result = management.memory_preflight(device, rows, selected, context_length,
                                             management.devices()[device].get('max_loaded_bytes'),
                                             telemetry.sample(management.devices()[device]))
    return {'device': device, 'model': model, 'planned_context_length': context_length,
            'current_instances': selected['loaded_instances'], 'preflight': result}


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=False, openWorldHint=False))
@diagnostics.traced
def pair_unload(device: str, instance_id: str) -> dict:
    """Unload one exact loaded instance from a device; model files stay installed.

    This may disrupt users outside this bridge. Do not unload unrelated models
    merely because they are loaded; use the user's requested scope. No unload-all.
    """
    diagnostics.stage('queue', device=device)
    with inference_lock(device, wait_seconds=30), management.client(device) as c:
        diagnostics.stage('inventory')
        before = management.models(c, device)
        owner = next((m['key'] for m in before for i in m['loaded_instances']
                      if i['id'] == instance_id), None)
        if owner is None:
            raise ValueError('Instance is not loaded. Refresh pair_list(device=...)')
        diagnostics.stage('unload')
        management.unload_model(c, device, owner, instance_id)
        diagnostics.stage('verification')
        remaining = management.models(c, device)
        still_loaded = any(i['id'] == instance_id for m in remaining for i in m['loaded_instances'])
        return {'device': device, 'instance_id': instance_id, 'status': 'not_confirmed' if still_loaded else 'unloaded'}


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False))
@diagnostics.traced
def pair_smart_ask(
    prompt: Annotated[str, Field(min_length=1, max_length=48000)],
    model: str | None = None,
    device: str | None = None,
    context_length: Annotated[int, Field(ge=512, le=262144)] = 8192,
    max_tokens: Annotated[int, Field(ge=32, le=8192)] = 2048,
    unload_after: bool = True,
    task_hint: Annotated[str, Field(pattern='^(general|code|fast|long_context|analysis)$')] = 'general',
    max_load_bytes: Annotated[int | None, Field(ge=1)] = None,
) -> dict:
    """Select a memory-fit installed LLM unless one is explicit, load if needed, ask once, and clean up only a new instance.

    No model downloads. Automatic selection checks memory before choosing; an explicit model is never substituted.
    Existing loaded instances are preserved. An ambiguous multi-instance model is not selected automatically.
    """
    if not prompt.strip():
        raise ValueError('prompt must not be blank')
    diagnostics.stage('inventory', device=device, model=model)
    try:
        router_models = {row['id'] for row in catalog()}
        router_status = 'online'
    except ValueError:
        router_models = set()
        router_status = 'unavailable'
    snapshot = pair_devices()
    preselected_unloaded = []
    memory_rejections = []
    if model is None:
        selected_device, selected, _selection_memory, preselected_unloaded, memory_rejections = select_model_for_memory(
            snapshot['devices'], context_length, task_hint, max_load_bytes, device=device)
    else:
        selected_device, selected = select_model(snapshot['devices'], model, device, context_length,
                                                 task_hint, max_load_bytes)
    measured = benchmarks.summaries(task_hint, context_length).get((selected_device, selected['key'])) if model is None else None
    diagnostics.stage('queue', device=selected_device, model=selected['key'])
    with inference_lock(selected_device, wait_seconds=30):
        with management.report_releases_on_error(lambda: preselected_unloaded):
            key = selected['key']
            started = time.monotonic()
            owned_id = None
            cleanup = 'not_needed'
            load_time_seconds = None
            engine_evicted_instances = []
            with management.client(selected_device) as c:
                # Recheck after selection: another application may have changed the load state.
                live = management.find_model(c, key, selected_device)
                if live.get('type') != 'llm':
                    raise ValueError('Selected model is no longer a chat LLM; refresh inventory')
                limit = live.get('max_context_length')
                if isinstance(limit, int) and limit < context_length:
                    raise ValueError('Selected model context is now smaller than requested; refresh inventory')
                instances = live['loaded_instances']
                if len(instances) > 1:
                    raise ValueError('Multiple instances of the selected model are loaded; choose and manage one explicitly')
                if instances:
                    load_config = instances[0].get('config')
                    actual_context = load_config.get('context_length') if isinstance(load_config, dict) else None
                    if isinstance(actual_context, int) and actual_context < context_length:
                        raise ValueError('Loaded instance context is smaller than requested; choose a smaller context or another model')
                if not instances:
                    diagnostics.stage('preflight')
                    before_rows = management.models(c, selected_device)
                    candidate = next((m for m in before_rows if m['key'] == key), None)
                    if candidate is None:
                        raise ValueError('Selected model disappeared before load; refresh inventory')
                    device_config = management.devices()[selected_device]
                    configured_cap = device_config.get('max_loaded_bytes')
                    cap = min(configured_cap, max_load_bytes) if configured_cap and max_load_bytes else (configured_cap or max_load_bytes)
                    try:
                        prepared = management.preflight_for_load(
                            c, selected_device, key, context_length, device_config, cap,
                            telemetry.sample(device_config), lambda: telemetry.sample(device_config))
                    except ValueError as exc:
                        exc.auto_unloaded_instances = [*preselected_unloaded, *management.device_releases(getattr(exc, 'auto_unloaded_instances', []), selected_device)]
                        if exc.auto_unloaded_instances:
                            error = ValueError(f'{exc}; confirmed configured releases: '
                                               f'{management.describe_unloaded_instances(exc.auto_unloaded_instances)}')
                            error.auto_unloaded_instances = exc.auto_unloaded_instances
                            raise error from exc
                        raise
                    memory = prepared['memory']
                    candidate = prepared['candidate']
                    preselected_unloaded = [*preselected_unloaded, *management.device_releases(prepared['auto_unloaded_instances'], selected_device)]
                    diagnostics.stage('load')
                    before_ids = {i['id'] for m in before_rows for i in m['loaded_instances']}
                    if candidate['loaded_instances']:
                        if len(candidate['loaded_instances']) != 1:
                            raise ValueError('Multiple instances of the selected model became loaded; inspect state')
                        instances = candidate['loaded_instances']
                    else:
                        try:
                            load_attempt = management.load_with_auto_unload(
                                c, selected_device, key, context_length, candidate.get('type'), device_config)
                            load_result = load_attempt['result']
                        except ValueError as exc:
                            preselected_unloaded.extend(management.device_releases(getattr(exc, 'auto_unloaded_instances', []), selected_device))
                            exc.auto_unloaded_instances = preselected_unloaded
                            details = management.describe_unloaded_instances(preselected_unloaded)
                            if details:
                                error = ValueError(f'{exc}; configured auto_unload_models released: {details}')
                                error.auto_unloaded_instances = preselected_unloaded
                                raise error from exc
                            raise
                        preselected_unloaded = [*preselected_unloaded, *management.device_releases(load_attempt['auto_unloaded_instances'], selected_device)]
                        load_time_seconds = load_result.get('load_time_seconds')
                        after = management.find_model(c, key, selected_device)
                        if len(after['loaded_instances']) != 1:
                            raise ValueError('Load state is not confirmed; inspect pair_list before retrying')
                        loaded_id = after['loaded_instances'][0]['id']
                        if management.engine_for(selected_device) != 'unsloth':
                            response_id = load_result.get('instance_id')
                            if not isinstance(response_id, str) or response_id != loaded_id:
                                raise ValueError('Load state is not confirmed; inspect pair_list before retrying')
                        if management.load_creates_owned_instance(selected_device, load_result):
                            owned_id = loaded_id
                        instances = after['loaded_instances']
                        after_ids = {i['id'] for m in management.models(c, selected_device) for i in m['loaded_instances']}
                        intentional = {row['instance_id'] for row in preselected_unloaded}
                        engine_evicted_instances = sorted(before_ids - after_ids - intentional)
                payload = {'model': management.chat_model_id(selected_device, key, instances[0]['id']),
                           'messages': [{'role': 'user', 'content': prompt}],
                           'max_tokens': max_tokens, 'stream': False}
                try:
                    diagnostics.stage('inference')
                    data = management.request(c, 'POST', '/v1/chat/completions', payload)
                    diagnostics.stage('validation')
                    result = completion(data, key, selected_device, started)
                except Exception as exc:
                    # A timeout may leave inference running. Retain the instance for inspection.
                    details = management.describe_unloaded_instances(preselected_unloaded)
                    if details:
                        error = ValueError(f'{exc}; configured auto_unload_models released: {details}')
                        error.auto_unloaded_instances = preselected_unloaded
                        raise error from exc
                    raise
                else:
                    if owned_id and unload_after:
                        diagnostics.stage('cleanup')
                        # This lock excludes other bridge calls, but cannot observe external clients.
                        current = management.find_model(c, key, selected_device)['loaded_instances']
                        if len(current) == 1 and current[0]['id'] == owned_id:
                            try:
                                management.unload_model(c, selected_device, key, owned_id)
                                confirmed = management.find_model(c, key, selected_device)['loaded_instances']
                                cleanup = 'unloaded' if not confirmed else 'not_confirmed'
                            except ValueError:
                                cleanup = 'failed_inspect_instance'
                        else:
                            cleanup = 'state_changed_preserved'
                    elif owned_id:
                        cleanup = 'new_instance_retained'
                    else:
                        cleanup = 'existing_instance_preserved'
                    if model or device:
                        selection_reason = 'explicit model/device'
                    elif measured and measured['pass_rate'] >= .6:
                        selection_reason = (f"benchmark: {measured['passed']}/{measured['cases']} cases, "
                                            f"median {measured['median_latency_ms']} ms among memory-fit candidates")
                    else:
                        selection_reason = 'highest-ranked installed chat model that passes memory preflight'
                    return dict(result, selected_model=key, instance_id=instances[0]['id'],
                                loaded_for_request=bool(owned_id), cleanup=cleanup,
                                selection_profile=task_hint,
                                selection_reason=selection_reason,
                                load_time_seconds=load_time_seconds,
                                engine_evicted_instances=engine_evicted_instances,
                                auto_unloaded_instances=preselected_unloaded,
                                memory_preflight_rejections=memory_rejections,
                                memory_preflight=memory if owned_id else {'status': 'already_loaded'},
                                router_status=router_status, router_advertises_model=key in router_models)


async def _job_stream(job: jobs.Job, c: httpx.AsyncClient, instance_id: str, prompt: str, max_tokens: int) -> str:
    """Consume engine SSE; retain message text only, never reasoning/tool content."""
    if management.engine_for(job.device) == 'unsloth':
        payload = {'model': management.chat_model_id(job.device, job.model, instance_id),
                   'messages': [{'role': 'user', 'content': prompt}],
                   'max_tokens': max_tokens, 'stream': True}
        ended = False
        async with c.stream('POST', '/v1/chat/completions', json=payload) as response:
            if not response.is_success:
                raise ValueError(f'Device returned HTTP {response.status_code}; no retry was made')
            async for line in response.aiter_lines():
                if job.cancel_event.is_set():
                    raise asyncio.CancelledError()
                if not line.startswith('data:'):
                    continue
                event = line[5:].strip()
                if not event:
                    continue
                if event == '[DONE]':
                    ended = True
                    break
                if len(event) > 1_000_000:
                    raise ValueError('Device SSE event exceeded size limit')
                try:
                    data = json.loads(event)
                except ValueError as exc:
                    raise ValueError('Device returned invalid SSE JSON') from exc
                if not isinstance(data, dict):
                    raise ValueError('Device returned invalid SSE event')
                if 'error' in data:
                    raise ValueError('Device reported a streaming error')
                choices = data.get('choices')
                if not isinstance(choices, list) or not choices:
                    continue
                first = choices[0]
                delta = first.get('delta') if isinstance(first, dict) else None
                content = delta.get('content') if isinstance(delta, dict) else None
                if isinstance(content, str):
                    job.add_text(content)
                    job.update(stage='generating', progress=None)
                elif isinstance(content, list):
                    text = ''.join(part.get('text', '') for part in content
                                   if isinstance(part, dict) and part.get('type') == 'text'
                                   and isinstance(part.get('text'), str))
                    if text:
                        job.add_text(text)
                        job.update(stage='generating', progress=None)
        if job.cancel_event.is_set():
            raise asyncio.CancelledError()
        if not ended:
            raise ValueError('Device stream ended before [DONE]')
        if not job.answer.strip():
            raise ValueError('Model returned no final text')
        return job.answer

    payload = {'model': instance_id, 'input': prompt, 'max_output_tokens': max_tokens,
               'stream': True, 'store': False, 'integrations': []}
    ended = False
    async with c.stream('POST', '/api/v1/chat', json=payload) as response:
        if not response.is_success:
            raise ValueError(f'Device returned HTTP {response.status_code}; no retry was made')
        if job.cancel_event.is_set():
            raise asyncio.CancelledError()
        event_type, data_text = None, ''
        async for line in response.aiter_lines():
            if job.cancel_event.is_set():
                raise asyncio.CancelledError()
            if line.startswith('event:'):
                event_type = line[6:].strip()
            elif line.startswith('data:'):
                data_text += line[5:].strip()
                if len(data_text) > 1_000_000:
                    raise ValueError('Device SSE event exceeded size limit')
            elif not line and data_text:
                try:
                    data = json.loads(data_text)
                except ValueError as exc:
                    raise ValueError('Device returned invalid SSE JSON') from exc
                if not isinstance(data, dict):
                    raise ValueError('Device returned invalid SSE event')
                kind = data.get('type') or event_type
                if kind in ('model_load.progress', 'prompt_processing.progress'):
                    progress = data.get('progress')
                    if isinstance(progress, (int, float)) and 0 <= progress <= 1:
                        job.update(stage=kind, progress=round(progress, 3))
                elif kind in ('model_load.start', 'prompt_processing.start', 'reasoning.start', 'message.start'):
                    job.update(stage=kind, progress=None)
                elif kind == 'message.delta' and isinstance(data.get('content'), str):
                    job.add_text(data['content'])
                    job.update(stage='generating', progress=None)
                elif kind == 'error':
                    raise ValueError('Device reported a streaming error')
                elif kind == 'chat.end':
                    result = data.get('result') or {}
                    output = result.get('output') if isinstance(result, dict) else None
                    if isinstance(output, list):
                        final = ''.join(x.get('content', '') for x in output
                                        if isinstance(x, dict) and x.get('type') == 'message' and isinstance(x.get('content'), str))
                        if final:
                            with job.lock:
                                if not job.cancel_event.is_set():
                                    job.answer = final[:48000]
                    ended = True
                event_type, data_text = None, ''
    if job.cancel_event.is_set():
        raise asyncio.CancelledError()
    if not ended:
        raise ValueError('Device stream ended before chat.end')
    if not job.answer.strip():
        raise ValueError('Model returned no final text')
    return job.answer


def _run_job(job: jobs.Job, prompt: str, context_length: int, max_tokens: int, unload_after: bool) -> None:
    owned_id = None
    try:
        with inference_lock(job.device, wait_seconds=30), management.client(job.device) as c:
            if job.cancel_event.is_set():
                job.update(status='cancelled', stage='cancelled')
                return
            job.update(status='running', stage='inventory')
            rows = management.models(c, job.device)
            selected = next((m for m in rows if m['key'] == job.model), None)
            if selected is None or selected.get('type') != 'llm':
                raise ValueError('Model is not an installed chat LLM on this device')
            maximum = selected.get('max_context_length')
            if isinstance(maximum, int) and context_length > maximum:
                raise ValueError('Requested context exceeds model maximum')
            instances = selected['loaded_instances']
            if len(instances) > 1:
                raise ValueError('Multiple model instances are loaded; choose one explicitly')
            if instances:
                actual = (instances[0].get('config') or {}).get('context_length')
                if isinstance(actual, int) and actual < context_length:
                    raise ValueError('Loaded instance context is smaller than requested')
                instance_id = instances[0]['id']
            else:
                job.update(stage='preflight')
                config = management.devices()[job.device]
                prepared = management.preflight_for_load(
                    c, job.device, job.model, context_length, config,
                    config.get('max_loaded_bytes'), telemetry.sample(config),
                    lambda: telemetry.sample(config))
                selected = prepared['candidate']
                job.update(auto_unloaded_instances=prepared['auto_unloaded_instances'])
                instances = selected['loaded_instances']
                if instances:
                    if len(instances) != 1:
                        raise ValueError('Multiple model instances became loaded; choose one explicitly')
                    actual = (instances[0].get('config') or {}).get('context_length')
                    if isinstance(actual, int) and actual < context_length:
                        raise ValueError('Loaded instance context is smaller than requested')
                    instance_id = instances[0]['id']
                else:
                    if job.cancel_event.is_set():
                        job.update(status='cancelled', stage='cancelled')
                        return
                    job.update(stage='loading')
                    load_attempt = management.load_with_auto_unload(
                        c, job.device, job.model, context_length, selected.get('type'), config)
                    job.update(auto_unloaded_instances=[*prepared['auto_unloaded_instances'],
                                                       *load_attempt['auto_unloaded_instances']])
                    confirmed = management.find_model(c, job.model, job.device)['loaded_instances']
                    if len(confirmed) != 1:
                        raise ValueError('Loaded instance could not be confirmed')
                    instance_id = confirmed[0]['id']
                    if management.engine_for(job.device) != 'unsloth':
                        response_id = load_attempt['result'].get('instance_id')
                        if not isinstance(response_id, str) or not response_id or response_id != instance_id:
                            raise ValueError('Load state is not confirmed; inspect pair_list before retrying')
                    if management.load_creates_owned_instance(job.device, load_attempt['result']):
                        owned_id = instance_id
                        job.update(instance_id=instance_id, owned=True)
            job.update(instance_id=instance_id)
            if job.cancel_event.is_set():
                if owned_id:
                    management.unload_model(c, job.device, job.model, owned_id)
                    job.update(cleanup='unloaded_before_inference')
                job.update(status='cancelled', stage='cancelled')
                return
            job.update(stage='inference')
            async def stream_request():
                async with httpx.AsyncClient(base_url=str(c.base_url), headers=c.headers,
                                             timeout=180, trust_env=False, follow_redirects=False) as stream_client:
                    return await _job_stream(job, stream_client, instance_id, prompt, max_tokens)
            loop = asyncio.new_event_loop()
            try:
                task = loop.create_task(stream_request())
                with job.lock:
                    job.loop, job.task = loop, task
                    if job.cancel_event.is_set():
                        loop.call_soon(task.cancel)
                loop.run_until_complete(task)
            finally:
                with job.lock:
                    job.loop, job.task = None, None
                loop.close()
            if job.cancel_event.is_set():
                job.update(status='cancelled', stage='cancelled')
                return
            cleanup = 'existing_instance_preserved'
            if owned_id:
                cleanup = 'new_instance_retained'
                if unload_after:
                    current = management.find_model(c, job.model, job.device)['loaded_instances']
                    if len(current) == 1 and current[0]['id'] == owned_id:
                        management.unload_model(c, job.device, job.model, owned_id)
                        cleanup = 'unloaded' if not management.find_model(c, job.model, job.device)['loaded_instances'] else 'not_confirmed'
                    else:
                        cleanup = 'state_changed_preserved'
            job.update(status='completed', stage='completed', progress=1, cleanup=cleanup)
    except asyncio.CancelledError:
        job.update(status='cancelled', stage='cancelled',
                   cleanup='new_instance_preserved_for_inspection' if owned_id else job.cleanup)
    except Exception as exc:
        released = list(job.auto_unloaded_instances)
        for row in getattr(exc, 'auto_unloaded_instances', []):
            if row not in released:
                released.append(row)
        job.update(auto_unloaded_instances=released)
        if job.cancel_event.is_set():
            job.update(status='cancelled', stage='cancelled',
                       cleanup='new_instance_preserved_for_inspection' if owned_id and job.cleanup != 'unloaded_before_inference' else job.cleanup)
        else:
            job.update(status='failed', stage='failed', error_code=diagnostics.reason(exc),
                       cleanup='new_instance_preserved_for_inspection' if owned_id else job.cleanup)
    finally:
        with job.lock:
            job.response = None


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False))
def pair_job_start(device: str, model: str, prompt: Annotated[str, Field(min_length=1, max_length=48000)],
                   context_length: Annotated[int, Field(ge=512, le=262144)] = 8192,
                   max_tokens: Annotated[int, Field(ge=32, le=8192)] = 2048,
                   unload_after: bool = True) -> dict:
    """Start a background model job; different explicit PCs can run concurrently.

    Start every target before polling pair_job_status. Same-device jobs serialize
    with a 30s queue timeout; at most eight active jobs per bridge process.
    Each call accepts its own prompt/model for an individual task on that PC.
    No download, automatic retry or fallback. Job IDs belong to this MCP process.
    """
    if device not in management.devices() or not prompt.strip():
        raise ValueError('Use an exact configured device and nonblank prompt')
    return jobs.create(device, model, _run_job, prompt, context_length, max_tokens, unload_after)


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False))
def pair_job_status(job_id: str) -> dict:
    """Read current progress and final answer; interrupted jobs retain prompt-free recovery metadata."""
    return jobs.get(job_id)


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=True, openWorldHint=False))
def pair_job_cancel(job_id: str) -> dict:
    """Request cancellation. Loading may finish first; an inference instance is preserved for inspection."""
    return jobs.cancel(job_id)


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False))
def pair_job_recover(job_id: str) -> dict:
    """Inspect the exact instance after interruption; never unload or retry automatically."""
    state = jobs.get(job_id)
    instance_id = state.get('instance_id')
    if not instance_id:
        return dict(state, recovery='no_owned_instance_recorded')
    try:
        with management.client(state['device']) as c:
            rows = management.models(c, state['device'])
        present = any(i['id'] == instance_id for row in rows for i in row['loaded_instances'])
        return dict(state, recovery='instance_present' if present else 'instance_absent')
    except ValueError:
        return dict(state, recovery='device_unreachable')


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False))
def pair_compare(prompt: Annotated[str, Field(min_length=1, max_length=48000)],
                 first_model: str, first_device: str, second_model: str, second_device: str,
                 max_tokens: Annotated[int, Field(ge=32, le=8192)] = 2048) -> dict:
    """Ask two explicitly named installed models sequentially; preserve both answers for Codex to assess."""
    if first_model == second_model and first_device == second_device:
        raise ValueError('Comparison requires two distinct model/device targets')
    answers = []
    for model, device in ((first_model, first_device), (second_model, second_device)):
        try:
            answers.append(pair_smart_ask(prompt, model=model, device=device, max_tokens=max_tokens))
        except ValueError as exc:
            answers.append({'device': device, 'model': model, 'error': str(exc)})
    observations = []
    for result in answers:
        if 'answer' not in result:
            observations.append([])
            continue
        observations.append([line.strip().lstrip('-*0123456789. ') for line in result['answer'].splitlines()
                             if len(line.strip()) >= 12][:20])
    shared, disputed = [], []
    if len(observations) == 2:
        used = set()
        for line in observations[0]:
            match = next((i for i, other in enumerate(observations[1]) if i not in used and
                          difflib.SequenceMatcher(None, line.casefold(), other.casefold()).ratio() >= .82), None)
            if match is None:
                disputed.append({'source': 'first', 'observation': line})
            else:
                used.add(match)
                shared.append({'first': line, 'second': observations[1][match]})
        disputed.extend({'source': 'second', 'observation': line} for i, line in enumerate(observations[1]) if i not in used)
    return {'results': answers, 'shared_observations': shared, 'disputed_observations': disputed,
            'verification_status': 'requires_codex_source_review',
            'notice': 'Similarity is textual only. Codex must inspect source files and validate disputed claims before reporting them as findings.'}


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False))
def pair_diagnose() -> dict:
    """Read-only device and router health snapshot without prompts, tokens or private URLs."""
    rows = []
    for name in management.devices():
        try:
            with management.client(name) as c:
                models = management.models(c, name)
            rows.append({'device': name, 'online': True, 'installed': len(models),
                         'chat_models': sum(m.get('type') == 'llm' for m in models),
                         'loaded_instances': sum(len(m['loaded_instances']) for m in models)})
        except ValueError as exc:
            rows.append({'device': name, 'online': False, 'error': str(exc)})
    try:
        advertised = len(catalog())
        router = {'online': True, 'advertised_models': advertised}
    except ValueError:
        router = {'online': False, 'error': 'PAIR router unavailable or catalog invalid'}
    recent, journal_status = diagnostics.recent_with_status()
    explanations = {'timeout': 'The device did not finish before the request deadline; inspect its load state before retrying.',
                    'device_unreachable': 'The device engine could not be reached; check its server and SSH/Tailscale path.',
                    'model_not_installed': 'The requested model is not installed on an online configured device.',
                    'empty_answer': 'The model returned no final text; a larger output budget may be needed.',
                    'load_failed': 'Loading did not complete or could not be confirmed; inspect memory and engine state.',
                    'request_failed': 'The request failed; check local engine logs without sharing prompts or tokens.'}
    return {'devices': rows, 'router': router, 'request_journal_status': journal_status,
            'recent_requests': [dict(row, explanation=explanations.get(row.get('reason')))
                                for row in recent]}


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False))
def pair_download_plan(device: str, model: str, estimated_size_bytes: Annotated[int | None, Field(ge=1)] = None,
                       destination: str = '', quantization: str | None = None) -> dict:
    """Prepare a one-use download review with independent metadata and disk checks.

    LM Studio's download API reveals total size only after starting. Inspect the model
    source, expected size and configured storage location independently before planning.
    """
    if not model or len(model) > 512 or not destination.strip() or len(destination) > 1024:
        raise ValueError('Provide an exact model ID and reviewed destination')
    if quantization is not None and not re.fullmatch(r'[A-Za-z0-9_.-]{1,32}', quantization):
        raise ValueError('Invalid quantization')
    if model.startswith('https://'):
        parts = urlsplit(model)
        if parts.hostname != 'huggingface.co' or parts.username or parts.password or parts.query or parts.fragment:
            raise ValueError('Only exact huggingface.co HTTPS links are accepted as model URLs')
        source = model
    elif re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', model):
        if quantization:
            raise ValueError('Quantization is supported only for Hugging Face repository links')
        source = 'LM Studio catalog: ' + model
    else:
        raise ValueError('Use an exact LM Studio catalog ID or huggingface.co model URL')
    configured = management.devices()
    if device not in configured:
        raise ValueError('Unknown device. Use pair_devices and an exact configured ID')
    if configured[device].get('engine', 'lmstudio') != 'lmstudio':
        raise ValueError('Model downloads are supported only for LM Studio devices')
    files = download_review.model_files(model, quantization)
    verified_bytes = files.get('size_bytes') if files['status'] == 'verified_file' else None
    planning_bytes = verified_bytes or estimated_size_bytes
    if planning_bytes is None:
        raise ValueError('File size is unknown; provide a reviewed estimated_size_bytes before planning')
    space = download_review.destination_space(configured[device], destination, round(planning_bytes * 1.1))
    if space['status'] == 'insufficient':
        raise ValueError('Destination filesystem has less than the planned size plus 10% headroom')
    plan_id = uuid.uuid4().hex
    _DOWNLOAD_PLANS[plan_id] = {'device': device, 'model': model, 'quantization': quantization,
                                'estimated_size_bytes': planning_bytes, 'verified_file': files if verified_bytes else None,
                                'destination': destination, 'created': time.monotonic()}
    return {'plan_id': plan_id, 'device': device, 'model': model, 'source': source,
            'estimated_disk_and_network_bytes': planning_bytes,
            'size_source': 'huggingface_file_metadata' if verified_bytes else 'caller_estimate',
            'model_file': files, 'destination': destination, 'destination_space': space,
            'quantization': quantization,
            'notice': 'Verify LM Studio storage settings and exact model variant. A repository file size may differ from the complete job. Plan expires in 10 minutes; ask never downloads.'}


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=True))
def pair_download(plan_id: str, confirm_model: str) -> dict:
    """Start a download from one reviewed plan, with the exact model ID repeated explicitly."""
    plan = _DOWNLOAD_PLANS.get(plan_id)
    if not plan or time.monotonic() - plan['created'] > 600:
        raise ValueError('Download plan is missing or expired; prepare a new plan')
    if plan['model'] != confirm_model:
        raise ValueError('Repeat the exact model ID in confirm_model before starting a download')
    del _DOWNLOAD_PLANS[plan_id]
    device, model, quantization = plan['device'], plan['model'], plan['quantization']
    if plan['verified_file']:
        current = download_review.model_files(model, quantization)
        if current.get('status') != 'verified_file' or any(
            current.get(key) != plan['verified_file'].get(key) for key in ('revision', 'file', 'size_bytes')
        ):
            raise ValueError('Hugging Face file metadata changed or is unavailable; prepare a new download plan')
    if management.engine_for(device) != 'lmstudio':
        raise ValueError('Model downloads are supported only for LM Studio devices')
    space = download_review.destination_space(management.devices()[device], plan['destination'],
                                              round(plan['estimated_size_bytes'] * 1.1))
    if space['status'] == 'insufficient':
        raise ValueError('Destination free space changed; prepare a new download plan')
    with management.client(device) as c:
        body = {'model': model}
        if quantization:
            body['quantization'] = quantization
        result = management.request(c, 'POST', '/api/v1/models/download', body)
    return {'device': device, 'model': model, 'job_id': result.get('job_id'),
            'status': result.get('status'), 'total_size_bytes': result.get('total_size_bytes')}


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False))
def pair_download_status(device: str, job_id: str) -> dict:
    """Read the progress of an explicitly started LM Studio download job."""
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', job_id):
        raise ValueError('Invalid download job ID')
    if management.engine_for(device) != 'lmstudio':
        raise ValueError('Model downloads are supported only for LM Studio devices')
    with management.client(device) as c:
        result = management.request(c, 'GET', '/api/v1/models/download/status/' + job_id)
    return {'device': device, 'job_id': job_id, **{k: result[k] for k in
            ('status', 'total_size_bytes', 'downloaded_bytes', 'bytes_per_second', 'estimated_completion') if k in result}}


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=True))
def pair_decide(state: Annotated[str, Field(min_length=1, max_length=48000)],
                instructions: Annotated[str, Field(min_length=1, max_length=1000)],
                criteria: dict[str, str], allow_external: bool = False) -> dict:
    """Ask optional cloud Jev for one typed Choice decision, only with explicit external-send opt-in.

    This is not a chat model and is never an implicit fallback for pair_ask.
    The state is sent to TypeSafe AI, not to local PAIR devices.
    """
    return jev.decide(state, instructions, criteria, allow_external=allow_external)


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=True))
def pair_score(state: Annotated[str, Field(min_length=1, max_length=48000)],
               instructions: Annotated[str, Field(min_length=1, max_length=1000)],
               levels: list[str], allow_external: bool = False) -> dict:
    """Ask external TypeSafe AI Jev to score a bounded state on ordered rubric levels.

    Requires explicit allow_external=true; never invoked by local model routing.
    """
    return jev.score(state, instructions, levels, allow_external=allow_external)


if __name__ == '__main__':
    import sys
    if '--self-test' in sys.argv:
        import unittest
        suite = unittest.defaultTestLoader.discover(str(Path(__file__).parent), pattern='test_*.py')
        sys.exit(0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1)
    else:
        mcp.run(transport='stdio')
