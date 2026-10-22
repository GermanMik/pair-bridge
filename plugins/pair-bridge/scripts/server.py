# /// script
# requires-python = ">=3.11,<3.15"
# dependencies = ["mcp==1.29.1", "httpx==0.28.1", "filelock>=3.18,<4", "platformdirs>=4,<5", "jsonschema==4.26.0", "zeroconf==0.151.5"]
# ///
"""Codex MCP tools for the local NVIDIA PAIR OpenAI-compatible proxy."""
from __future__ import annotations

import contextlib
import asyncio
import difflib
import management
import cluster_inventory
import telemetry
import benchmarks
import benchmark_runner
import benchmark_store
import benchmark_suites
import jobs
import inference_io
import download_review
import jev
import diagnostics
import recovery
import hashlib
from filelock import FileLock, Timeout
from platformdirs import user_cache_path
from urllib.parse import urlsplit
import json
import os
import re
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

import httpx
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import Field, BaseModel, ConfigDict, StrictBool

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
_DEVICE_INVENTORY_CACHE: dict[str, dict] = {}
_DEVICE_INVENTORY_CACHE_LOCK = threading.RLock()
_SCRIPT_ROOT = Path(__file__).resolve().parent
BRIDGE_VERSION = json.loads((_SCRIPT_ROOT.parent / '.codex-plugin' / 'plugin.json').read_text(encoding='utf-8'))['version']
SOURCE_BUILD = hashlib.sha256(b''.join((_SCRIPT_ROOT / name).read_bytes() for name in (
    'server.py', 'recovery.py', 'jobs.py', 'diagnostics.py', 'inference_io.py', 'management.py', 'telemetry.py'))).hexdigest()[:16]
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
        'never substitute a model after loading or inference has started. Empty final answers are retried indefinitely '
        'on the same loaded instance with bounded backoff; other errors are not retried. '
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
            result.append(dict({'id': name, 'kind_hint': hint}, **{k: item[k] for k in ('max_context_length', 'max_output_tokens') if k in item}))
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


def cached_loaded_model(device: str, key: str) -> dict | None:
    """Return one last-known loaded chat instance; callers must surface the stale-state warning."""
    with _DEVICE_INVENTORY_CACHE_LOCK:
        cached = _DEVICE_INVENTORY_CACHE.get(device)
    if not cached:
        return None
    row = next((item for item in cached.get('models', []) if item.get('key') == key), None)
    instances = row.get('loaded_instances', []) if isinstance(row, dict) else []
    if row and row.get('type') == 'llm' and len(instances) == 1:
        return row
    return None


def resource_summary(rows: list[dict], cap: int | None) -> dict:
    return {'max_loaded_bytes': cap,
            'loaded_model_weight_bytes': sum(m.get('size_bytes', 0) for m in rows if m['loaded_instances']),
            'note': 'Disk weight sizes are not memory estimates. Cold-load preflight uses the configured engine estimate and fresh RAM/VRAM telemetry when available.'}


def completion(data: dict, requested_model: str, device: str | None, started: float) -> dict:
    choices = data.get('choices')
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise recovery.InferenceFailure('invalid_response')
    choice = choices[0]
    message = choice.get('message')
    if not isinstance(message, dict):
        raise recovery.InferenceFailure('invalid_response', recovery.metadata(choice.get('finish_reason'), data.get('usage')))
    finish_reason = choice.get('finish_reason')
    if finish_reason is not None and not isinstance(finish_reason, str):
        raise recovery.InferenceFailure('invalid_response')
    reasoning = message.get('reasoning_content', message.get('reasoning', '')) if isinstance(message, dict) else ''
    metadata = recovery.metadata(finish_reason, data.get('usage'),
                                 len(reasoning) if isinstance(reasoning, str) else 0)
    try:
        content, content_reasoning, special = recovery.content_text(message.get('content'))
    except recovery.InferenceFailure as exc:
        raise recovery.InferenceFailure(exc.code, metadata) from None
    if content_reasoning:
        metadata = recovery.metadata(finish_reason, data.get('usage'), metadata['reasoning_characters'] + content_reasoning)
    recovery.check_final(content, metadata, dict(message, **special))
    return {
        'device': device, 'route': 'direct_engine' if device else 'pair_router',
        'requested_model': requested_model, 'reported_model': data.get('model'),
        'answer': content, 'finish_reason': choice.get('finish_reason'),
        'truncated': choice.get('finish_reason') == 'length',
        'elapsed_seconds': round(time.monotonic() - started, 2), 'usage': recovery.safe_usage(data.get('usage')),
        'attempt_metadata': metadata,
    }


def _empty_answer_retry_delay(attempt: int) -> float:
    """Back off to at most 30s; retry count itself is deliberately unbounded."""
    return recovery.delay(attempt)


def _request_nonempty_completion(send_request, model: str, device: str | None, started: float,
                                 response_format: dict | None = None, on_empty=None) -> dict:
    """Retry only empty final text, preserving this exact request and target indefinitely."""
    retries = 0
    while True:
        data = send_request()
        try:
            result = inference_io.validate_completion(completion(data, model, device, started), response_format)
        except ValueError as exc:
            if diagnostics.reason(exc) != 'empty_answer':
                raise
            retries += 1
            metadata = dict(getattr(exc, 'metadata', {}), attempt=retries)
            diagnostics.record_attempt(uuid.uuid4().hex, device, model, metadata, retries)
            diagnostics.note_empty_answer_retry(retries)
            if on_empty is not None:
                return on_empty(metadata)
            time.sleep(_empty_answer_retry_delay(retries))
            continue
        result['empty_answer_retries'] = retries
        return result


def _recovery_preflight(c, device, model, instance_id):
    """Read-only checks for each retry; never load or substitute an instance."""
    if device is None:
        raise recovery.InferenceFailure('target_changed')
    warnings = []
    try:
        row = management.find_model(c, model, device)
    except ValueError:
        warnings.append('Live inventory is unconfirmed; attempting the original instance with a warning.')
    else:
        instances = row.get('loaded_instances', [])
        if row.get('type') != 'llm' or len(instances) != 1 or instances[0].get('id') != instance_id:
            raise recovery.InferenceFailure('target_changed')
    config = management.devices().get(device)
    if config is None:
        raise recovery.InferenceFailure('target_changed')
    capacity = management.validate_runtime_capacity(telemetry.sample(config), require_gpu=True)
    return dict(capacity, warnings=[*warnings, *capacity.get('warnings', [])],
                status='warning' if warnings or capacity.get('status') == 'warning' else 'ok')


def _start_completion_recovery(model, device, payload, fmt, instance_id, metadata):
    metadata = recovery.safe_metadata(dict(metadata, **{k: payload[k] for k in ('max_tokens', 'enable_thinking') if k in payload}))
    if recovery.budget_exhausted(metadata):
        metadata['cause'] = 'budget_exhausted'
    pending = jobs.create(device or 'pair-router', model, _run_completion_recovery,
                          device, dict(payload), fmt, instance_id, metadata)
    return {'status': 'recovering', 'job_id': pending['job_id'], 'device': device,
            'requested_model': model, 'empty_answer_retries': 1,
            'attempt_metadata': recovery.safe_metadata(metadata),
            'notice': 'Use pair_job_status, pair_job_reconfigure and pair_job_cancel. Retry attempts have no limit.'}


def _run_completion_recovery(job, device, payload, fmt, instance_id, metadata):
    """Continue a synchronous empty completion as an observable cancellable job."""
    job.recovery_initial_preflight = True
    job.recovery_unpinned = device is None
    job.recovery_device = device
    job.recovery_target_instance = instance_id
    job.recovery_parameters = {k: payload[k] for k in ('max_tokens', 'enable_thinking') if k in payload}
    job.update(empty_answer_retries=1, attempt_metadata=metadata, instance_id=instance_id)
    diagnostics.record_attempt(job.id, job.device, job.model, metadata, 1)
    if recovery.budget_exhausted(metadata):
        job.recovery_wait_revision = job.recovery_revision
    loop = None
    try:
        job.update(status='running', stage='queue')
        def preflight():
            with job.lock:
                selected_device, selected_instance = job.recovery_device, job.instance_id
            with management.client(selected_device) as c:
                return _recovery_preflight(c, selected_device, job.model, selected_instance)
        async def send():
            with job.lock:
                selected_device = job.recovery_device
                current = dict(payload, **job.recovery_parameters)
                selected_instance = job.instance_id
            # Release the per-device lock while waiting; recheck after acquiring it
            # so another client cannot make the earlier capacity snapshot authoritative.
            with inference_lock(selected_device, wait_seconds=30), management.client(selected_device) as c:
                capacity = _recovery_preflight(c, selected_device, job.model, selected_instance)
                job.update(memory_preflight=capacity)
                current['model'] = management.chat_model_id(selected_device, job.model, selected_instance)
                if job.cancel_event.is_set():
                    raise asyncio.CancelledError()
                data = management.request(c, 'POST', '/v1/chat/completions', current)
            row = inference_io.validate_completion(completion(data, job.model, selected_device, time.monotonic()), fmt)
            job.answer = row['answer']
            job.finish_reason = row['finish_reason']
            job.output_truncated = row['truncated']
            job.structured_output = row.get('structured_output')
            job.attempt_metadata = row['attempt_metadata']
            return job.answer
        loop = asyncio.new_event_loop()
        task = loop.create_task(_retry_job_empty_answer(job, send, preflight))
        with job.lock:
            job.loop, job.task = loop, task
        loop.run_until_complete(task)
        if job.cancel_event.is_set():
            raise asyncio.CancelledError()
        job.update(status='completed', stage='completed', cleanup='existing_instance_preserved')
    except asyncio.CancelledError:
        job.update(status='cancelled', stage='cancelled')
    except Exception as exc:
        job.update(status='cancelled' if job.cancel_event.is_set() else 'failed', stage='failed',
                   error_code=diagnostics.reason(exc),
                   attempt_metadata=getattr(exc, 'metadata', job.attempt_metadata))
    finally:
        with job.lock:
            job.loop, job.task = None, None
        if loop is not None:
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()


def rank_models(inventory: list[dict], model: str | None = None, device: str | None = None,
                context_length: int = 8192, task_hint: str = 'general',
                max_load_bytes: int | None = None) -> list[tuple[str, dict]]:
    """Return installed LLMs in deterministic preference order."""
    candidates = []
    for row in inventory:
        if (row.get('online') is not True and row.get('availability') != 'unknown') or \
                (device is not None and row.get('device') != device):
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


def _validate_smart_candidate(item: dict, context_length: int, max_tokens: int | None = None) -> None:
    inference_io.output_budget(item, max_tokens, context_length)
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
                            device: str | None = None, max_tokens: int | None = None) -> tuple[str, dict, dict, list[dict], list[str]]:
    """Choose the highest-ranked model that passes preflight, then try configured evictions."""
    ranked = rank_models(inventory, device=device, context_length=context_length, task_hint=task_hint,
                         max_load_bytes=max_load_bytes)
    resident = [candidate for candidate in ranked if candidate[1].get('loaded_instances')]
    cold = [candidate for candidate in ranked if not candidate[1].get('loaded_instances')]
    ranked = resident + cold
    last_success = diagnostics.last_successful_model(device=device)
    if last_success is None and device is not None:
        last_success = diagnostics.last_successful_model()
    last_candidate = next((candidate for candidate in cold
                           if last_success and (last_success.get('device') is None
                                                or candidate[0] == last_success.get('device'))
                           and candidate[1].get('key') == last_success.get('model')), None)
    if last_candidate is not None:
        ranked = resident + [last_candidate] + [candidate for candidate in cold if candidate != last_candidate]
    errors = []
    configs = management.devices()
    all_unloaded = []
    for target, item in ranked:
        device_row = next(row for row in inventory if row.get('device') == target)
        try:
            _validate_smart_candidate(item, context_length, max_tokens)
            if item.get('loaded_instances'):
                return target, item, {'status': 'already_loaded', 'selection_reason': 'reused_loaded_model'}, [], errors
            config = configs[target]
            configured_cap = config.get('max_loaded_bytes')
            cap = min(configured_cap, max_load_bytes) if configured_cap and max_load_bytes else (configured_cap or max_load_bytes)
            memory = management.memory_preflight(target, device_row['models'], item, context_length,
                                                 cap, telemetry.sample(config))
            reason = 'last_successful_model' if last_candidate == (target, item) else 'memory_fit_rank'
            return target, item, dict(memory, selection_reason=reason), [], errors
        except ValueError as exc:
            errors.append(f"{target}/{item['key']}: {exc}")

    for target, item in ranked:
        config = configs[target]
        if not config.get('auto_unload_models'):
            continue
        try:
            _validate_smart_candidate(item, context_length, max_tokens)
            configured_cap = config.get('max_loaded_bytes')
            cap = min(configured_cap, max_load_bytes) if configured_cap and max_load_bytes else (configured_cap or max_load_bytes)
            with inference_lock(target, wait_seconds=30), management.client(target) as c:
                prepared = management.preflight_for_load(
                    c, target, item['key'], context_length, config, cap,
                    telemetry.sample(config), lambda: telemetry.sample(config))
            all_unloaded.extend(dict(row, device=target) for row in prepared['auto_unloaded_instances'])
            _validate_smart_candidate(prepared['candidate'], context_length, max_tokens)
            reason = 'last_successful_model' if last_candidate == (target, item) else 'memory_fit_rank'
            memory = dict(prepared['memory'], selection_reason=reason)
            return target, prepared['candidate'], memory, all_unloaded, errors
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
def pair_benchmark_results(profile: Annotated[str | None, Field(pattern='^(general|code|fast|long_context|analysis)$')] = None,
                           context_length: Annotated[int, Field(ge=512, le=262144)] = 8192,
                           device: str | None = None, model: str | None = None,
                           suite: Annotated[str | None, Field(pattern='^(smoke|short|full)$')] = None,
                           run_id: str | None = None,
                           limit: Annotated[int, Field(ge=1, le=200)] = 50,
                           offset: Annotated[int, Field(ge=0)] = 0) -> dict:
    """Read durable suite runs, or legacy profile routing summaries when profile is given.

    Omit profile for quality/speed/error tables. Poll by run_id even after restart.
    """
    if profile is not None:
        if any(value is not None for value in (device, model, suite, run_id)) or limit != 50 or offset != 0:
            raise ValueError('Use profile for routing summaries, or suite filters without profile')
        scores = benchmarks.summaries(profile, context_length)
        return {'profile': profile, 'minimum_context_length': context_length,
                'results': [dict(device=d, model=m, **value) for (d, m), value in sorted(scores.items())]}
    return benchmark_store.results(device, model, suite, run_id, limit, offset)


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False))
def pair_benchmark_suites() -> dict:
    """Discover versioned nested smoke/short/full suites and their objective checks."""
    return benchmark_suites.catalog()


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False))
def pair_benchmark_run(device: str, model: str,
                       suite: Annotated[str, Field(pattern='^(smoke|short|full)$')] = 'short',
                       repeats: Annotated[int, Field(ge=1, le=5)] = 3,
                       max_tokens: Annotated[int, Field(ge=32)] = 256,
                       request_timeout_seconds: Annotated[int, Field(ge=30, le=1200)] = 180,
                       enable_thinking: bool | None = None) -> dict:
    """Start a benchmark job on one exact already-loaded model; never loads/retries.

    Poll pair_job_status(job_id), cancel with pair_job_cancel(job_id).
    Durable metrics are available via pair_benchmark_results(run_id), including partial failures.
    """
    data = benchmark_runner.validate_request(device, model, suite, repeats, max_tokens,
                                              request_timeout_seconds, enable_thinking)
    row = benchmark_store.create(device, model, data, repeats, max_tokens,
                                 request_timeout_seconds, enable_thinking)
    try:
        job = jobs.create(device, model, benchmark_runner.worker, row, data, inference_lock)
    except Exception:
        row.update(status='failed', error_code='job_start_failed')
        benchmark_store.persist(row)
        raise
    if job['status'] == 'failed' and job.get('error_code') == 'worker_start_failed':
        row.update(status='failed', error_code='worker_start_failed')
        benchmark_store.persist(row)
    return {'job_id': job['job_id'], 'run_id': row['run_id'], 'device': device, 'model': model,
            'suite': suite, 'suite_version': data['suite_version'], 'status': job['status'],
            'expected_samples': len(data['cases']) * repeats,
            'request_timeout_seconds': request_timeout_seconds, 'enable_thinking': enable_thinking,
            'notice': 'One warmup excluded. No model loading, retries or fallback; prompts and answers are not saved.'}


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False))
def pair_benchmark_compare(run_ids: list[str]) -> dict:
    """Compare 1-20 completed runs of the same workload; return metrics and Markdown table."""
    return benchmark_store.compare(run_ids)


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
        if device.startswith('pair-') and device not in management.devices():
            inventory = cluster_inventory.inspect()
            for node in inventory['nodes']:
                if node['device'] == device:
                    return node
            raise ValueError('Unknown native PAIR node; refresh pair_devices')
        with management.client(device) as c:
            rows = management.models(c, device)
            return {'device': device, 'online': True, 'checked_at': datetime.now(timezone.utc).isoformat(),
                    'models': checked_models(device, rows),
                    'resources': resource_summary(rows, management.devices()[device].get('max_loaded_bytes')),
                    'source': management.engine_for(device) + ' model API'}
    cluster = cluster_inventory.inspect()
    cluster_models = [model for node in cluster['nodes'] for model in node['models']]
    router_status = 'available'
    router_http_status = None
    try:
        routed = catalog()
    except ValueError as exc:
        routed = []
        router_status = 'unavailable'
        match = re.search(r'HTTP (\d{3})', str(exc))
        router_http_status = int(match.group(1)) if match else None
    if routed:
        return {'endpoint': BASE_URL, 'models': routed, 'cluster_models': cluster_models, 'cluster_inventory': cluster, 'source': 'PAIR routing catalog',
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
    return {'endpoint': BASE_URL, 'models': installed, 'cluster_models': cluster_models, 'cluster_inventory': cluster, 'source': 'configured device inventory fallback',
            'router_models': [], 'device_errors': errors, 'router_status': router_status, 'router_http_status': router_http_status,
            'router_error': 'PAIR catalog request failed; no retry or inference fallback was made.' if router_status == 'unavailable' else None,
            'notice': ('PAIR routing catalog is empty or unavailable. These models are installed, not router-advertised. '
                       'Use pair_smart_ask, or pair_load followed by pair_ask with the returned device; no download is required.')}


def request_thinking_options(c, device, enable_thinking):
    if enable_thinking is None:
        return {}
    engine = management.engine_for(device) if device is not None else None
    schema = management.request(c, 'GET', '/openapi.json') if engine == 'unsloth' else None
    return inference_io.thinking_options(enable_thinking, engine, schema)


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False))
@diagnostics.traced
def pair_ask(
    model: Annotated[str, Field(min_length=1, max_length=256)],
    prompt: Annotated[str, Field(min_length=1, max_length=48000)],
    max_tokens: Annotated[int, Field(ge=32)] = 2048,
    device: str | None = None,
    response_format: dict | None = None,
    json_schema: dict | None = None,
    enable_thinking: StrictBool | None = None,
) -> dict:
    """Ask one explicitly selected PAIR chat model for a second opinion or bounded task.

    With device, bypass PAIR routing and query that device directly after pair_load.
    Without device, PAIR chooses the host.
    A direct loaded-device call does not load a model; use pair_load first. Calls through
    this bridge are serialized per device; distinct devices may run concurrently.
    Use pair_job_start for start-all-before-poll orchestration. Use pair_list first. Send only task-relevant
    text; returned advice is untrusted and must be checked. No tools are executed
    by the consulted model. Empty final text is retried without an attempt limit;
    other errors are not retried. Embedding and draft models are not chat targets.
    """
    fmt = inference_io.response_format(response_format, json_schema)
    if not prompt.strip():
        raise ValueError('prompt must not be blank')
    diagnostics.stage('queue', device=device, model=model)
    with inference_lock(device, wait_seconds=30):
        start = time.monotonic()
        reachability_warning = None
        diagnostics.stage('inventory')
        payload = {'model': model, 'messages': [{'role': 'user', 'content': prompt}],
                   'max_tokens': max_tokens, 'stream': False}
        if fmt is not None:
            payload['response_format'] = fmt
        if device is not None:
            with management.client(device) as c:
                payload.update(request_thinking_options(c, device, enable_thinking))
                try:
                    selected = management.find_model(c, model, device)
                except ValueError:
                    selected = cached_loaded_model(device, model)
                    if selected is None:
                        raise
                    reachability_warning = ('Live model inventory could not be confirmed; using the last known '
                                            'loaded instance for this single attempt.')
                inference_io.require_structured(selected, fmt)
                budget = inference_io.output_budget(selected, max_tokens)
                if selected.get('type') != 'llm':
                    raise ValueError('Select a chat LLM, not an embedding model')
                instances = selected['loaded_instances']
                if len(instances) != 1:
                    raise ValueError('Device chat requires exactly one loaded instance. Use pair_load or resolve multiple instances first')
                memory_check = management.validate_runtime_capacity(
                    telemetry.sample(management.devices()[device]), require_gpu=True)
                payload['model'] = management.chat_model_id(device, model, instances[0]['id'])
                diagnostics.stage('inference')
                result = _request_nonempty_completion(
                    lambda: management.request(c, 'POST', '/v1/chat/completions', payload),
                    model, device, start, fmt,
                    on_empty=lambda metadata: _start_completion_recovery(
                        model, device, payload, fmt, instances[0]['id'], metadata))
        else:
            inference_io.thinking_options(enable_thinking, None)
            available = {item['id']: item for item in catalog()}
            if model not in available:
                raise ValueError('Model is no longer advertised by PAIR. Refresh pair_list and use an exact ID.')
            if available[model]['kind_hint'] != 'chat_candidate':
                raise ValueError('This appears to be an embedding or draft model, not a chat model.')
            budget = inference_io.output_budget(available[model], max_tokens)
            diagnostics.stage('inference')
            result = _request_nonempty_completion(
                lambda: request('POST', '/chat/completions', payload), model, device, start, fmt,
                on_empty=lambda metadata: _start_completion_recovery(model, None, payload, fmt, None, metadata))
        diagnostics.stage('validation')
        result = dict(result, output_budget=budget)
        if device is not None:
            result['memory_preflight'] = memory_check
        if reachability_warning:
            result['reachability_warnings'] = [reachability_warning]
        return result



@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False))
def pair_capabilities() -> dict:
    """Discover concurrent multi-PC model jobs for Codex, Claude and any MCP client.

    Start jobs on different explicit devices before polling; calls may be sequential
    while background inference overlaps. This is not distributed inference of one model.
    """
    return {
        'bridge_version': BRIDGE_VERSION,
        'source_build': SOURCE_BUILD,
        'empty_answer_recovery': {'attempt_limit': None, 'backoff_max_seconds': 30,
                                  'budget_exhaustion': 'wait_for_explicit_reconfiguration',
                                  'sync_handoff': 'job_id', 'reconfigure_tool': 'pair_job_reconfigure',
                                  'fresh_retry_preflight': True},
        'result_logs': ['pair_logs', 'pair_logs_export'],
        'model_downloads_supported': False,
        'model_loading': 'installed models into RAM/VRAM only',
        'cross_device_parallel_jobs': True,
        'model_capability_discovery': 'pair_model_capabilities',
        'structured_output': ['pair_ask', 'pair_smart_ask', 'pair_job_start', 'pair_batch_start', 'pair_vision_ask'],
        'job_listing': 'pair_job_list',
        'batch_jobs': ['pair_batch_start', 'pair_batch_status', 'pair_batch_cancel'],
        'embeddings': 'pair_embeddings (explicit already-loaded embedding instance)',
        'vision': 'pair_vision_ask (confirmed vision metadata, inline images)',
        'benchmarks': {'suites': 'pair_benchmark_suites', 'start': 'pair_benchmark_run',
                       'results': 'pair_benchmark_results (omit profile)', 'compare': 'pair_benchmark_compare',
                       'requires': 'one exact already-loaded chat instance', 'durable_key': 'run_id'},
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
    inventory plus native paired-node read-only inventory. No model is loaded by this call.
    """
    result = []
    for name, config in management.devices().items():
        try:
            with management.client(name) as c:
                models = management.models(c, name)
                enriched = checked_models(name, models)
                result.append({'device': name, 'online': True,
                               'checked_at': datetime.now(timezone.utc).isoformat(),
                               'models': enriched,
                               'resources': resource_summary(models, config.get('max_loaded_bytes')),
                               'capacity': telemetry.sample(config)})
                with _DEVICE_INVENTORY_CACHE_LOCK:
                    _DEVICE_INVENTORY_CACHE[name] = {'checked_at': datetime.now(timezone.utc).isoformat(),
                                                     'models': enriched}
        except ValueError as exc:
            failure = diagnostics.device_failure(exc)
            with _DEVICE_INVENTORY_CACHE_LOCK:
                cached = _DEVICE_INVENTORY_CACHE.get(name)
            if cached:
                result.append({'device': name, 'online': None, 'availability': 'unknown',
                               'reachability_warning': 'Live inventory check failed; using the last known installed-model inventory. The selected device will be rechecked before inference.',
                               'checked_at': cached['checked_at'], 'models': cached['models'],
                               'capacity': {'status': 'partial', 'reason': 'Live device inventory unavailable'},
                               'last_check_error': failure.get('reason', 'inspection_failed')})
            else:
                result.append({'device': name, 'checked_at': datetime.now(timezone.utc).isoformat(),
                               'check_status': 'authentication_required' if failure.get('auth_status') else ('unreachable' if failure['online'] is False else 'inspection_failed'),
                               **failure})
    cluster = cluster_inventory.inspect()
    return {'devices': result + cluster['nodes'], 'cluster_inventory_status': cluster['status'],
            'capabilities': pair_capabilities(), 'notice': cluster['notice']}


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False))
@diagnostics.traced
def pair_load(device: str, model: str,
              context_length: Annotated[int, Field(ge=512, le=262144)] = 8192) -> dict:
    """Load an installed model on one device into RAM/VRAM; never download weights.

    Reuses existing loaded instances without changing their configuration. Cold
    loads use an engine estimate and fresh Bridge RAM/VRAM telemetry when available,
    preserving a 1 GiB free reserve. Unknown telemetry is returned as a warning.
    May trigger engine auto-eviction; verify the result.
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
    max_tokens: Annotated[int, Field(ge=32)] = 2048,
    unload_after: bool = True,
    task_hint: Annotated[str, Field(pattern='^(general|code|fast|long_context|analysis)$')] = 'general',
    max_load_bytes: Annotated[int | None, Field(ge=1)] = None,
    response_format: dict | None = None,
    json_schema: dict | None = None,
    enable_thinking: StrictBool | None = None,
) -> dict:
    """Select a memory-fit installed LLM unless one is explicit; retry empty answers on the same instance.

    No model downloads. Automatic selection checks memory before choosing; an explicit model is never substituted.
    Existing loaded instances are preserved. Cold-load preflight and telemetry warnings are returned in
    memory_preflight. An ambiguous multi-instance model is not selected automatically.
    """
    fmt = inference_io.response_format(response_format, json_schema)
    thinking = {}
    if enable_thinking is not None:
        if device is None:
            raise ValueError('Explicit thinking control requires a configured Unsloth device')
        with management.client(device) as control_client:
            thinking = request_thinking_options(control_client, device, enable_thinking)
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
    reachability_warnings = [f"{row['device']}: {row['reachability_warning']}" for row in snapshot['devices']
                             if row.get('reachability_warning')]
    preselected_unloaded = []
    memory_rejections = []
    if model is None:
        selected_device, selected, selection_memory, preselected_unloaded, memory_rejections = select_model_for_memory(
            snapshot['devices'], context_length, task_hint, max_load_bytes, device=device, max_tokens=max_tokens)
    else:
        selected_device, selected = select_model(snapshot['devices'], model, device, context_length,
                                                 task_hint, max_load_bytes)
        selection_memory = {'selection_reason': 'explicit model/device'}
    measured = benchmarks.summaries(task_hint, context_length).get((selected_device, selected['key'])) if model is None else None
    diagnostics.stage('queue', device=selected_device, model=selected['key'])
    with inference_lock(selected_device, wait_seconds=30):
        with management.report_releases_on_error(lambda: preselected_unloaded):
            key = selected['key']
            started = time.monotonic()
            owned_id = None
            cleanup = 'not_needed'
            load_time_seconds = None
            memory = {'status': 'already_loaded'}
            engine_evicted_instances = []
            with management.client(selected_device) as c:
                # Recheck after selection: another application may have changed the load state.
                try:
                    live = management.find_model(c, key, selected_device)
                except ValueError:
                    live = cached_loaded_model(selected_device, key)
                    if live is None:
                        raise
                    reachability_warnings.append(
                        f'{selected_device}: live model inventory check failed; using the last known loaded instance for one attempt.')
                inference_io.require_structured(live, fmt)
                inference_io.output_budget(live, max_tokens, context_length)
                if live.get('type') != 'llm':
                    raise ValueError('Selected model is no longer a chat LLM; refresh inventory')
                limit = live.get('max_context_length')
                if isinstance(limit, int) and limit < context_length:
                    raise ValueError('Selected model context is now smaller than requested; refresh inventory')
                instances = live['loaded_instances']
                if len(instances) > 1:
                    raise ValueError('Multiple instances of the selected model are loaded; choose and manage one explicitly')
                device_config = management.devices()[selected_device]
                if instances:
                    load_config = instances[0].get('config')
                    actual_context = load_config.get('context_length') if isinstance(load_config, dict) else None
                    if isinstance(actual_context, int) and actual_context < context_length:
                        raise ValueError('Loaded instance context is smaller than requested; choose a smaller context or another model')
                    diagnostics.stage('preflight')
                    memory['runtime_capacity'] = management.validate_runtime_capacity(
                        telemetry.sample(device_config), require_gpu=True)
                    if memory['runtime_capacity']['status'] == 'warning':
                        memory['status'] = 'warning'
                        memory['warnings'] = memory['runtime_capacity'].get('warnings', [])
                if not instances:
                    diagnostics.stage('preflight')
                    before_rows = management.models(c, selected_device)
                    candidate = next((m for m in before_rows if m['key'] == key), None)
                    if candidate is None:
                        raise ValueError('Selected model disappeared before load; refresh inventory')
                    inference_io.output_budget(candidate, max_tokens, context_length)
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
                    inference_io.output_budget(candidate, max_tokens, context_length)
                    diagnostics.stage('load')
                    before_ids = {i['id'] for m in before_rows for i in m['loaded_instances']}
                    if candidate['loaded_instances']:
                        if len(candidate['loaded_instances']) != 1:
                            raise ValueError('Multiple instances of the selected model became loaded; inspect state')
                        instances = candidate['loaded_instances']
                        live = candidate
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
                        live = after
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
                runtime_memory = management.validate_runtime_capacity(telemetry.sample(device_config), require_gpu=True)
                memory['runtime_capacity'] = runtime_memory
                if runtime_memory['status'] == 'warning':
                    memory['status'] = 'warning'
                    memory['warnings'] = list(dict.fromkeys([
                        *memory.get('warnings', []), *runtime_memory.get('warnings', [])]))
                budget = inference_io.output_budget(dict(live, loaded_instances=instances), max_tokens)
                payload = {'model': management.chat_model_id(selected_device, key, instances[0]['id']),
                           'messages': [{'role': 'user', 'content': prompt}],
                           'max_tokens': max_tokens, 'stream': False}
                payload.update(thinking)
                if fmt is not None:
                    payload['response_format'] = fmt
                try:
                    diagnostics.stage('inference')
                    result = _request_nonempty_completion(
                        lambda: management.request(c, 'POST', '/v1/chat/completions', payload),
                        key, selected_device, started, fmt,
                        on_empty=lambda metadata: _start_completion_recovery(
                            key, selected_device, payload, fmt, instances[0]['id'], metadata))
                    diagnostics.stage('validation')
                except Exception as exc:
                    # A timeout may leave inference running. Retain the instance for inspection.
                    details = management.describe_unloaded_instances(preselected_unloaded)
                    if details:
                        error = ValueError(f'{exc}; configured auto_unload_models released: {details}')
                        error.auto_unloaded_instances = preselected_unloaded
                        raise error from exc
                    raise
                else:
                    if owned_id and unload_after and result.get('status') != 'recovering':
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
                    elif selection_memory.get('selection_reason') == 'reused_loaded_model':
                        selection_reason = 'reused an eligible already-loaded model'
                    elif selection_memory.get('selection_reason') == 'last_successful_model':
                        selection_reason = 'loaded the most recently successful model'
                    elif measured and measured['pass_rate'] >= .6:
                        selection_reason = (f"benchmark: {measured['passed']}/{measured['cases']} cases, "
                                            f"median {measured['median_latency_ms']} ms among memory-fit candidates")
                    else:
                        selection_reason = 'highest-ranked installed chat model that passes memory preflight'
                    return dict(result, output_budget=budget, selected_model=key, instance_id=instances[0]['id'],
                                loaded_for_request=bool(owned_id), cleanup=cleanup,
                                selection_profile=task_hint,
                                selection_reason=selection_reason,
                                load_time_seconds=load_time_seconds,
                                engine_evicted_instances=engine_evicted_instances,
                                auto_unloaded_instances=preselected_unloaded,
                                memory_preflight_rejections=memory_rejections,
                                memory_preflight=memory,
                                reachability_warnings=reachability_warnings,
                                router_status=router_status, router_advertises_model=key in router_models)


async def _job_stream(job: jobs.Job, c: httpx.AsyncClient, instance_id: str, prompt: str, max_tokens: int,
                      response_format: dict | None = None, enable_thinking: bool | None = None) -> str:
    """Consume engine SSE; retain message text only, never reasoning/tool content."""
    if management.engine_for(job.device) == 'unsloth' or response_format is not None:
        payload = {'model': management.chat_model_id(job.device, job.model, instance_id),
                   'messages': [{'role': 'user', 'content': prompt}],
                   'max_tokens': max_tokens, 'stream': True}
        if response_format is not None:
            payload['response_format'] = response_format
        if enable_thinking is not None:
            payload['enable_thinking'] = enable_thinking
        ended = False
        usage, reasoning_characters, special = {}, 0, {}
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
                usage.update(recovery.safe_usage(data.get('usage')))
                choices = data.get('choices')
                if not isinstance(choices, list) or not choices:
                    if choices not in (None, []) or (choices is None and 'usage' not in data):
                        raise recovery.InferenceFailure('invalid_response')
                    continue
                first = choices[0]
                if not isinstance(first, dict):
                    raise recovery.InferenceFailure('invalid_response')
                if isinstance(first, dict) and first.get('finish_reason') is not None:
                    reason = first['finish_reason']
                    if not isinstance(reason, str):
                        raise recovery.InferenceFailure('invalid_response')
                    job.finish_reason = reason if isinstance(reason, str) and reason in (
                        'stop', 'length', 'tool_calls', 'content_filter', 'function_call') else 'unknown'
                delta = first.get('delta') if isinstance(first, dict) else None
                if delta is not None and not isinstance(delta, dict):
                    raise recovery.InferenceFailure('invalid_response')
                if isinstance(delta, dict):
                    reasoning = delta.get('reasoning_content', delta.get('reasoning'))
                    if isinstance(reasoning, str):
                        reasoning_characters += len(reasoning)
                    for key in ('refusal', 'tool_calls', 'function_call'):
                        if delta.get(key):
                            special[key] = True
                try:
                    text, part_reasoning, part_special = recovery.content_text(
                        delta.get('content') if isinstance(delta, dict) else None, separator='')
                except recovery.InferenceFailure as exc:
                    raise recovery.InferenceFailure(exc.code, recovery.metadata(
                        job.finish_reason, usage, reasoning_characters)) from None
                reasoning_characters += part_reasoning
                special.update(part_special)
                if text:
                    job.add_text(text)
                    job.update(stage='generating', progress=None)
        if job.cancel_event.is_set():
            raise asyncio.CancelledError()
        if not ended:
            raise ValueError('Device stream ended before [DONE]')
        job.attempt_metadata = recovery.metadata(job.finish_reason, usage, reasoning_characters)
        recovery.check_final(job.answer, job.attempt_metadata, special)
        job.output_truncated = job.output_truncated or job.finish_reason == 'length'
        return job.answer

    payload = {'model': instance_id, 'input': prompt, 'max_output_tokens': max_tokens,
               'stream': True, 'store': False, 'integrations': []}
    ended = False
    native_usage, native_reasoning = {}, 0
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
                elif kind == 'reasoning.delta' and isinstance(data.get('content'), str):
                    native_reasoning += len(data['content'])
                elif kind == 'error':
                    raise ValueError('Device reported a streaming error')
                elif kind == 'chat.end':
                    result = data.get('result')
                    if not isinstance(result, dict) or not isinstance(result.get('output'), list):
                        raise recovery.InferenceFailure('invalid_response')
                    output = result['output']
                    final_parts, special = [], {}
                    result_reasoning = 0
                    for item in output:
                        if not isinstance(item, dict):
                            raise recovery.InferenceFailure('invalid_response')
                        kind = item.get('type')
                        if kind in ('tool_call', 'invalid_tool_call', 'function_call'):
                            special['tool_calls'] = True
                        elif kind == 'refusal':
                            special['refusal'] = True
                        elif kind in ('message', 'reasoning') and isinstance(item.get('content'), str):
                            if kind == 'message':
                                final_parts.append(item['content'])
                            else:
                                result_reasoning += len(item['content'])
                        else:
                            raise recovery.InferenceFailure('invalid_response')
                    final = ''.join(final_parts)
                    native_reasoning = max(native_reasoning, result_reasoning)
                    with job.lock:
                        if not job.cancel_event.is_set():
                            job.answer = final[:48000]
                            job.output_truncated = len(final) > 48000
                    ended = True
                    stats = result.get('stats') if isinstance(result, dict) else None
                    stats = stats if isinstance(stats, dict) else {}
                    reason = result.get('finish_reason', data.get('finish_reason', stats.get('stop_reason')))
                    mapped = ({'maxPredictedTokensReached': 'length', 'eosFound': 'stop',
                               'stopStringFound': 'stop'}.get(reason, reason) if isinstance(reason, str) else None)
                    job.finish_reason = mapped if isinstance(mapped, str) and mapped in recovery.FINISH_REASONS else 'unknown'
                    native_usage = recovery.safe_usage(result.get('usage'))
                    native_usage.update(recovery.safe_usage({
                        'prompt_tokens': stats.get('input_tokens'),
                        'completion_tokens': stats.get('total_output_tokens'),
                        'reasoning_tokens': stats.get('reasoning_output_tokens')}))
                    job.attempt_metadata = recovery.metadata(job.finish_reason, native_usage, native_reasoning)
                    recovery.check_final(job.answer, job.attempt_metadata, special)
                event_type, data_text = None, ''
    if job.cancel_event.is_set():
        raise asyncio.CancelledError()
    if not ended:
        raise ValueError('Device stream ended before chat.end')
    job.attempt_metadata = recovery.metadata(job.finish_reason, native_usage, native_reasoning)
    recovery.check_final(job.answer, job.attempt_metadata)
    job.output_truncated = job.output_truncated or job.finish_reason == 'length'
    return job.answer


async def _retry_job_empty_answer(job: jobs.Job, send_request, before_attempt=None) -> str:
    """Retry confirmed empty completions indefinitely; explicit job cancellation stops the loop."""
    retries = job.empty_answer_retries
    while True:
        if job.cancel_event.is_set():
            raise asyncio.CancelledError()
        if job.recovery_unpinned:
            job.update(stage='waiting_for_device', next_retry_at=None,
                       recovery_advice={'reason': 'router_target_unconfirmed',
                                        'action': 'Use pair_job_reconfigure with an explicit configured device containing this exact loaded model. No further inference is sent until you choose it.'})
            await asyncio.sleep(.25)
            continue
        if job.recovery_wait_revision is not None:
            with job.lock:
                waiting = job.recovery_revision == job.recovery_wait_revision
            if waiting:
                job.update(stage='waiting_for_configuration', next_retry_at=None,
                           recovery_advice={'reason': 'budget_exhausted',
                                            'action': 'Use pair_job_reconfigure with an explicitly approved budget or thinking setting.',
                                            'context_and_output_target_tokens': 40000,
                                            'output_budget': job.output_budget,
                                            'saved_profile': recovery.load_profile(job.device, job.model)})
                await asyncio.sleep(.25)
                continue
            job.recovery_wait_revision = None
        if before_attempt is not None and (retries or job.recovery_initial_preflight):
            try:
                capacity = before_attempt()
            except ValueError as exc:
                if not management.is_capacity_error(exc):
                    raise
                job.update(stage='waiting_for_resources', next_retry_at=time.time() + 1,
                           memory_preflight={'status': 'insufficient', 'reserve_bytes': management.MEMORY_RESERVE_BYTES})
                await asyncio.sleep(1)
                continue
            job.update(memory_preflight=capacity)
        with job.lock:
            job.answer = ''
            job.finish_reason = None
            job.output_truncated = False
        job.update(stage='inference', next_retry_at=None, recovery_advice=None)
        try:
            answer = await send_request()
            recovery.check_final(answer, job.attempt_metadata)
            with job.lock:
                job.answer = answer[:48000]
                job.output_truncated = job.output_truncated or len(answer) > 48000
            metadata = dict(job.attempt_metadata, attempt=retries + 1, **job.recovery_parameters)
            job.attempt_metadata = recovery.safe_metadata(metadata)
            diagnostics.record_attempt(job.id, job.device, job.model, metadata, retries, 'attempt_completed')
            job.update(empty_answer_retries=retries)
            return answer
        except ValueError as exc:
            if management.is_capacity_error(exc):
                job.update(stage='waiting_for_resources', next_retry_at=time.time() + 1,
                           memory_preflight={'status': 'insufficient', 'reserve_bytes': management.MEMORY_RESERVE_BYTES})
                await asyncio.sleep(1)
                continue
            if diagnostics.reason(exc) != 'empty_answer':
                if isinstance(exc, recovery.InferenceFailure):
                    job.attempt_metadata = exc.metadata
                raise
            retries += 1
            metadata = recovery.safe_metadata(dict(getattr(exc, 'metadata', {}), attempt=retries,
                                                    **job.recovery_parameters))
            exhausted = recovery.budget_exhausted(metadata)
            if exhausted:
                metadata['cause'] = 'budget_exhausted'
            diagnostics.record_attempt(job.id, job.device, job.model, metadata, retries)
            delay = _empty_answer_retry_delay(retries)
            job.update(stage='retrying_empty_answer', empty_answer_retries=retries,
                       attempt_metadata=metadata, next_retry_at=time.time() + delay)
            if exhausted:
                with job.lock:
                    job.recovery_wait_revision = job.recovery_revision
                continue
            await asyncio.sleep(delay)


def _run_job(job: jobs.Job, prompt: str, context_length: int, max_tokens: int, unload_after: bool,
             response_format: dict | None = None, enable_thinking: bool | None = None) -> None:
    owned_id = None
    try:
        with inference_lock(job.device, wait_seconds=30), management.client(job.device) as c:
            if job.cancel_event.is_set():
                job.update(status='cancelled', stage='cancelled')
                return
            job.update(status='running', stage='inventory')
            request_thinking_options(c, job.device, enable_thinking)
            rows = management.models(c, job.device)
            selected = next((m for m in rows if m['key'] == job.model), None)
            if selected is None or selected.get('type') != 'llm':
                raise ValueError('Model is not an installed chat LLM on this device')
            inference_io.require_structured(selected, response_format)
            inference_io.output_budget(selected, max_tokens, context_length)
            maximum = selected.get('max_context_length')
            if isinstance(maximum, int) and context_length > maximum:
                raise ValueError('Requested context exceeds model maximum')
            instances = selected['loaded_instances']
            if len(instances) > 1:
                raise ValueError('Multiple model instances are loaded; choose one explicitly')
            config = management.devices()[job.device]
            if instances:
                memory = {'status': 'already_loaded'}
                actual = (instances[0].get('config') or {}).get('context_length')
                if isinstance(actual, int) and actual < context_length:
                    raise ValueError('Loaded instance context is smaller than requested')
                instance_id = instances[0]['id']
            else:
                job.update(stage='preflight')
                prepared = management.preflight_for_load(
                    c, job.device, job.model, context_length, config,
                    config.get('max_loaded_bytes'), telemetry.sample(config),
                    lambda: telemetry.sample(config))
                selected = prepared['candidate']
                inference_io.output_budget(selected, max_tokens, context_length)
                job.update(auto_unloaded_instances=prepared['auto_unloaded_instances'],
                           memory_preflight=prepared['memory'])
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
                    selected = management.find_model(c, job.model, job.device)
                    confirmed = selected['loaded_instances']
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
                memory = prepared['memory']
            try:
                runtime_capacity = management.validate_runtime_capacity(
                    telemetry.sample(config), require_gpu=True)
            except ValueError as exc:
                if management.is_capacity_error(exc):
                    memory['runtime_capacity'] = {
                        'status': 'insufficient', 'reserve_bytes': management.MEMORY_RESERVE_BYTES,
                        'error': str(exc)}
                    job.update(memory_preflight=memory)
                raise
            memory['runtime_capacity'] = runtime_capacity
            if runtime_capacity['status'] == 'warning':
                memory['status'] = 'warning'
                memory['warnings'] = list(dict.fromkeys([
                    *memory.get('warnings', []), *runtime_capacity.get('warnings', [])]))
            job.update(memory_preflight=memory)
            budget = inference_io.output_budget(selected, max_tokens)
            job.update(instance_id=instance_id, output_budget=budget)
            if job.cancel_event.is_set():
                if owned_id:
                    management.unload_model(c, job.device, job.model, owned_id)
                    job.update(cleanup='unloaded_before_inference')
                job.update(status='cancelled', stage='cancelled')
                return
            job.update(stage='inference')
            job.recovery_target_instance = instance_id
            job.recovery_parameters = {'max_tokens': max_tokens}
            if enable_thinking is not None:
                job.recovery_parameters['enable_thinking'] = enable_thinking
            async def stream_request():
                async with httpx.AsyncClient(base_url=str(c.base_url), headers=c.headers,
                                             timeout=180, trust_env=False, follow_redirects=False) as stream_client:
                    def send_request():
                        with job.lock:
                            tokens = job.recovery_parameters['max_tokens']
                            thinking = job.recovery_parameters.get('enable_thinking')
                        if thinking is not None:
                            return _job_stream(job, stream_client, instance_id, prompt, tokens, response_format, thinking)
                        if response_format is not None:
                            return _job_stream(job, stream_client, instance_id, prompt, tokens, response_format)
                        return _job_stream(job, stream_client, instance_id, prompt, tokens)
                    return await _retry_job_empty_answer(
                        job, send_request, lambda: _recovery_preflight(c, job.device, job.model, instance_id))
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
                loop.run_until_complete(loop.shutdown_asyncgens())
                loop.close()
            if job.cancel_event.is_set():
                job.update(status='cancelled', stage='cancelled')
                return
            if response_format is not None:
                if job.finish_reason is None:
                    raise ValueError('Structured stream omitted finish reason')
                validated = inference_io.validate_completion(
                    {'answer': job.answer, 'finish_reason': job.finish_reason,
                     'truncated': job.finish_reason == 'length' or job.output_truncated}, response_format)
                job.structured_output = validated['structured_output']
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
                   max_tokens: Annotated[int, Field(ge=32)] = 2048,
                   unload_after: bool = True, response_format: dict | None = None,
                   json_schema: dict | None = None, enable_thinking: StrictBool | None = None) -> dict:
    """Start a background model job; different explicit PCs can run concurrently.

    Start every target before polling pair_job_status. Same-device jobs serialize
    with a 30s queue timeout; at most eight active jobs per bridge process.
    Each call accepts its own prompt/model for an individual task on that PC.
    Empty final answers retry indefinitely with bounded backoff and the same loaded instance; cancel with pair_job_cancel.
    Cold-load telemetry warnings appear in pair_job_status.memory_preflight. Other errors are not retried.
    No download or model fallback. Job IDs belong to this MCP process.
    """
    fmt = inference_io.response_format(response_format, json_schema)
    if device not in management.devices() or not prompt.strip():
        raise ValueError('Use an exact configured device and nonblank prompt')
    if enable_thinking is not None:
        if type(enable_thinking) is not bool:
            raise ValueError('enable_thinking must be a boolean or null')
        if management.engine_for(device) != 'unsloth':
            raise ValueError('Explicit thinking control requires a configured Unsloth device')
        return jobs.create(device, model, _run_job, prompt, context_length, max_tokens,
                           unload_after, fmt, enable_thinking)
    if fmt is not None:
        return jobs.create(device, model, _run_job, prompt, context_length, max_tokens, unload_after, fmt)
    return jobs.create(device, model, _run_job, prompt, context_length, max_tokens, unload_after)


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False))
def pair_job_status(job_id: str) -> dict:
    """Read current progress and final answer; interrupted jobs retain prompt-free recovery metadata."""
    return jobs.get(job_id)


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False))
def pair_model_capabilities(device: str, model: str,
                            max_tokens: Annotated[int | None, Field(ge=32)] = None) -> dict:
    """Inspect fresh engine model metadata without loading or probing inference.

    Unknown does not mean unsupported. LM Studio reports vision/tool training;
    JSON Schema model support is unknown unless explicitly reported by the engine.
    """
    with management.client(device) as c:
        item = management.find_model(c, model, device)
    result = dict(inference_io.model_capabilities(item, management.engine_for(device)), device=device)
    budget = inference_io.output_budget(item)
    if max_tokens is not None:
        if not inference_io._positive_int(max_tokens) or max_tokens < 32:
            raise ValueError('max_tokens must be an integer >= 32')
        budget['requested_max_tokens'] = max_tokens
        budget['increase_request'] = inference_io.token_increase_request(max_tokens, budget['known_ceiling'])
        budget['request_within_known_ceiling'] = (max_tokens <= budget['known_ceiling']
                                                  if budget['known_ceiling'] is not None else None)
    result['output_budget'] = budget
    return result


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False))
def pair_job_list(device: str | None = None, include_terminal: bool = False,
                  limit: Annotated[int, Field(ge=1, le=64)] = 64) -> dict:
    """List process-local metadata, queue waits and active device jobs; no prompts/answers.

    Queue order is not FIFO-guaranteed. Synchronous calls and other processes are
    not visible; an empty running list does not prove a physical device is idle.
    """
    if device is not None and device not in management.devices():
        raise ValueError('Use an exact configured device')
    return jobs.listing(device, include_terminal, limit)


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    device: Annotated[str, Field(min_length=1, max_length=64)]
    model: Annotated[str, Field(min_length=1, max_length=256)]
    prompt: Annotated[str, Field(min_length=1, max_length=48000)]
    context_length: Annotated[int, Field(ge=512, le=262144)] = 8192
    max_tokens: Annotated[int, Field(ge=32)] = 2048
    unload_after: bool = True
    response_format: dict | None = None
    json_schema: dict | None = None
    enable_thinking: StrictBool | None = None


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False))
def pair_batch_start(requests: Annotated[list[BatchRequest], Field(min_length=1, max_length=8)]) -> dict:
    """Atomically admit 1-8 independent device/model prompts before polling any.

    All inputs and available process capacity are checked before workers start.
    Inventory/load errors remain independent job outcomes. Same-device jobs retain
    the 30s lock timeout; use distinct physical PCs for actual parallel inference.
    """
    if not isinstance(requests, list) or not 1 <= len(requests) <= 8:
        raise ValueError('Batch requires 1-8 requests')
    configs = management.devices()
    prepared = []
    for request in requests:
        # Also validate direct Python use, not only FastMCP's typed call path.
        try:
            item = request if isinstance(request, BatchRequest) else BatchRequest.model_validate(request)
        except ValueError as exc:
            raise ValueError('Invalid batch request') from exc
        if item.device not in configs or not item.model.strip() or not item.prompt.strip():
            raise ValueError('Batch requires configured devices and nonblank model/prompt')
        row = item.model_dump(exclude={'json_schema'})
        row['response_format'] = inference_io.response_format(item.response_format, item.json_schema)
        prepared.append(row)
    return jobs.create_batch(prepared, _run_job)


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False))
def pair_batch_status(batch_id: str) -> dict:
    """Read independent batch results in this MCP process.

    Each member retries confirmed empty final answers indefinitely on its same
    instance. This status read never resubmits a completed or failed member.
    """
    return jobs.batch_status(batch_id)


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=True, openWorldHint=False))
def pair_batch_cancel(batch_id: str) -> dict:
    """Request cancellation for live members; completed results remain unchanged.

    Bridge stream closure does not confirm engine computation has stopped.
    """
    return jobs.batch_cancel(batch_id)


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False))
@diagnostics.traced
def pair_embeddings(device: str, model: str, input: str | list[str],
                    expected_dimensions: Annotated[int | None, Field(ge=1, le=16384)] = None) -> dict:
    """Embed bounded texts using an exact already-loaded embedding model instance.

    1-32 texts, <=8192 characters each, <=48000 total. expected_dimensions only
    validates the output; it never requests vector resizing. No load or download.
    """
    texts = inference_io.embedding_inputs(input)
    if expected_dimensions is not None and (not isinstance(expected_dimensions, int) or
            isinstance(expected_dimensions, bool) or not 1 <= expected_dimensions <= 16384):
        raise ValueError('Expected dimensions must be 1-16384')
    diagnostics.stage('queue', device=device, model=model)
    with inference_lock(device, wait_seconds=30), management.client(device) as c:
        item = management.find_model(c, model, device)
        capabilities = inference_io.model_capabilities(item, management.engine_for(device))
        if capabilities['capabilities']['embeddings']['status'] != 'supported':
            raise ValueError('Engine metadata does not confirm an embedding model')
        instances = item['loaded_instances']
        if len(instances) != 1:
            raise ValueError('Embeddings require exactly one already-loaded instance; use pair_load first')
        memory = management.validate_runtime_capacity(
            telemetry.sample(management.devices()[device]), require_gpu=True)
        diagnostics.stage('inference')
        data = management.request(c, 'POST', '/v1/embeddings',
                                  {'model': management.chat_model_id(device, model, instances[0]['id']),
                                   'input': texts, 'encoding_format': 'float'})
        diagnostics.stage('validation')
        result = inference_io.embeddings(data, len(texts), expected_dimensions)
    return dict(result, device=device, requested_model=model, instance_id=instances[0]['id'],
                memory_preflight=memory)


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False))
@diagnostics.traced
def pair_vision_ask(device: str, model: str, prompt: Annotated[str, Field(min_length=1, max_length=48000)],
                    images: Annotated[list[str], Field(min_length=1, max_length=4)],
                    max_tokens: Annotated[int, Field(ge=32)] = 2048,
                    response_format: dict | None = None, json_schema: dict | None = None) -> dict:
    """Ask a confirmed vision model with inline PNG/JPEG/WebP data URLs only.

    Requires one loaded chat instance; no loading, file reads or image URL fetching.
    Header/container checks and limits: <=4 MiB decoded/image, <=12 MiB total.
    Unknown/unsupported vision capability is rejected before inference.
    """
    content = inference_io.vision_content(prompt, images)
    fmt = inference_io.response_format(response_format, json_schema)
    diagnostics.stage('queue', device=device, model=model)
    with inference_lock(device, wait_seconds=30), management.client(device) as c:
        item = management.find_model(c, model, device)
        capabilities = inference_io.model_capabilities(item, management.engine_for(device))
        if capabilities['capabilities']['chat']['status'] != 'supported' or capabilities['capabilities']['vision']['status'] != 'supported':
            raise ValueError('Fresh engine metadata does not confirm vision support for this chat model')
        inference_io.require_structured(item, fmt)
        budget = inference_io.output_budget(item, max_tokens)
        instances = item['loaded_instances']
        if len(instances) != 1:
            raise ValueError('Vision requires exactly one already-loaded instance; use pair_load first')
        memory = management.validate_runtime_capacity(
            telemetry.sample(management.devices()[device]), require_gpu=True)
        payload = {'model': management.chat_model_id(device, model, instances[0]['id']),
                   'messages': [{'role': 'user', 'content': content}], 'max_tokens': max_tokens, 'stream': False}
        if fmt is not None:
            payload['response_format'] = fmt
        started = time.monotonic()
        diagnostics.stage('inference')
        result = _request_nonempty_completion(
            lambda: management.request(c, 'POST', '/v1/chat/completions', payload),
            model, device, started, fmt,
            on_empty=lambda metadata: _start_completion_recovery(
                model, device, payload, fmt, instances[0]['id'], metadata))
        diagnostics.stage('validation')
    return dict(result, output_budget=budget, instance_id=instances[0]['id'], image_count=len(images),
                memory_preflight=memory)


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False))
def pair_job_reconfigure(job_id: str,
                         max_tokens: Annotated[int | None, Field(ge=32)] = None,
                         enable_thinking: StrictBool | None = None,
                         use_saved_profile: StrictBool = False,
                         remember: StrictBool = True,
                         device: str | None = None) -> dict:
    """Resume waiting recovery with explicitly approved settings; never load/reload a model.

    Changing loaded context requires a separate authorized operation. Use_saved_profile
    reuses only settings previously explicitly approved for this exact device/model.
    """
    if any(type(v) is not bool for v in (use_saved_profile, remember)):
        raise ValueError('Profile controls must be booleans')
    with jobs._lock:
        job = jobs._jobs.get(job_id)
    if job is None:
        raise ValueError('Recovery reconfiguration requires a live process-local job')
    with job.lock:
        revision = job.recovery_revision
        if job.status in jobs.TERMINAL or job.stage not in (
                'waiting_for_device', 'waiting_for_configuration', 'waiting_for_resources', 'retrying_empty_answer'):
            raise ValueError('Job is not waiting for recovery; inspect pair_job_status')
        previous = dict(job.recovery_parameters)
        metadata = dict(job.attempt_metadata)
        original_device = job.device
        binding = job.recovery_unpinned and original_device == 'pair-router'
        selected_device = device if binding else original_device
        if binding and (not device or device not in management.devices()):
            raise ValueError('Router recovery requires an explicit configured device with this exact loaded model')
        if not binding and device is not None and device != original_device:
            raise ValueError('Cannot move pinned recovery to another device')
    parameters = dict(previous)
    if use_saved_profile:
        parameters.update(recovery.load_profile(selected_device, job.model))
    if max_tokens is not None:
        if type(max_tokens) is not int or max_tokens < 32:
            raise ValueError('max_tokens must be an integer >= 32')
        parameters['max_tokens'] = max_tokens
    if enable_thinking is not None:
        if type(enable_thinking) is not bool:
            raise ValueError('enable_thinking must be a boolean')
        parameters['enable_thinking'] = enable_thinking
    if parameters == previous and not binding:
        raise ValueError('Approve a changed budget/thinking setting; unchanged exhausted requests are not resubmitted')
    with management.client(selected_device) as c:
        item = management.find_model(c, job.model, selected_device)
        instances = item.get('loaded_instances', [])
        if item.get('type') != 'llm' or len(instances) != 1 or (not binding and instances[0]['id'] != job.instance_id):
            raise recovery.InferenceFailure('target_changed')
        selected_instance = instances[0]['id']
        if 'enable_thinking' in parameters:
            request_thinking_options(c, selected_device, parameters['enable_thinking'])
    budget = inference_io.output_budget(item, parameters['max_tokens'])
    prompt_budget = inference_io.prompt_budget(item, parameters['max_tokens'], metadata.get('usage', {}).get('prompt_tokens'))
    if prompt_budget['request_fits'] is False:
        raise ValueError('Prompt plus requested output exceeds the loaded context; authorize a different budget or context separately')
    with job.lock:
        if job.recovery_revision != revision or job.cancel_event.is_set() or job.status in jobs.TERMINAL or job.stage not in (
                'waiting_for_device', 'waiting_for_configuration', 'waiting_for_resources', 'retrying_empty_answer'):
            raise ValueError('Job state changed during validation; inspect pair_job_status')
        if remember:
            recovery.save_profile(selected_device, job.model, parameters)
        if binding:
            job.recovery_device = selected_device
            job.recovery_unpinned = False
            job.recovery_target_instance = selected_instance
            job.device = selected_device
            job.instance_id = selected_instance
        job.update(recovery_parameters=parameters, recovery_revision=revision + 1,
                   recovery_advice=None, output_budget=dict(budget, prompt_budget=prompt_budget))
    return job.snapshot()


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=True, openWorldHint=False))
def pair_job_cancel(job_id: str) -> dict:
    """Request cancellation. Loading may finish first; an inference instance is preserved for inspection."""
    return jobs.cancel(job_id)


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False))
def pair_logs(limit: Annotated[int, Field(ge=1, le=200)] = 50) -> dict:
    """Read recent locally stored model results and sanitized error outcomes."""
    rows, status = diagnostics.recent_outcomes(limit)
    return {'status': status, 'records': rows, 'count': len(rows),
            'contains_model_output': any('answer' in row or 'structured_output' in row for row in rows),
            'notice': 'Results may contain sensitive text. Prompts, credentials, raw engine errors and embedding vectors are not stored.'}


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False))
def pair_logs_export(output_path: Annotated[str, Field(min_length=1, max_length=4096)],
                     limit: Annotated[int, Field(ge=1, le=200)] = 200) -> dict:
    """Export recent outcomes to a new local JSONL file; existing files are never overwritten."""
    return diagnostics.export_outcomes(output_path, limit)


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
                 max_tokens: Annotated[int, Field(ge=32)] = 2048) -> dict:
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
            rows.append({'device': name, **diagnostics.device_failure(exc)})
    try:
        advertised = len(catalog())
        router = {'online': True, 'advertised_models': advertised}
    except ValueError:
        router = {'online': False, 'error': 'PAIR router unavailable or catalog invalid'}
    inventory = cluster_inventory.inspect()
    discovery = {'status': inventory.get('network_discovery_status', 'unknown'),
                 'nodes': [{k: node.get(k) for k in ('device', 'source', 'online', 'network_present',
                            'peer_pin_available', 'model_inventory_status')} for node in inventory['nodes']],
                 'hint': 'Discovery does not grant trust. Pair remote devices in NVIDIA PAIR before reading their models. Empty discovery does not identify a firewall failure; check awake devices, PAIR service and LAN connectivity.'}
    recent, journal_status = diagnostics.recent_with_status()
    explanations = {'timeout': 'The device did not finish before the request deadline; inspect its load state before retrying.',
                    'device_unreachable': 'The device engine could not be reached; check its server and SSH/Tailscale path.',
                    'model_not_installed': 'The requested model is not installed on an online configured device.',
                    'empty_answer': 'The model returned no final text; a larger output budget may be needed.',
                    'device_busy': 'Another Bridge request owns the device lock; synchronous/other-process calls may not appear in pair_job_list.',
                    'load_timeout': 'Total loading deadline reached; engine may still be loading. Inspect inventory before retrying.',
                    'load_failed': 'Loading did not complete or could not be confirmed; inspect memory and engine state.',
                    'request_failed': 'The request failed; check local engine logs without sharing prompts or tokens.'}
    return {'devices': rows, 'router': router, 'discovery': discovery, 'request_journal_status': journal_status,
            'recent_requests': [dict(row, explanation=explanations.get(row.get('reason')))
                                for row in recent]}


def pair_download_plan(device: str, model: str, estimated_size_bytes: Annotated[int | None, Field(ge=1)] = None,
                       destination: str = '', quantization: str | None = None) -> dict:
    """Disabled compatibility entry point: Bridge only loads installed models."""
    raise ValueError("Model downloads are disabled; install model files in the engine, then use pair_load")


def pair_download(plan_id: str, confirm_model: str) -> dict:
    """Disabled compatibility entry point: Bridge only loads installed models."""
    raise ValueError("Model downloads are disabled; install model files in the engine, then use pair_load")


def pair_download_status(device: str, job_id: str) -> dict:
    """Disabled compatibility entry point: Bridge only loads installed models."""
    raise ValueError("Model downloads are disabled; install model files in the engine, then use pair_load")


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
