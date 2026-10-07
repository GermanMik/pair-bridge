"""Device-scoped local model engine management; never starts a second PAIR broker."""
import contextlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import time
from urllib.parse import urlsplit
import httpx


def devices():
    path = Path.home() / '.pair-bridge.json'
    if not path.exists():
        path = Path.home() / '.codex-pair-bridge.json'
    config = json.loads(path.read_text()) if path.exists() else {}
    rows = config.get('devices', [])
    if not isinstance(rows, list):
        raise ValueError('devices must be a list')
    result = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError('Each device must be an object')
        name = row.get('id', '')
        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', name) or name in result:
            raise ValueError('Device IDs must be unique short names')
        url = row.get('base_url', '')
        p = urlsplit(url)
        if p.scheme not in ('http', 'https') or not p.hostname or p.username or p.password or p.query or p.fragment or p.path not in ('', '/'):
            raise ValueError('Device base_url must be an HTTP(S) origin without credentials')
        if row.get('engine', 'lmstudio') not in ('lmstudio', 'unsloth'):
            raise ValueError('Device engine must be lmstudio or unsloth')
        host = row.get('ssh_host')
        if host is not None and (not isinstance(host, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', host)):
            raise ValueError('ssh_host must be an existing SSH config alias')
        if host and (p.scheme != 'http' or p.hostname not in ('localhost', '127.0.0.1')):
            raise ValueError('SSH devices must target the remote HTTP loopback origin')
        key = row.get('api_key_env')
        if key is not None and (not isinstance(key, str) or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', key)):
            raise ValueError('api_key_env must name an environment variable')
        cap = row.get('max_loaded_bytes')
        if cap is not None and (not isinstance(cap, int) or isinstance(cap, bool) or cap <= 0):
            raise ValueError('max_loaded_bytes must be a positive integer')
        unload_models = row.get('auto_unload_models', [])
        if (not isinstance(unload_models, list) or
                any(not isinstance(model, str) or not model or model != model.strip() or len(model) > 512 or
                    any(ch in model for ch in '\r\n\x00') for model in unload_models) or
                len(set(unload_models)) != len(unload_models)):
            raise ValueError('auto_unload_models must be a list of unique exact model keys')
        models_path = row.get('models_path')
        if models_path is not None and (not isinstance(models_path, str) or not models_path or len(models_path) > 512 or
                                        any(ch in models_path for ch in '\r\n\x00') or
                                        not (Path(models_path).is_absolute() or re.match(r'^[A-Za-z]:[\\/]', models_path))):
            raise ValueError('models_path must be a short absolute path configured by the user')
        result[name] = dict(row, base_url=url.rstrip('/'))
    return result


def engine_for(device_id):
    device = devices().get(device_id)
    if device is None:
        raise ValueError('Unknown device. Use pair_devices and an exact configured ID')
    return device.get('engine', 'lmstudio')


@contextlib.contextmanager
def endpoint(device):
    if not device.get('ssh_host'):
        yield device['base_url']
        return
    p = urlsplit(device['base_url'])
    try:
        with socket.socket() as s:
            s.bind(('127.0.0.1', 0))
            port = s.getsockname()[1]
    except OSError as exc:
        raise ValueError('Cannot create a local SSH tunnel for this device') from exc
    # SSH configuration supplies authentication. No remote shell is invoked.
    args = ['ssh', '-N', '-T', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
            '-o', 'ExitOnForwardFailure=yes', '-o', 'ConnectTimeout=10',
            '-L', f'127.0.0.1:{port}:127.0.0.1:{p.port or 80}', device['ssh_host']]
    try:
        process = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError as exc:
        raise ValueError('SSH is unavailable; install OpenSSH and configure the device alias') from exc
    try:
        deadline = time.monotonic() + 12
        while True:
            if process.poll() is not None:
                raise ValueError('SSH tunnel failed. Check the existing host alias, host key, and authentication')
            try:
                with socket.create_connection(('127.0.0.1', port), timeout=.2):
                    break
            except OSError:
                if time.monotonic() >= deadline:
                    raise ValueError('SSH tunnel startup timed out')
                time.sleep(.05)
        yield f'http://127.0.0.1:{port}'
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


@contextlib.contextmanager
def client(device_id):
    all_devices = devices()
    if device_id not in all_devices:
        raise ValueError('Unknown device. Use pair_devices and an exact configured ID')
    device = all_devices[device_id]
    key_env = device.get('api_key_env')
    key = os.environ.get(key_env) if key_env else None
    if key_env and not key:
        raise ValueError('Configured device API token environment variable is missing')
    with endpoint(device) as url:
        with httpx.Client(base_url=url, timeout=180, trust_env=False, follow_redirects=False,
                          headers={'Authorization': 'Bearer ' + key} if key else {}) as c:
            yield c


def request(c, method, route, body=None):
    try:
        r = c.request(method, route, json=body)
    except httpx.TimeoutException as exc:
        raise ValueError('Device timed out; operation may still be running. Inspect status before retrying') from exc
    except httpx.RequestError as exc:
        raise ValueError('Device engine is unreachable') from exc
    if not r.is_success:
        if (method == 'POST' and route in ('/api/v1/models/load', '/api/inference/load') and
                _response_reports_capacity_error(r)):
            raise ValueError('Device reports insufficient available memory for this model; load was not confirmed')
        raise ValueError(f'Device returned HTTP {r.status_code}; no retry was made')
    try:
        data = r.json()
    except ValueError as exc:
        raise ValueError('Device returned invalid JSON') from exc
    if isinstance(data, dict) and data.get('_deferred_error'):
        if method == 'POST' and route == '/api/inference/load' and _response_reports_capacity_error(r):
            raise ValueError('Device reports insufficient available memory for this model; load was not confirmed')
        raise ValueError('Device returned a deferred error; operation was not confirmed')
    if not isinstance(data, dict) or 'error' in data:
        raise ValueError('Device returned an error or invalid response')
    return data


def _response_reports_capacity_error(response):
    try:
        detail = response.text[:8192].lower()
    except Exception:
        return False
    native_shortage = re.search(
        r'this model needs about \d+(?:\.\d+)? gb of gpu memory at a \d+ context, '
        r'and \d+(?:\.\d+)? gb is free next to the models already loaded\.', detail)
    return bool(native_shortage) or any(marker in detail for marker in (
        'out of memory', 'not enough memory', 'insufficient memory',
        'failed to allocate', 'cannot allocate memory', 'cuda error: out of memory',
        'cublas_status_alloc_failed', 'hip out of memory',
    ))


def _unsloth_model_type(row):
    model_id = row['id'].lower()
    task = row.get('task')
    task = re.sub(r'[_\s]+', '-', task.lower()) if isinstance(task, str) else ''
    if any(term in model_id or term in task for term in ('embed', 'rerank')):
        return 'embedding'
    if any(term in model_id or term in task for term in ('dflash', 'draft')):
        return 'draft'
    if task and task not in ('text-generation', 'text-generation-inference', 'conversational', 'chat', 'llm'):
        return 'other'
    return 'llm'


def models(c, device_id=None):
    if device_id is not None and engine_for(device_id) == 'unsloth':
        rows = request(c, 'GET', '/v1/models').get('data')
        if not isinstance(rows, list):
            raise ValueError('Unsloth Studio OpenAI models API is required')
        out = []
        for row in rows:
            if (not isinstance(row, dict) or not isinstance(row.get('id'), str) or
                    not row['id'] or not isinstance(row.get('loaded'), bool)):
                raise ValueError('Invalid Unsloth model inventory; cannot determine loaded state')
            item = {'key': row['id'], 'type': _unsloth_model_type(row),
                    'loaded_instances': ([{'id': row['id'], 'config': {
                        'context_length': row['context_length']
                    } if isinstance(row.get('context_length'), int) and not isinstance(row.get('context_length'), bool) else {}}]
                                         if row['loaded'] else [])}
            if isinstance(row.get('display_name'), str):
                item['display_name'] = row['display_name']
            if isinstance(row.get('max_context_length'), int) and not isinstance(row.get('max_context_length'), bool):
                item['max_context_length'] = row['max_context_length']
            if isinstance(row.get('size_bytes'), int) and not isinstance(row.get('size_bytes'), bool):
                item['size_bytes'] = row['size_bytes']
            out.append(item)
        return out
    rows = request(c, 'GET', '/api/v1/models').get('models')
    if not isinstance(rows, list):
        raise ValueError('LM Studio native v1 API is required')
    out = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get('key'), str) or not isinstance(row.get('loaded_instances'), list):
            raise ValueError('Invalid model inventory; cannot determine loaded state')
        instances = row['loaded_instances']
        if any(not isinstance(x, dict) or not isinstance(x.get('id'), str) for x in instances):
            raise ValueError('Invalid loaded instance inventory')
        out.append({k: row[k] for k in ('key', 'display_name', 'type', 'size_bytes', 'max_context_length', 'loaded_instances') if k in row})
    return out


def find_model(c, key, device_id=None):
    rows = [m for m in models(c, device_id) if m['key'] == key]
    if len(rows) != 1:
        raise ValueError('Model is not installed on this device. Refresh pair_list(device=...); no download was made')
    return rows[0]


def load_model(c, device_id, model_key, context_length, model_type='llm'):
    if engine_for(device_id) == 'unsloth':
        return request(c, 'POST', '/api/inference/load', {
            'model_path': model_key,
            'n_ctx': context_length,
            'max_seq_length': context_length,
            'load_in_4bit': True,
        })
    body = {'model': model_key}
    if model_type == 'llm':
        body['context_length'] = context_length
    return request(c, 'POST', '/api/v1/models/load', body)


def load_with_auto_unload(c, device_id, model_key, context_length, model_type, device_config):
    """Preserve confirmed releases on every failed load retry path."""
    unloaded = []
    try:
        return _load_with_auto_unload(c, device_id, model_key, context_length, model_type, device_config, unloaded)
    except ValueError as exc:
        exc.auto_unloaded_instances = list(unloaded)
        raise


def _load_with_auto_unload(c, device_id, model_key, context_length, model_type, device_config, unloaded):
    """Retry a load only after a confirmed capacity failure and exact configured unloads."""
    try:
        return {'result': load_model(c, device_id, model_key, context_length, model_type),
                'auto_unloaded_instances': []}
    except ValueError as exc:
        if not is_capacity_error(exc):
            raise
        last_error = exc

    allowed = device_config.get('auto_unload_models', [])
    if not allowed:
        raise last_error
    for allowed_key in allowed:
        if allowed_key == model_key:
            continue
        rows = models(c, device_id)
        candidate = next((row for row in rows if row['key'] == model_key), None)
        if candidate is None:
            raise ValueError('Candidate model disappeared after capacity failure; no retry was made')
        if candidate['loaded_instances']:
            raise ValueError(f'{last_error}; candidate now appears loaded, inspect its state before retrying')
        while True:
            candidate = next((row for row in rows if row['key'] == model_key), None)
            if candidate is None:
                raise ValueError('Candidate model disappeared after capacity failure; no retry was made')
            if candidate['loaded_instances']:
                raise ValueError(f'{last_error}; candidate now appears loaded, inspect its state before retrying')
            target = next((row for row in rows if row['key'] == allowed_key and row['loaded_instances']), None)
            if target is None:
                break
            instance_id = target['loaded_instances'][0]['id']
            unload_model(c, device_id, allowed_key, instance_id)
            rows = models(c, device_id)
            if any(instance['id'] == instance_id for row in rows for instance in row['loaded_instances']):
                raise ValueError('Configured auto-unload did not remove the exact instance; load was not retried')
            unloaded.append({'model': allowed_key, 'instance_id': instance_id})
            candidate = next((row for row in rows if row['key'] == model_key), None)
            if candidate is None:
                raise ValueError('Candidate model disappeared after capacity failure; no retry was made')
            if candidate['loaded_instances']:
                raise ValueError(f'{last_error}; candidate now appears loaded, inspect its state before retrying')
            try:
                result = load_model(c, device_id, model_key, context_length, model_type)
                return {'result': result, 'auto_unloaded_instances': unloaded}
            except ValueError as exc:
                if not is_capacity_error(exc):
                    details = describe_unloaded_instances(unloaded)
                    if details:
                        raise ValueError(f'{exc}; configured auto_unload_models released: {details}') from exc
                    raise
                last_error = exc
                rows = models(c, device_id)

    if unloaded:
        raise ValueError(f'{last_error}; configured auto_unload_models did not free enough memory '
                         f'after releasing: {describe_unloaded_instances(unloaded)}')
    raise last_error


def unload_model(c, device_id, model_key, instance_id):
    if engine_for(device_id) == 'unsloth':
        return request(c, 'POST', '/api/inference/unload', {'model_path': model_key})
    return request(c, 'POST', '/api/v1/models/unload', {'instance_id': instance_id})


def chat_model_id(device_id, model_key, instance_id):
    """Return the engine's accepted model identifier for OpenAI chat calls."""
    return model_key if engine_for(device_id) == 'unsloth' else instance_id


def ensure_capacity(rows, candidate, max_loaded_bytes):
    """Conservative weight-size policy; KV cache and runtime overhead are extra."""
    if max_loaded_bytes is None or candidate['loaded_instances']:
        return
    used = sum(m.get('size_bytes', 0) for m in rows if m['loaded_instances'])
    incoming = candidate.get('size_bytes')
    if not isinstance(incoming, int) or used + incoming > max_loaded_bytes:
        raise ValueError('Device model-weight limit would be exceeded; no model was unloaded or loaded')


def estimate_memory(device_id, model, context_length):
    """Ask the configured engine for a read-only memory estimate."""
    if not isinstance(context_length, int) or context_length < 1:
        raise ValueError('A positive planned context length is required')
    device = devices()[device_id]
    if device.get('engine', 'lmstudio') == 'unsloth':
        with client(device_id) as c:
            data = request(c, 'POST', '/api/inference/estimate-memory', {
                'model_path': model, 'n_ctx': context_length,
                'max_seq_length': context_length,
            })
        if data.get('available') is not True:
            reason = data.get('reason')
            safe_reason = reason if isinstance(reason, str) and re.fullmatch(r'[a-z_]{1,64}', reason) else 'unavailable'
            raise ValueError(f'Unsloth Studio cannot estimate this model at the planned context ({safe_reason})')
        total_bytes, gpu_bytes = data.get('total_bytes'), data.get('gpu_bytes')
        if (not isinstance(total_bytes, int) or isinstance(total_bytes, bool) or total_bytes <= 0 or
                not isinstance(gpu_bytes, int) or isinstance(gpu_bytes, bool) or gpu_bytes < 0 or
                data.get('kv_estimable') is False or data.get('drafter_kv_unsized') is True or
                data.get('adapters_unsized') is True):
            raise ValueError('Unsloth Studio returned an incomplete memory estimate')
        return {'total_bytes': total_bytes, 'gpu_bytes': gpu_bytes,
                'context_length': context_length, 'source': 'Unsloth Studio /api/inference/estimate-memory'}
    cli = shutil.which('lms') or str(Path.home() / '.lmstudio' / 'bin' / 'lms')
    if not Path(cli).is_file():
        raise ValueError('LM Studio CLI is unavailable; memory estimate is unknown')
    with endpoint(device) as origin:
        p = urlsplit(origin)
        args = [cli, 'load', '--estimate-only', '--context-length', str(context_length),
                '--host', p.hostname, '--port', str(p.port or (443 if p.scheme == 'https' else 80)), model]
        try:
            result = subprocess.run(args, stdin=subprocess.DEVNULL, capture_output=True,
                                    text=True, timeout=30, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ValueError('LM Studio memory estimate is unavailable') from exc
    if result.returncode:
        raise ValueError('LM Studio could not estimate this model at the planned context')
    values = {}
    for label, key in (('Estimated GPU Memory', 'gpu_bytes'), ('Estimated Total Memory', 'total_bytes')):
        match = re.search(r'^' + label + r':\s*([\d.,\s\u00a0\u202f]+)\s*(GiB|MiB|GB|MB)',
                          result.stdout + '\n' + result.stderr, re.MULTILINE)
        if not match:
            raise ValueError('LM Studio returned an unrecognized memory estimate')
        number = float(match.group(1).strip().replace(',', '.').replace(' ', '').replace('\u00a0', '').replace('\u202f', ''))
        scale = {'GiB': 2**30, 'MiB': 2**20, 'GB': 10**9, 'MB': 10**6}[match.group(2)]
        values[key] = round(number * scale)
    return dict(values, context_length=context_length, source='lms load --estimate-only')


def memory_preflight(device_id, rows, candidate, context_length, max_loaded_bytes, capacity=None):
    """Estimate each running instance and the candidate at its actual/planned context."""
    if candidate['loaded_instances']:
        return {'status': 'already_loaded'}
    try:
        incoming = estimate_memory(device_id, candidate['key'], context_length)
        current = []
        for row in rows:
            for instance in row['loaded_instances']:
                config = instance.get('config') or {}
                actual = config.get('context_length')
                if not isinstance(actual, int) or actual < 1:
                    raise ValueError('Loaded instance context is unknown')
                current.append(dict(model=row['key'], instance_id=instance['id'],
                                    **estimate_memory(device_id, row['key'], actual)))
        total = incoming['total_bytes'] + sum(item['total_bytes'] for item in current)
        required = round(total * 1.1)
        if max_loaded_bytes is not None and required > max_loaded_bytes:
            raise ValueError('Device estimated memory limit would be exceeded; no model was loaded')
        if isinstance(capacity, dict) and time.time() - capacity.get('checked_at', 0) <= 15:
            available = capacity.get('memory', {}).get('available_bytes')
            gpu_rows = capacity.get('gpu', {}).get('devices') or []
            gpu_free = sum(g.get('free_bytes', 0) for g in gpu_rows)
            # Compare incremental need with currently available capacity: existing loads already occupy memory.
            system_needed = (max(incoming['total_bytes'] - incoming['gpu_bytes'], 2**30) if gpu_rows
                             else incoming['total_bytes'])
            if isinstance(available, int) and round(system_needed * 1.1) > available:
                raise ValueError('Insufficient currently available system memory for this model and context')
            if gpu_rows and round(incoming['gpu_bytes'] * 1.1) > gpu_free:
                raise ValueError('Insufficient currently available GPU memory for this model and context')
        return {'status': 'estimated', 'candidate': incoming, 'loaded': current,
                'estimated_total_bytes': total, 'required_with_headroom_bytes': required,
                'max_loaded_bytes': max_loaded_bytes, 'capacity': capacity,
                'note': 'CLI load estimate plus a separate live capacity sample when available; leave headroom for the OS and other applications.'}
    except ValueError as exc:
        if max_loaded_bytes is not None or 'limit would be exceeded' in str(exc) or 'Insufficient currently available' in str(exc):
            raise
        return {'status': 'unknown', 'reason': str(exc),
                'note': 'No configured memory cap; loading may still fail or evict another instance.'}


def is_capacity_error(error):
    text = str(error)
    return any(marker in text for marker in (
        'Insufficient currently available',
        'Device reports insufficient available memory',
        'estimated memory limit would be exceeded',
        'model-weight limit would be exceeded',
        'Loaded instance context is unknown',
    ))


def describe_unloaded_instances(instances):
    return ', '.join(f"{row['model']} [{row['instance_id']}]" for row in instances)


def preflight_for_load(c, device_id, model_key, context_length, device_config,
                       max_loaded_bytes=None, capacity=None, capacity_sampler=None):
    """Preserve confirmed unload records even if a later preflight step fails."""
    unloaded = []
    try:
        return _preflight_for_load(c, device_id, model_key, context_length, device_config,
                                   max_loaded_bytes, capacity, capacity_sampler, unloaded)
    except ValueError as exc:
        exc.auto_unloaded_instances = list(unloaded)
        raise


def _preflight_for_load(c, device_id, model_key, context_length, device_config,
                        max_loaded_bytes, capacity, capacity_sampler, unloaded):
    """Preflight a cold load and, only for exact allowlisted keys, unload instances until it fits."""
    rows = models(c, device_id)
    candidate = next((row for row in rows if row['key'] == model_key), None)
    if candidate is None:
        raise ValueError('Model is not installed on this device; refresh the model inventory')
    try:
        memory = memory_preflight(device_id, rows, candidate, context_length, max_loaded_bytes, capacity)
        return {'rows': rows, 'candidate': candidate, 'memory': memory, 'auto_unloaded_instances': []}
    except ValueError as exc:
        if not is_capacity_error(exc):
            raise
        last_error = exc

    allowed = device_config.get('auto_unload_models', [])
    if not allowed:
        raise last_error

    for allowed_key in allowed:
        if allowed_key == model_key:
            continue
        while True:
            rows = models(c, device_id)
            candidate = next((row for row in rows if row['key'] == model_key), None)
            if candidate is None:
                raise ValueError('Candidate model disappeared while making room; no load was started')
            target = next((row for row in rows if row['key'] == allowed_key and row['loaded_instances']), None)
            if target is None:
                break
            instance_id = target['loaded_instances'][0]['id']
            try:
                unload_model(c, device_id, allowed_key, instance_id)
            except ValueError as exc:
                details = describe_unloaded_instances(unloaded)
                if details:
                    raise ValueError(f'{exc}; already released configured instances: {details}') from exc
                raise
            rows = models(c, device_id)
            if any(instance['id'] == instance_id for row in rows for instance in row['loaded_instances']):
                raise ValueError('Configured auto-unload did not remove the exact instance; stopped before loading')
            unloaded.append({'model': allowed_key, 'instance_id': instance_id})
            candidate = next((row for row in rows if row['key'] == model_key), None)
            if candidate is None:
                raise ValueError('Candidate model disappeared while making room; no load was started')
            current_capacity = capacity_sampler() if capacity_sampler else None
            try:
                memory = memory_preflight(device_id, rows, candidate, context_length,
                                          max_loaded_bytes, current_capacity)
                return {'rows': rows, 'candidate': candidate, 'memory': memory,
                        'auto_unloaded_instances': unloaded}
            except ValueError as exc:
                if not is_capacity_error(exc):
                    raise
                last_error = exc

    if unloaded:
        raise ValueError(f'{last_error}; configured auto_unload_models did not free enough memory '
                         f'after releasing: {describe_unloaded_instances(unloaded)}')
    raise last_error
