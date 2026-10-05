"""Device-scoped LM Studio management; never starts a second PAIR broker."""
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
        if row.get('engine', 'lmstudio') != 'lmstudio':
            raise ValueError('This version supports LM Studio management only')
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
        raise ValueError(f'Device returned HTTP {r.status_code}; no retry was made')
    try:
        data = r.json()
    except ValueError as exc:
        raise ValueError('Device returned invalid JSON') from exc
    if not isinstance(data, dict) or 'error' in data:
        raise ValueError('Device returned an error or invalid response')
    return data


def models(c):
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


def find_model(c, key):
    rows = [m for m in models(c) if m['key'] == key]
    if len(rows) != 1:
        raise ValueError('Model is not installed on this device. Refresh pair_list(device=...); no download was made')
    return rows[0]


def ensure_capacity(rows, candidate, max_loaded_bytes):
    """Conservative weight-size policy; KV cache and runtime overhead are extra."""
    if max_loaded_bytes is None or candidate['loaded_instances']:
        return
    used = sum(m.get('size_bytes', 0) for m in rows if m['loaded_instances'])
    incoming = candidate.get('size_bytes')
    if not isinstance(incoming, int) or used + incoming > max_loaded_bytes:
        raise ValueError('Device model-weight limit would be exceeded; no model was unloaded or loaded')


def estimate_memory(device_id, model, context_length):
    """Ask the target LM Studio CLI for a read-only memory estimate."""
    cli = shutil.which('lms') or str(Path.home() / '.lmstudio' / 'bin' / 'lms')
    if not Path(cli).is_file():
        raise ValueError('LM Studio CLI is unavailable; memory estimate is unknown')
    if not isinstance(context_length, int) or context_length < 1:
        raise ValueError('A positive planned context length is required')
    device = devices()[device_id]
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
        'estimated memory limit would be exceeded',
        'model-weight limit would be exceeded',
        'Loaded instance context is unknown',
    ))


def describe_unloaded_instances(instances):
    return ', '.join(f"{row['model']} [{row['instance_id']}]" for row in instances)


def preflight_for_load(c, device_id, model_key, context_length, device_config,
                       max_loaded_bytes=None, capacity=None, capacity_sampler=None):
    """Preflight a cold load and, only for exact allowlisted keys, unload instances until it fits."""
    rows = models(c)
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

    unloaded = []
    for allowed_key in allowed:
        if allowed_key == model_key:
            continue
        while True:
            rows = models(c)
            candidate = next((row for row in rows if row['key'] == model_key), None)
            if candidate is None:
                raise ValueError('Candidate model disappeared while making room; no load was started')
            target = next((row for row in rows if row['key'] == allowed_key and row['loaded_instances']), None)
            if target is None:
                break
            instance_id = target['loaded_instances'][0]['id']
            try:
                request(c, 'POST', '/api/v1/models/unload', {'instance_id': instance_id})
            except ValueError as exc:
                details = describe_unloaded_instances(unloaded)
                if details:
                    raise ValueError(f'{exc}; already released configured instances: {details}') from exc
                raise
            rows = models(c)
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
