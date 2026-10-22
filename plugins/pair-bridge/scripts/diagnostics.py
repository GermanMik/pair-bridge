"""Local, redacted request journal for PAIR model calls."""
from __future__ import annotations

import contextvars
import functools
import json
import os
import time
import uuid
import recovery
from pathlib import Path
from platformdirs import user_cache_path

_current = contextvars.ContextVar('pair_trace', default=None)


def device_failure(exc):
    """Whitelist diagnostic outcomes; never echo engine error bodies or origins."""
    message = str(exc).lower()
    if ('http 401' in message or 'http 403' in message or
            'api token environment variable is missing' in message or
            'desktop credential' in message or 'desktop login' in message):
        return {'online': None, 'auth_status': 'required', 'reason': 'auth_required',
                'error': 'Engine authorization is required or misconfigured.',
                'hint': 'Check the configured API-key environment variable or local desktop credential. For remote keyless Unsloth, verify Chat and inference access; Full access is localhost-only.'}
    if 'unreachable' in message or 'timed out' in message or 'ssh tunnel' in message:
        return {'online': False, 'reason': 'device_unreachable',
                'error': 'Configured engine could not be reached.',
                'hint': 'Check that the device is awake and its API server is listening at the configured origin. No port scan or inference retry was made.'}
    return {'online': None, 'reason': 'inspection_failed',
            'error': 'Engine inspection failed or its response was invalid.',
            'hint': 'Inspect local engine settings/logs without sharing credentials or prompts.'}


def journal_path() -> Path:
    return user_cache_path('codex-pair-bridge', appauthor=False) / 'requests.jsonl'


def outcome_path() -> Path:
    return journal_path().with_name('outcomes.jsonl')


def reason(exc: Exception) -> str:
    if isinstance(exc, recovery.InferenceFailure):
        return exc.code
    message = str(exc).lower()
    if 'another codex pair request is running' in message:
        return 'device_busy'
    if 'load timed out at its total deadline' in message:
        return 'load_timeout'
    if 'timed out' in message or 'timeout' in message:
        return 'timeout'
    if 'unreachable' in message or 'cannot reach' in message:
        return 'device_unreachable'
    if 'not installed' in message or 'no suitable installed' in message:
        return 'model_not_installed'
    if 'no final text' in message:
        return 'empty_answer'
    if 'load' in message:
        return 'load_failed'
    return 'request_failed'


_SAFE_ERRORS = {
    'refusal': 'The model declined the request.',
    'tool_response': 'The model returned a tool call instead of final text.',
    'invalid_response': 'The engine returned an invalid completion.',
    'target_changed': 'The original model instance changed; inspect device state.',
    'device_busy': 'Another Bridge request is using the device.',
    'load_timeout': 'The model load deadline was reached; inspect device state before retrying.',
    'timeout': 'The request timed out; inspect engine state before retrying.',
    'device_unreachable': 'The configured device could not be reached.',
    'model_not_installed': 'No suitable installed model was available.',
    'empty_answer': 'The model returned no final text.',
    'load_failed': 'Model loading failed or could not be confirmed.',
    'request_failed': 'The model request failed; inspect local engine logs.',
}


def safe_error(code: str) -> str:
    return _SAFE_ERRORS.get(code, _SAFE_ERRORS['request_failed'])


def _append(row: dict) -> None:
    path = journal_path()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    data = (json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n').encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(fd, data)
    finally:
        os.close(fd)


def record_outcome(row: dict) -> None:
    """Persist only model result fields, retry counts, or a safe error classification."""
    safe = {key: row[key] for key in ('timestamp', 'request_id', 'operation', 'device', 'model',
                                       'status', 'answer', 'structured_output', 'finish_reason',
                                       'empty_answer_retries', 'error_code', 'error') if key in row}
    if isinstance(row.get('attempt_metadata'), dict):
        safe['attempt_metadata'] = recovery.safe_metadata(row['attempt_metadata'])
    path = outcome_path()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    data = (json.dumps(safe, ensure_ascii=False, separators=(',', ':')) + '\n').encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(fd, data)
    finally:
        os.close(fd)


def _result_fields(trace: dict, result: object) -> dict | None:
    if trace.get('operation') not in ('pair_ask', 'pair_smart_ask', 'pair_vision_ask') or not isinstance(result, dict):
        return None
    row = {'timestamp': int(time.time()), 'request_id': trace['request_id'],
           'operation': trace['operation'], 'device': result.get('device') or trace.get('device'),
           'model': result.get('selected_model') or result.get('requested_model') or trace.get('model'),
           'status': 'completed', 'empty_answer_retries': trace.get('empty_answer_retries', 0)}
    if isinstance(result.get('answer'), str):
        row['answer'] = result['answer'][:48000]
    structured = result.get('structured_output')
    if isinstance(structured, (dict, list, str, int, float, bool)) or structured is None:
        if 'structured_output' in result:
            row['structured_output'] = structured
    if isinstance(result.get('finish_reason'), str):
        row['finish_reason'] = result['finish_reason'][:128]
    if isinstance(result.get('attempt_metadata'), dict):
        row['attempt_metadata'] = recovery.safe_metadata(result['attempt_metadata'])
    return row if 'answer' in row or 'structured_output' in row else None


def record_attempt(request_id, device, model, metadata, retries, status='retrying'):
    try:
        record_outcome({'timestamp': int(time.time()), 'request_id': request_id,
                        'operation': 'inference_attempt', 'device': device, 'model': model,
                        'status': status, 'empty_answer_retries': retries,
                        'attempt_metadata': recovery.safe_metadata(metadata)})
    except (OSError, TypeError, ValueError):
        pass


def _error_outcome(trace: dict) -> dict | None:
    if trace.get('operation') not in ('pair_ask', 'pair_smart_ask', 'pair_vision_ask'):
        return None
    code = trace.get('reason', 'request_failed')
    return {'timestamp': int(time.time()), 'request_id': trace['request_id'],
            'operation': trace['operation'], 'device': trace.get('device'),
            'model': trace.get('model'), 'status': 'error',
            'empty_answer_retries': trace.get('empty_answer_retries', 0),
            'error_code': code, 'error': safe_error(code),
            'attempt_metadata': recovery.safe_metadata(trace.get('attempt_metadata', {}))}


def recent_outcomes(limit: int = 50) -> tuple[list[dict], str]:
    if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 200:
        raise ValueError('Log limit must be between 1 and 200')
    try:
        lines = outcome_path().read_text(encoding='utf-8', errors='replace').splitlines()
    except FileNotFoundError:
        return [], 'not_created'
    except PermissionError:
        return [], 'permission_denied'
    except OSError:
        return [], 'unavailable'
    rows = []
    for line in lines[-limit * 2:]:
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict) and row.get('status') in ('completed', 'error', 'cancelled', 'retrying', 'attempt_completed'):
            rows.append(row)
    return rows[-limit:], 'ok'


def last_successful_model(device: str | None = None) -> dict | None:
    try:
        with outcome_path().open('rb') as stream:
            position = stream.seek(0, os.SEEK_END)
            pending = b''
            while position:
                size = min(8192, position)
                position -= size
                stream.seek(position)
                chunks = stream.read(size) + pending
                lines = chunks.split(b'\n')
                pending = lines.pop(0) if position else b''
                for raw in reversed(lines):
                    try:
                        row = json.loads(raw)
                    except (ValueError, UnicodeDecodeError):
                        continue
                    if (isinstance(row, dict) and row.get('status') == 'completed' and row.get('model')
                            and ('answer' in row or 'structured_output' in row)
                            and (device is None or row.get('device') == device)):
                        return {'device': row.get('device'), 'model': row.get('model')}
    except OSError:
        return None
    return None


def export_outcomes(output_path: str, limit: int = 200) -> dict:
    if not isinstance(output_path, str) or not output_path or len(output_path) > 4096:
        raise ValueError('Provide an absolute .jsonl export path')
    target = Path(output_path)
    if not target.is_absolute() or target.suffix.lower() != '.jsonl':
        raise ValueError('Export path must be absolute and end in .jsonl')
    if not target.parent.is_dir():
        raise ValueError('Export directory must already exist')
    rows, status = recent_outcomes(limit)
    if status not in ('ok', 'not_created'):
        raise OSError(f'Local outcome log is {status}')
    payload = ''.join(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n' for row in rows).encode()
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(payload)
    except Exception:
        try:
            target.unlink(missing_ok=True)
        finally:
            raise
    return {'status': 'exported', 'path': str(target), 'records': len(rows),
            'format': 'jsonl', 'overwritten': False}


def traced(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        started = time.monotonic()
        trace = {'request_id': uuid.uuid4().hex, 'started_at': int(time.time()),
                 'operation': func.__name__, 'device': None, 'model': None,
                 'stage': 'start', 'stage_started': started, 'durations_ms': {},
                 'empty_answer_retries': 0}
        token = _current.set(trace)
        try:
            result = func(*args, **kwargs)
            trace['status'] = 'ok'
            outcome = _result_fields(trace, result)
            if outcome is not None:
                try:
                    record_outcome(outcome)
                except (OSError, TypeError, ValueError):
                    pass
            return result
        except Exception as exc:
            trace['status'] = 'error'
            trace['reason'] = reason(exc)
            if isinstance(exc, recovery.InferenceFailure):
                trace['attempt_metadata'] = exc.metadata
            outcome = _error_outcome(trace)
            if outcome is not None:
                try:
                    record_outcome(outcome)
                except (OSError, TypeError, ValueError):
                    pass
            raise
        finally:
            now = time.monotonic()
            trace['durations_ms'][trace['stage']] = round((now - trace['stage_started']) * 1000)
            row = {k: trace[k] for k in ('request_id', 'started_at', 'operation', 'device', 'model',
                                          'stage', 'durations_ms', 'status', 'empty_answer_retries')}
            if 'reason' in trace:
                row['reason'] = trace['reason']
            row['total_ms'] = round((now - started) * 1000)
            try:
                _append(row)
            except OSError:
                pass  # Diagnostics must never turn a successful answer into a failure.
            _current.reset(token)
    return wrapper


def note_empty_answer_retry(count: int) -> None:
    """Expose retry progress without recording prompts, answers, or engine bodies."""
    trace = _current.get()
    if trace is None:
        return
    trace['empty_answer_retries'] = count
    now = time.monotonic()
    trace['durations_ms'][trace['stage']] = round((now - trace['stage_started']) * 1000)
    trace['stage'], trace['stage_started'] = 'retrying_empty_answer', now
    try:
        _append({'request_id': trace['request_id'], 'started_at': trace['started_at'],
                 'operation': trace['operation'], 'device': trace['device'], 'model': trace['model'],
                 'stage': trace['stage'], 'durations_ms': dict(trace['durations_ms']),
                 'status': 'running', 'reason': 'empty_answer', 'empty_answer_retries': count})
    except OSError:
        pass


def stage(name: str, *, device: str | None = None, model: str | None = None) -> None:
    trace = _current.get()
    if trace is None:
        return
    now = time.monotonic()
    trace['durations_ms'][trace['stage']] = round((now - trace['stage_started']) * 1000)
    trace['stage'], trace['stage_started'] = name, now
    if device is not None:
        trace['device'] = device[:128]
    if model is not None:
        trace['model'] = model[:256]
    try:
        _append({'request_id': trace['request_id'], 'started_at': trace['started_at'],
                 'operation': trace['operation'], 'device': trace['device'], 'model': trace['model'],
                 'stage': name, 'durations_ms': dict(trace['durations_ms']), 'status': 'running',
                 'empty_answer_retries': trace.get('empty_answer_retries', 0)})
    except OSError:
        pass


def recent_with_status(limit: int = 20) -> tuple[list[dict], str]:
    """Read the optional local journal without letting ACL errors break tools."""
    path = journal_path()
    try:
        content = path.read_text(errors='replace')
    except FileNotFoundError:
        return [], 'not_created'
    except PermissionError:
        return [], 'permission_denied'
    except OSError:
        return [], 'unavailable'

    rows = []
    for line in content.splitlines()[-max(limit * 10, limit):]:
        try:
            row = json.loads(line)
        except ValueError:
            continue  # A partial last write must not hide earlier diagnostics.
        if isinstance(row, dict) and row.get('operation') in ('pair_ask', 'pair_smart_ask', 'pair_load', 'pair_unload',
                                                           'pair_embeddings', 'pair_vision_ask'):
            rows.append(row)
    latest = {}
    for row in rows:
        latest[row.get('request_id')] = row
    result = list(latest.values())[-limit:]
    for row in result:
        if row.get('status') == 'running' and time.time() - row.get('started_at', 0) > 600:
            row['status'] = 'interrupted_or_stale'
    return result, 'ok'


def recent(limit: int = 20) -> list[dict]:
    return recent_with_status(limit)[0]
