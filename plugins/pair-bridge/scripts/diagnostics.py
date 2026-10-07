"""Local, redacted request journal for PAIR model calls."""
from __future__ import annotations

import contextvars
import functools
import json
import os
import time
import uuid
from pathlib import Path
from platformdirs import user_cache_path

_current = contextvars.ContextVar('pair_trace', default=None)


def journal_path() -> Path:
    return user_cache_path('codex-pair-bridge', appauthor=False) / 'requests.jsonl'


def reason(exc: Exception) -> str:
    message = str(exc).lower()
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


def _append(row: dict) -> None:
    path = journal_path()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    data = (json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n').encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(fd, data)
    finally:
        os.close(fd)


def traced(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        started = time.monotonic()
        trace = {'request_id': uuid.uuid4().hex, 'started_at': int(time.time()),
                 'operation': func.__name__, 'device': None, 'model': None,
                 'stage': 'start', 'stage_started': started, 'durations_ms': {}}
        token = _current.set(trace)
        try:
            result = func(*args, **kwargs)
            trace['status'] = 'ok'
            return result
        except Exception as exc:
            trace['status'] = 'error'
            trace['reason'] = reason(exc)
            raise
        finally:
            now = time.monotonic()
            trace['durations_ms'][trace['stage']] = round((now - trace['stage_started']) * 1000)
            row = {k: trace[k] for k in ('request_id', 'started_at', 'operation', 'device', 'model',
                                          'stage', 'durations_ms', 'status')}
            if 'reason' in trace:
                row['reason'] = trace['reason']
            row['total_ms'] = round((now - started) * 1000)
            try:
                _append(row)
            except OSError:
                pass  # Diagnostics must never turn a successful answer into a failure.
            _current.reset(token)
    return wrapper


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
                 'stage': name, 'durations_ms': dict(trace['durations_ms']), 'status': 'running'})
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
