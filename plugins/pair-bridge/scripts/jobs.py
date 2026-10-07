"""Bounded in-process inference jobs with prompt-free recovery metadata."""
from __future__ import annotations

import json
import os
from pathlib import Path
import threading
import time
import uuid

from platformdirs import user_cache_path


_lock = threading.RLock()
_jobs: dict[str, 'Job'] = {}
_batches: dict[str, list[str]] = {}
TERMINAL = {'completed', 'failed', 'cancelled'}
MAX_ACTIVE_JOBS = 8
MAX_RETAINED_JOBS = 512


def journal_path() -> Path:
    return user_cache_path('pair-bridge', appauthor=False) / 'jobs.jsonl'


def _append(row: dict) -> None:
    target = journal_path()
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(fd, (json.dumps(row, separators=(',', ':')) + '\n').encode())
    finally:
        os.close(fd)


class Job:
    def __init__(self, device: str, model: str):
        self.id = uuid.uuid4().hex
        self.device, self.model = device, model
        self.created_at = int(time.time())
        self.status, self.stage = 'queued', 'queue'
        self.progress = None
        self.instance_id = None
        self.owned = False
        self.cleanup = None
        self.auto_unloaded_instances = []
        self.answer = ''
        self.error_code = None
        self.cancel_event = threading.Event()
        self.response = None
        self.loop = None
        self.task = None
        self.lock = threading.RLock()
        self.updated_at = self.created_at
        self.queued_monotonic = time.monotonic()
        self.started_monotonic = None
        self.finished_monotonic = None
        self.batch_id = None
        self.structured_output = None
        self.finish_reason = None
        self.output_truncated = False
        self.output_budget = None
        self._last_persist = 0.0
        self.update()

    def snapshot(self) -> dict:
        with self.lock:
            return {'job_id': self.id, 'device': self.device, 'model': self.model,
                    'created_at': self.created_at, 'updated_at': self.updated_at,
                    'status': self.status, 'stage': self.stage, 'progress': self.progress,
                    'batch_id': self.batch_id,
                    'queue_wait_seconds': round((self.started_monotonic or self.finished_monotonic or time.monotonic()) - self.queued_monotonic, 3),
                    'instance_id': self.instance_id, 'loaded_for_job': self.owned,
                    'cleanup': self.cleanup, 'auto_unloaded_instances': self.auto_unloaded_instances,
                    'error_code': self.error_code,
                    'answer': self.answer if self.status == 'completed' else None,
                    'structured_output': self.structured_output if self.status == 'completed' else None,
                    'finish_reason': self.finish_reason,
                    'output_truncated': self.output_truncated,
                    'output_budget': self.output_budget,
                    'partial_answer': self.answer[-4000:] if self.status in ('running', 'cancel_requested') else None}

    def update(self, **changes) -> None:
        with self.lock:
            old_status, old_stage = self.status, self.stage
            for key, value in changes.items():
                setattr(self, key, value)
            if self.status == 'running' and self.started_monotonic is None:
                self.started_monotonic = time.monotonic()
            if self.status in TERMINAL and self.finished_monotonic is None:
                self.finished_monotonic = time.monotonic()
            self.updated_at = int(time.time())
            now = time.monotonic()
            if self.status == old_status and self.stage == old_stage and now - self._last_persist < 1:
                return
            self._last_persist = now
            row = self.snapshot()
            row.pop('answer', None)
            row.pop('partial_answer', None)
            row.pop('structured_output', None)
            try:
                _append(row)
            except OSError:
                pass

    def add_text(self, text: str) -> None:
        with self.lock:
            if not self.cancel_event.is_set() and len(self.answer) + len(text) > 48000:
                self.output_truncated = True
            if not self.cancel_event.is_set() and len(self.answer) < 48000:
                self.answer += text[:48000 - len(self.answer)]


def _start(job, worker, args):
    try:
        threading.Thread(target=worker, args=(job, *args), daemon=True, name='pair-job-' + job.id[:8]).start()
    except RuntimeError:
        job.update(status='failed', stage='failed', error_code='worker_start_failed')


def _prune():
    for key in list(_jobs):
        if len(_jobs) <= MAX_RETAINED_JOBS:
            break
        if _jobs[key].status in TERMINAL:
            del _jobs[key]


def create(device: str, model: str, worker, *args) -> dict:
    with _lock:
        if sum(job.status not in TERMINAL for job in _jobs.values()) >= MAX_ACTIVE_JOBS:
            raise ValueError('Too many active PAIR jobs; wait or cancel one')
        job = Job(device, model)
        _jobs[job.id] = job
        _prune()
    _start(job, worker, args)
    return job.snapshot()


def create_batch(requests: list[dict], worker) -> dict:
    """Reserve all capacity before any worker runs; failures stay per job."""
    with _lock:
        active = sum(job.status not in TERMINAL for job in _jobs.values())
        if active + len(requests) > MAX_ACTIVE_JOBS:
            raise ValueError('Batch exceeds available PAIR job capacity; no jobs were started')
        batch_id = uuid.uuid4().hex
        pending = []
        for request in requests:
            job = Job(request['device'], request['model'])
            job.update(batch_id=batch_id)
            _jobs[job.id] = job
            pending.append((job, (request['prompt'], request['context_length'], request['max_tokens'],
                                  request['unload_after'], request.get('response_format'))))
        _batches[batch_id] = [job.id for job, _ in pending]
        _prune()
        # Retain bounded group metadata; eviction does not cancel live workers.
        if len(_batches) > 64:
            for old, members in list(_batches.items()):
                if old != batch_id and all(_jobs.get(key) is None or _jobs[key].status in TERMINAL for key in members):
                    del _batches[old]
                    break
    for job, args in pending:
        _start(job, worker, args)
    return batch_status(batch_id)


def batch_status(batch_id: str) -> dict:
    with _lock:
        members = list(_batches.get(batch_id, []))
    if not members:
        raise ValueError('Unknown process-local PAIR batch ID')
    rows = []
    for key in members:
        try:
            rows.append(get(key))
        except ValueError:
            rows.append({'job_id': key, 'status': 'unavailable', 'error_code': 'metadata_evicted'})
    return {'batch_id': batch_id, 'scope': 'current_process', 'jobs': rows,
            'finished': all(row['status'] in TERMINAL for row in rows)}


def batch_cancel(batch_id: str) -> dict:
    with _lock:
        members = list(_batches.get(batch_id, []))
    if not members:
        raise ValueError('Unknown process-local PAIR batch ID')
    for key in members:
        try:
            cancel(key)
        except ValueError:
            pass
    return batch_status(batch_id)


def listing(device: str | None = None, include_terminal: bool = False, limit: int = 64) -> dict:
    """Metadata only; locks do not offer FIFO or cross-process queue positions."""
    if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 64:
        raise ValueError('Job list limit must be between 1 and 64')
    with _lock:
        snapshots = [job.snapshot() for job in _jobs.values() if device is None or job.device == device]
    safe = [{k: v for k, v in row.items() if k not in ('answer', 'partial_answer', 'structured_output')}
            for row in snapshots]
    queues = {}
    for row in safe:
        if row['status'] in TERMINAL:
            continue
        queue = queues.setdefault(row['device'], {'queued_job_ids': [], 'running_job_ids': []})
        queue['queued_job_ids' if row['stage'] == 'queue' else 'running_job_ids'].append(row['job_id'])
    selected = [row for row in safe if include_terminal or row['status'] not in TERMINAL]
    return {'scope': 'current_process', 'jobs': selected[-limit:], 'total_matching': len(selected),
            'active_count': sum(row['status'] not in TERMINAL for row in safe),
            'devices': queues, 'queue_order': 'not_guaranteed',
            'notice': 'Other Bridge processes, synchronous calls and external clients are not visible here.'}


def get(job_id: str) -> dict:
    with _lock:
        job = _jobs.get(job_id)
    if job:
        return job.snapshot()
    target = journal_path()
    if target.exists():
        for line in reversed(target.read_text(errors='replace').splitlines()[-1000:]):
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get('job_id') == job_id:
                if row.get('status') not in TERMINAL:
                    row['status'] = 'interrupted_or_restarted'
                    row['stage'] = 'recovery_needed'
                return row
    raise ValueError('Unknown PAIR job ID')


def cancel(job_id: str) -> dict:
    with _lock:
        job = _jobs.get(job_id)
    if not job:
        return get(job_id)
    with job.lock:
        if job.status in TERMINAL:
            return job.snapshot()
        job.cancel_event.set()
        loop, task = job.loop, job.task
        job.update(status='cancel_requested')
    if loop is not None and task is not None:
        try:
            loop.call_soon_threadsafe(task.cancel)
        except RuntimeError:
            pass
    return job.snapshot()
