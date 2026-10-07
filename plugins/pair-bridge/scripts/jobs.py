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
TERMINAL = {'completed', 'failed', 'cancelled'}
MAX_ACTIVE_JOBS = 8


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
        self._last_persist = 0.0
        self.update()

    def snapshot(self) -> dict:
        with self.lock:
            return {'job_id': self.id, 'device': self.device, 'model': self.model,
                    'created_at': self.created_at, 'updated_at': self.updated_at,
                    'status': self.status, 'stage': self.stage, 'progress': self.progress,
                    'instance_id': self.instance_id, 'loaded_for_job': self.owned,
                    'cleanup': self.cleanup, 'auto_unloaded_instances': self.auto_unloaded_instances,
                    'error_code': self.error_code,
                    'answer': self.answer if self.status == 'completed' else None,
                    'partial_answer': self.answer[-4000:] if self.status in ('running', 'cancel_requested') else None}

    def update(self, **changes) -> None:
        with self.lock:
            old_status, old_stage = self.status, self.stage
            for key, value in changes.items():
                setattr(self, key, value)
            self.updated_at = int(time.time())
            now = time.monotonic()
            if self.status == old_status and self.stage == old_stage and now - self._last_persist < 1:
                return
            self._last_persist = now
            row = self.snapshot()
            row.pop('answer', None)
            row.pop('partial_answer', None)
            try:
                _append(row)
            except OSError:
                pass

    def add_text(self, text: str) -> None:
        with self.lock:
            if not self.cancel_event.is_set() and len(self.answer) < 48000:
                self.answer += text[:48000 - len(self.answer)]


def create(device: str, model: str, worker, *args) -> dict:
    with _lock:
        if sum(job.status not in TERMINAL for job in _jobs.values()) >= MAX_ACTIVE_JOBS:
            raise ValueError('Too many active PAIR jobs; wait or cancel one')
        job = Job(device, model)
        _jobs[job.id] = job
        if len(_jobs) > 64:
            for key in list(_jobs):
                if _jobs[key].status in TERMINAL and key != job.id:
                    del _jobs[key]
                    break
    threading.Thread(target=worker, args=(job, *args), daemon=True, name='pair-job-' + job.id[:8]).start()
    return job.snapshot()


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
