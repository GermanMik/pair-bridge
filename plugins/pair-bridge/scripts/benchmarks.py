"""Prompt-free local benchmark summaries for explainable model ranking."""
from __future__ import annotations

import json
import os
from pathlib import Path
import statistics
import time
import sqlite3

import benchmark_store

from platformdirs import user_cache_path


PROFILES = {'general', 'code', 'fast', 'long_context', 'analysis'}


def path() -> Path:
    return user_cache_path('pair-bridge', appauthor=False) / 'benchmarks.jsonl'


def save(device: str, model: str, profile: str, context_length: int, outcomes: list[dict]) -> dict:
    """Persist metrics only; prompts, expected answers and generated text are discarded."""
    if profile not in PROFILES or not 3 <= len(outcomes) <= 12:
        raise ValueError('Benchmark requires a known profile and 3–12 cases')
    row = {'created_at': int(time.time()), 'device': device, 'model': model, 'profile': profile,
           'context_length': context_length, 'cases': len(outcomes),
           'passed': sum(bool(x['passed']) for x in outcomes),
           'median_latency_ms': round(statistics.median(x['latency_ms'] for x in outcomes))}
    target = path()
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(fd, (json.dumps(row, separators=(',', ':')) + '\n').encode())
    finally:
        os.close(fd)
    return row


def summaries(profile: str, context_length: int, max_age_days: int = 30) -> dict[tuple[str, str], dict]:
    try:
        measured = benchmark_store.routing(profile, context_length, max_age_days)
    except (OSError, sqlite3.Error):
        measured = {}  # Optional history must not prevent ordinary inference.
    if measured:
        return measured
    target = path()
    if not target.exists():
        return {}
    grouped: dict[tuple[str, str], list[dict]] = {}
    cutoff = time.time() - max_age_days * 86400
    for line in target.read_text(errors='replace').splitlines()[-1000:]:
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if not isinstance(row, dict) or row.get('profile') != profile or row.get('created_at', 0) < cutoff or row.get('context_length', 0) < context_length:
            continue
        if not isinstance(row.get('cases'), int) or not isinstance(row.get('passed'), int) or not isinstance(row.get('median_latency_ms'), int):
            continue
        grouped.setdefault((row.get('device'), row.get('model')), []).append(row)
    result = {}
    for key, rows in grouped.items():
        count = sum(row['cases'] for row in rows)
        if count < 3:
            continue
        result[key] = {'cases': count, 'passed': sum(row['passed'] for row in rows),
                       'pass_rate': round(sum(row['passed'] for row in rows) / count, 3),
                       'median_latency_ms': round(statistics.median(row['median_latency_ms'] for row in rows)),
                       'latest_at': max(row['created_at'] for row in rows)}
    return result
