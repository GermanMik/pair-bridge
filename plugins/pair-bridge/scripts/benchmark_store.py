"""Local, metrics-only benchmark history and compatible comparison views."""
from __future__ import annotations

from collections import Counter
import contextlib
import hashlib
import json
import math
import sqlite3
import statistics
import time
import uuid

from platformdirs import user_cache_path
import benchmark_suites

SESSION = uuid.uuid4().hex
TERMINAL = {'completed', 'failed', 'cancelled'}


def path():
    return user_cache_path('pair-bridge', appauthor=False) / 'benchmark-runs.sqlite3'


@contextlib.contextmanager
def _connect():
    target = path()
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    c = sqlite3.connect(target, timeout=10)
    try:
        with c:
            c.execute('CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, created REAL NOT NULL, device TEXT NOT NULL, '
                      'model TEXT NOT NULL, suite TEXT NOT NULL, payload TEXT NOT NULL)')
            yield c
    finally:
        c.close()


def _write(row):
    with _connect() as c:
        c.execute('INSERT OR REPLACE INTO runs VALUES (?, ?, ?, ?, ?, ?)',
                  (row['run_id'], row['created_at'], row['device'], row['model'], row['suite'],
                   json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(',', ':'))))


def workload(row):
    keys = ('suite', 'suite_version', 'suite_fingerprint', 'context_length', 'max_tokens',
            'temperature', 'repeats', 'warmup_cases')
    settings = {k: row.get(k) for k in keys}
    # Legacy history used these exact defaults; keep its workload hashes stable.
    if row.get('enable_thinking') is not None or row.get('request_timeout_seconds', 180) != 180:
        settings.update(enable_thinking=row.get('enable_thinking'),
                        request_timeout_seconds=row.get('request_timeout_seconds', 180))
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()


def create(device, model, data, repeats, max_tokens, request_timeout_seconds=180, enable_thinking=None):
    row = {'run_id': uuid.uuid4().hex, 'created_at': time.time(), 'updated_at': time.time(),
           'session': SESSION, 'device': device, 'model': model, 'suite': data['suite'],
           'suite_version': data['suite_version'], 'suite_fingerprint': data['suite_fingerprint'],
           'status': 'queued', 'repeats': repeats, 'max_tokens': max_tokens, 'temperature': 0,
           'warmup_cases': 1, 'expected_samples': len(data['cases']) * repeats, 'samples': [],
           'request_timeout_seconds': request_timeout_seconds, 'enable_thinking': enable_thinking,
           'context_length': None, 'engine': None, 'runtime_version': None, 'quantization': None,
           'hardware': None, 'instance_id': None, 'loaded_config': {}, 'error_code': None,
           'unknown_reasons': {'runtime_version': 'not_collected_by_bridge',
                               'quantization': 'not_collected_by_bridge', 'hardware': 'not_sampled',
                               'load_time_ms': 'already_loaded_model'}, 'load_time_ms': None}
    _write(row)
    return row


def persist(row):
    row['updated_at'] = time.time()
    row['workload_id'] = workload(row)
    _write(row)


def _metric(samples, key, percentile=False):
    values = sorted(s[key] for s in samples if isinstance(s.get(key), (int, float))
                    and not isinstance(s[key], bool) and math.isfinite(s[key]) and s[key] >= 0)
    if not values:
        return {'value': None, 'samples': 0, 'reason': 'not_reported_or_no_successful_samples'}
    value = values[max(0, math.ceil(.95 * len(values)) - 1)] if percentile else statistics.median(values)
    return {'value': round(value, 3), 'samples': len(values), 'reason': None}


def _quality(samples):
    passed = sum(s['status'] == 'pass' for s in samples)
    total = len(samples)
    return {'passed': passed, 'total': total, 'pass_rate': round(passed / total, 4) if total else None}


def summarize(row, max_age_days=30):
    row = dict(row)
    row.setdefault('request_timeout_seconds', 180)
    row.setdefault('enable_thinking', None)
    if row.get('session') != SESSION and row['status'] not in TERMINAL:
        row['status'] = 'interrupted_or_other_process'
    samples = row.pop('samples')
    row.pop('session', None)
    row['measured_samples'] = len(samples)
    row['quality'] = _quality(samples)
    row['quality']['by_category'] = {key: _quality([s for s in samples if s['category'] == key])
                                    for key in sorted({s['category'] for s in samples})}
    row['quality']['by_language'] = {key: _quality([s for s in samples if s['language'] == key])
                                    for key in ('ru', 'en')}
    errors = Counter(s['status'] for s in samples if s['status'] not in ('pass', 'mismatch'))
    row['errors'] = {'total': sum(errors.values()), 'by_type': dict(errors),
                     'incorrect_answers': sum(s['status'] == 'mismatch' for s in samples)}
    # Transport errors are excluded from speed, but valid incorrect answers remain measured.
    measured = [s for s in samples if s['status'] in ('pass', 'mismatch', 'format_error')]
    row['speed'] = {'median_latency_ms': _metric(measured, 'latency_ms'),
                    'p95_latency_ms': _metric(measured, 'latency_ms', percentile=True),
                    'median_ttft_ms': _metric(measured, 'ttft_ms'),
                    'median_tokens_per_second': _metric(measured, 'tokens_per_second')}
    row['cases'] = samples
    row['routing_eligible'] = (
        row['status'] == 'completed' and row['suite'] in ('short', 'full') and
        row['measured_samples'] == row['expected_samples'] and row['measured_samples'] > 0 and
        row['created_at'] >= time.time() - max_age_days * 86400 and
        row['suite_version'] == benchmark_suites.VERSION and
        row['suite_fingerprint'] == benchmark_suites.suite(row['suite'])['suite_fingerprint'])
    return row


def _read_record(payload, max_age_days=30):
    """Damaged optional history must not disable inference or hide valid rows."""
    try:
        row = summarize(json.loads(payload), max_age_days=max_age_days)
        if (type(row.get('context_length')) not in (int, type(None)) or
                not isinstance(row.get('workload_id'), (str, type(None))) or
                (row['status'] == 'completed' and
                 (type(row.get('context_length')) is not int or not row.get('workload_id')))):
            return None
        return row
    except (ValueError, TypeError, KeyError, AttributeError, OverflowError, RecursionError):
        return None


def results(device=None, model=None, suite=None, run_id=None, limit=50, offset=0):
    if type(limit) is not int or not 1 <= limit <= 200 or type(offset) is not int or offset < 0:
        raise ValueError('Results require limit 1-200 and nonnegative offset')
    if suite is not None:
        benchmark_suites.suite(suite)
    conditions, args = [], []
    for field, value in (('device', device), ('model', model), ('suite', suite), ('id', run_id)):
        if value is not None:
            conditions.append(field + ' = ?')
            args.append(value)
    where = (' WHERE ' + ' AND '.join(conditions)) if conditions else ''
    if not path().exists():
        return {'status': 'not_measured', 'results': [], 'total': 0, 'limit': limit,
                'offset': offset, 'next_offset': None}
    with _connect() as c:
        total = c.execute('SELECT COUNT(*) FROM runs' + where, args).fetchone()[0]
        records = c.execute('SELECT payload FROM runs' + where + ' ORDER BY created DESC, id DESC LIMIT ? OFFSET ?',
                            args + [limit, offset]).fetchall()
    decoded = [_read_record(record[0]) for record in records]
    rows = [row for row in decoded if row is not None]
    invalid = len(decoded) - len(rows)
    return {'status': 'measured' if rows else 'not_measured', 'results': rows, 'total': total,
            'invalid_records': invalid, 'history_warning': 'invalid_records_skipped' if invalid else None,
            'limit': limit, 'offset': offset,
            'next_offset': offset + len(records) if offset + len(records) < total else None}


def compare(run_ids):
    if not isinstance(run_ids, list) or not 1 <= len(run_ids) <= 20 or len(set(run_ids)) != len(run_ids):
        raise ValueError('Compare requires 1-20 unique run IDs')
    rows = []
    for run_id in run_ids:
        found = results(run_id=run_id)['results']
        if not found:
            raise ValueError('Unknown benchmark run ID')
        row = found[0]
        if row['status'] != 'completed' or row['measured_samples'] != row['expected_samples']:
            raise ValueError('Only completed benchmark runs may be compared')
        rows.append(row)
    if len({row['workload_id'] for row in rows}) != 1:
        raise ValueError('Benchmark workload settings or suite versions differ; runs are not comparable')
    columns = ['Model / device', 'Suite / version', 'Quality', 'Median ms', 'p95 ms', 'TTFT ms', 'Tokens/s', 'Errors']
    table = ['| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join(['---'] * len(columns)) + ' |']
    def cell(value):
        return '—' if value is None else str(value).replace('|', '\\|').replace('\n', ' ').replace('\r', ' ')
    for row in rows:
        quality, speed = row['quality'], row['speed']
        values = [row['model'] + ' / ' + row['device'], row['suite'] + ' / ' + row['suite_version'],
                  f"{quality['passed']}/{quality['total']}", speed['median_latency_ms']['value'],
                  speed['p95_latency_ms']['value'], speed['median_ttft_ms']['value'],
                  speed['median_tokens_per_second']['value'], row['errors']['total']]
        table.append('| ' + ' | '.join(cell(value) for value in values) + ' |')
    return {'status': 'comparable', 'workload_id': rows[0]['workload_id'], 'results': rows,
            'table_markdown': '\n'.join(table),
            'notice': 'Same workload; hardware, engines and quantizations may differ. Inspect each provenance. — means not measured.'}


def routing(profile, context_length, max_age_days=30):
    """Pick one compatible suite, newest complete result per device/model. No pooling."""
    relevant = {'general': None, 'code': 'code', 'analysis': 'logic', 'long_context': 'long_context', 'fast': None}
    if profile not in relevant:
        return {}
    if not path().exists():
        return {}
    cutoff = time.time() - max_age_days * 86400
    with _connect() as c:
        records = c.execute('SELECT payload FROM runs WHERE created >= ? ORDER BY created DESC, id DESC LIMIT 1000',
                            (cutoff,)).fetchall()
    valid = []
    for record in records:
        row = _read_record(record[0], max_age_days=max_age_days)
        if row is None:
            continue
        if (not row['routing_eligible'] or row['context_length'] < context_length or
                row['measured_samples'] != row['expected_samples'] or row['suite_version'] != benchmark_suites.VERSION or
                row['suite_fingerprint'] != benchmark_suites.suite(row['suite'])['suite_fingerprint']):
            continue
        q = row['quality'] if relevant[profile] is None else row['quality']['by_category'].get(relevant[profile])
        latency = row['speed']['median_latency_ms']['value']
        if q and q['total'] and latency is not None:
            valid.append((row, q, latency))
    if not valid:
        return {}
    # The newest eligible run selects the common experiment; never mix suite sizes/settings.
    reference = valid[0][0]['workload_id']
    scores = {}
    for row, q, latency in valid:
        key = (row['device'], row['model'])
        if row['workload_id'] == reference and key not in scores:
            scores[key] = {'cases': q['total'], 'passed': q['passed'], 'pass_rate': q['pass_rate'],
                           'median_latency_ms': latency, 'latest_at': row['created_at'],
                           'suite': row['suite'], 'run_id': row['run_id']}
    return scores
