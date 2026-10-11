import asyncio
import contextlib
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import httpx
import benchmark_runner as runner
import benchmark_store as store
import benchmark_suites as suites
import benchmarks
import jobs
import management
import server


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        for module, function, name in ((store, 'path', 'runs.db'), (jobs, 'journal_path', 'jobs.jsonl'),
                                       (benchmarks, 'path', 'legacy.jsonl')):
            patcher = patch.object(module, function, return_value=root / name)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.addCleanup(self.temp.cleanup)

    def row(self, model='m', suite='short', repeats=1, context=8192, status='completed'):
        data = suites.suite(suite)
        row = store.create('pc', model, data, repeats, 256)
        row.update(status=status, context_length=context, engine='lmstudio')
        for repeat in range(repeats):
            for case in data['cases']:
                row['samples'].append({'case_id': case['id'], 'category': case['category'],
                                       'language': case['language'], 'repeat': repeat+1, 'status': 'pass',
                                       'latency_ms': 100, 'ttft_ms': 20, 'tokens_per_second': None,
                                       'completion_tokens': None})
        store.persist(row)
        return row

    def test_nested_deterministic_bilingual_and_oracles(self):
        smoke, short, full = [suites.suite(name) for name in ('smoke', 'short', 'full')]
        self.assertEqual([len(row['cases']) for row in (smoke, short, full)], [6, 20, 100])
        self.assertEqual(smoke['cases'], short['cases'][:6])
        self.assertEqual(short['cases'], full['cases'][:20])
        self.assertEqual(full, suites.suite('full'))
        self.assertEqual(len({case['id'] for case in full['cases']}), 100)
        for case in full['cases']:
            answer = json.dumps(case['expected']) if isinstance(case['expected'], dict) else case['expected']
            self.assertEqual(suites.evaluate(case, answer, 'stop'), 'pass')
            self.assertEqual(suites.evaluate(case, answer, 'length'), 'format_error')
        self.assertEqual(suites.catalog()['suites'][-1]['languages'], {'ru': 50, 'en': 50})
        # Source-grounded representative oracles, independently calculated.
        cases = {c['id']: c for c in full['cases']}
        self.assertEqual(cases['arithmetic-01-en']['expected'], '30')
        self.assertEqual(cases['code-02-ru']['expected'], '12')
        self.assertEqual(cases['logic-02-en']['expected'], 'B')
        self.assertIn('B > C, C > D, D > A', cases['logic-02-en']['prompt'])
        self.assertIn('target_9: VALUE_46', cases['long_context-10-ru']['prompt'])
        with self.assertRaises(ValueError):
            suites.suite('unknown')

    def test_validation_separates_format_and_wrong_content(self):
        case = next(c for c in suites.suite('short')['cases'] if c['category'] == 'json')
        for answer in ('{"count":7,"ok":1}', '{"count":7.0,"ok":true}',
                       '{"count":7,"count":7,"ok":true}', '{"count":NaN,"ok":true}',
                       '```json\n{"count":7,"ok":true}\n```', '{"count":7,"ok":true,"extra":0}'):
            self.assertEqual(suites.evaluate(case, answer, 'stop'), 'format_error')
        self.assertEqual(suites.evaluate(case, '{"count":8,"ok":true}', 'stop'), 'mismatch')
        self.assertEqual(suites.evaluate(case, '{"ok":true,"count":7}', 'stop'), 'pass')
        exact = suites.suite('smoke')['cases'][0]
        self.assertEqual(suites.evaluate(exact, 'The answer is 30', 'stop'), 'mismatch')

    def test_persistence_metrics_privacy_and_restart(self):
        row = self.row()
        row['samples'][0]['status'] = 'mismatch'
        row['samples'][1]['status'] = 'format_error'
        row['samples'][2]['status'] = 'timeout'
        row['samples'][2]['latency_ms'] = 999999
        row['samples'][3]['latency_ms'] = 400
        store.persist(row)
        result = store.results(run_id=row['run_id'])['results'][0]
        self.assertEqual(result['quality']['passed'], 17)
        self.assertEqual(result['quality']['total'], 20)
        self.assertEqual(result['quality']['by_language']['ru']['total'], 10)
        self.assertEqual(result['errors']['by_type'], {'format_error': 1, 'timeout': 1})
        self.assertEqual(result['errors']['incorrect_answers'], 1)
        self.assertEqual(result['speed']['median_latency_ms']['value'], 100)
        self.assertEqual(result['speed']['p95_latency_ms']['value'], 400)
        self.assertIsNone(result['speed']['median_tokens_per_second']['value'])
        self.assertIsNotNone(result['speed']['median_tokens_per_second']['reason'])
        raw = store.path().read_bytes()
        for value in (b'"prompt":', b'"expected":', b'"answer":', b'VALUE_37', b'http://'):
            self.assertNotIn(value, raw)
        with patch.object(store, 'SESSION', 'new-process'):
            self.assertEqual(store.results(run_id=row['run_id'])['results'][0]['status'], 'completed')
            interrupted = self.row(status='running')
        self.assertEqual(store.results(run_id=interrupted['run_id'])['results'][0]['status'], 'interrupted_or_other_process')

    def test_filters_empty_pagination_and_comparison(self):
        self.assertEqual(server.pair_benchmark_results()['status'], 'not_measured')
        a, b = self.row('a'), self.row('b')
        self.assertEqual(store.results(model='a')['total'], 1)
        page = store.results(limit=1)
        self.assertEqual(page['total'], 2)
        self.assertEqual(page['next_offset'], 1)
        self.assertEqual(store.results(limit=1, offset=1)['next_offset'], None)

        result = server.pair_benchmark_compare([a['run_id'], b['run_id']])
        self.assertEqual(result['status'], 'comparable')
        self.assertIn('| a / pc |', result['table_markdown'])
        c = self.row(context=16384)
        with self.assertRaisesRegex(ValueError, 'not comparable'):
            store.compare([a['run_id'], c['run_id']])
        c = self.row(status='failed')
        with self.assertRaisesRegex(ValueError, 'completed'):
            store.compare([c['run_id']])
        with self.assertRaises(ValueError):
            store.compare(['absent'])
        with self.assertRaises(ValueError):
            server.pair_benchmark_results(profile='code', suite='short')

    def test_corrupt_history_does_not_disable_routing_or_pagination(self):
        valid = self.row()
        bad_payloads = ['{broken', 'null', '[]', '{}', json.dumps(dict(valid, samples=[None]))]
        with store._connect() as connection:
            for index, payload in enumerate(bad_payloads):
                connection.execute('INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?)',
                                   (str(index), valid['created_at'] + index + 1, 'pc', 'm', 'short', payload))
        self.assertIn(('pc', 'm'), benchmarks.summaries('code', 8192))
        page = store.results(limit=5)
        self.assertEqual(page['results'], [])
        self.assertEqual(page['invalid_records'], 5)
        self.assertEqual(page['history_warning'], 'invalid_records_skipped')
        self.assertEqual(page['next_offset'], 5)
        last = store.results(limit=5, offset=5)
        self.assertEqual(last['results'][0]['run_id'], valid['run_id'])
        self.assertIsNone(last['next_offset'])

    def test_routing_current_complete_workloads_only_and_legacy(self):
        self.row('smoke', 'smoke')
        self.assertEqual(store.routing('general', 8192), {})
        a = self.row('a')
        self.assertEqual(benchmarks.summaries('code', 8192)[('pc', 'a')]['pass_rate'], 1)
        self.row('failed', status='failed')
        b = self.row('b', context=16384)
        self.assertEqual(set(store.routing('general', 8192)), {('pc', 'b')})
        b['suite_fingerprint'] = 'obsolete'
        store.persist(b)
        self.assertEqual(set(store.routing('general', 8192)), {('pc', 'a')})
        a['created_at'] = 0
        store.persist(a)
        self.assertEqual(store.routing('general', 8192), {})
        benchmarks.save('pc', 'legacy', 'code', 8192,
                        [{'passed': True, 'latency_ms': 123}] * 3)
        self.assertEqual(server.pair_benchmark_results('code', 8192)['results'][0]['model'], 'legacy')

    def test_public_routing_eligibility_matches_current_complete_evidence(self):
        for mutate in ('obsolete', 'partial', 'old'):
            row = self.row()
            if mutate == 'obsolete':
                row['suite_fingerprint'] = 'old-fingerprint'
            elif mutate == 'partial':
                row['samples'].pop()
            else:
                row['created_at'] = 0
            store.persist(row)
            self.assertFalse(store.results(run_id=row['run_id'])['results'][0]['routing_eligible'])

    def test_terminal_publication_serializes_with_cancel(self):
        data = suites.suite('smoke')
        row = store.create('pc', 'm', data, 1, 256)
        original_persist = store.persist
        entered = threading.Event()
        observations = []
        callers = []
        def measure(job, client, payload):
            with jobs._lock:
                jobs._jobs[job.id] = job
            case = next(c for c in data['cases'] if c['prompt'] == payload['messages'][0]['content'])
            answer = json.dumps(case['expected']) if isinstance(case['expected'], dict) else case['expected']
            return dict(answer=answer, finish_reason='stop', latency_ms=1, ttft_ms=1,
                        completion_tokens=1, tokens_per_second=1)
        def persist(current):
            if current['status'] == 'completed':
                job = jobs._jobs[current['job_id']]
                def cancel():
                    entered.set()
                    observations.append(jobs.cancel(job.id)['status'])
                caller = threading.Thread(target=cancel, daemon=True)
                callers.append(caller)
                caller.start()
                self.assertTrue(entered.wait(3))
                # Another thread must not acquire the terminal lock while we persist.
                acquired = []
                def probe():
                    with_acquired = job.lock.acquire(blocking=False)
                    acquired.append(with_acquired)
                    if with_acquired:
                        job.lock.release()
                probe_thread = threading.Thread(target=probe)
                probe_thread.start(); probe_thread.join(3)
                self.assertEqual(acquired, [False])
            original_persist(current)
        with patch.object(store, 'persist', side_effect=persist):
            job = self.run_worker(row, data, measure)
        for caller in callers:
            caller.join(3)
        with jobs._lock:
            jobs._jobs.pop(job.id, None)
        self.assertEqual(job.status, 'completed')
        self.assertEqual(observations, ['completed'])
        self.assertFalse(job.cancel_event.is_set())

    def run_worker(self, row, data, measure, context=8192, engine='lmstudio', thinking_supported=True):
        model = {'key': row['model'], 'type': 'llm', 'max_context_length': context,
                 'loaded_instances': [{'id': 'exact-instance', 'config': {'context_length': context,
                                                                         'private': 'NEVER_SAVE'}}]}
        job = jobs.Job('pc', row['model'])
        with patch.object(management, 'client', return_value=contextlib.nullcontext(object())), \
             patch.object(management, 'find_model', return_value=model), \
             patch.object(management, 'engine_for', return_value=engine), \
             patch.object(management, 'request', return_value={'components': {'schemas': {'ChatCompletionRequest': {
                 'properties': {'enable_thinking': {}} if thinking_supported else {}}}}}), \
             patch.object(management, 'chat_model_id', return_value='exact-instance'), \
             patch.object(runner, 'measure', side_effect=measure), \
             patch.object(management, 'load_model') as load, patch.object(management, 'unload_model') as unload:
            runner.worker(job, row, data, lambda *a, **k: contextlib.nullcontext())
            load.assert_not_called()
            unload.assert_not_called()
        return job

    def test_explicit_thinking_and_timeout_are_guarded_workload_settings(self):
        data = suites.suite('smoke')
        row = store.create('pc', 'm', data, 1, 16384, 300, False)
        calls = []
        def measure(job, client, payload):
            calls.append((payload['enable_thinking'], job.benchmark_request_timeout_seconds, payload['max_tokens']))
            case = next(c for c in data['cases'] if c['prompt'] == payload['messages'][0]['content'])
            answer = json.dumps(case['expected']) if isinstance(case['expected'], dict) else case['expected']
            return dict(answer=answer, finish_reason='stop', latency_ms=1, ttft_ms=1,
                        completion_tokens=1, tokens_per_second=1)
        job = self.run_worker(row, data, measure, context=32768, engine='unsloth')
        self.assertEqual(job.status, 'completed')
        self.assertEqual(calls, [(False, 300, 16384)] * 7)
        for engine, supported in [('lmstudio', True), ('unsloth', False)]:
            calls.clear()
            bad = store.create('pc', 'm', data, 1, 256, 300, False)
            job = self.run_worker(bad, data, measure, engine=engine, thinking_supported=supported)
            self.assertEqual(job.status, 'failed')
            self.assertEqual(calls, [])
        a = self.row('a')
        legacy = dict(a)
        legacy.pop('enable_thinking'); legacy.pop('request_timeout_seconds')
        self.assertEqual(store.workload(a), store.workload(legacy))
        for changes in ({'enable_thinking': False}, {'enable_thinking': True}, {'request_timeout_seconds': 300}):
            changed = dict(a, **changes)
            store.persist(changed)
            self.assertNotEqual(store.workload(a), store.workload(changed))
            other = self.row('other')
            with self.assertRaisesRegex(ValueError, 'not comparable'):
                store.compare([changed['run_id'], other['run_id']])
        for timeout in (True, 29, 1201, '300'):
            with self.assertRaises(ValueError):
                runner.validate_request('pc', 'm', 'smoke', 1, 256, timeout)
        with self.assertRaises(ValueError):
            runner.validate_request('pc', 'm', 'smoke', 1, 256, 180, 1)

    def test_worker_exact_target_warmup_repeat_and_privacy(self):
        data = suites.suite('smoke')
        row = store.create('pc', 'm', data, 2, 256)
        calls = []
        def measure(job, client, payload):
            calls.append(payload)
            self.assertEqual(payload['model'], 'exact-instance')
            prompt = payload['messages'][0]['content']
            case = next(c for c in data['cases'] if c['prompt'] == prompt)
            answer = json.dumps(case['expected']) if isinstance(case['expected'], dict) else case['expected']
            return dict(answer=answer, finish_reason='stop', latency_ms=100, ttft_ms=5,
                        tokens_per_second=10, completion_tokens=1)
        job = self.run_worker(row, data, measure)
        self.assertEqual(job.status, 'completed')
        self.assertEqual(len(calls), 13)
        result = store.results(run_id=row['run_id'])['results'][0]
        self.assertEqual(result['quality']['passed'], 12)
        self.assertEqual(result['instance_id'], 'exact-instance')
        self.assertEqual(job.answer, '')
        self.assertNotIn(b'NEVER_SAVE', store.path().read_bytes())
        self.assertNotIn(b'"prompt"', jobs.journal_path().read_bytes())

    def test_worker_timeout_stops_without_retry_and_keeps_partial(self):
        data = suites.suite('smoke')
        row = store.create('pc', 'm', data, 1, 256)
        calls = []
        def measure(*args):
            calls.append(1)
            if len(calls) > 1:
                raise TimeoutError('private prompt must not leak')
            return {}
        job = self.run_worker(row, data, measure)
        result = store.results(run_id=row['run_id'])['results'][0]
        self.assertEqual(len(calls), 2)
        self.assertEqual(job.status, 'failed')
        self.assertEqual(result['errors']['by_type'], {'timeout': 1})
        self.assertEqual(result['measured_samples'], 1)
        self.assertFalse(result['routing_eligible'])
        self.assertNotIn(b'private prompt', store.path().read_bytes())

    def test_worker_cancellation_and_warmup_failure(self):
        data = suites.suite('smoke')
        for exc, status in ((asyncio.CancelledError(), 'cancelled'), (ValueError('http_error'), 'failed')):
            row = store.create('pc', 'm', data, 1, 256)
            job = self.run_worker(row, data, lambda *args: (_ for _ in ()).throw(exc))
            self.assertEqual(job.status, status)
            result = store.results(run_id=row['run_id'])['results'][0]
            self.assertEqual(result['measured_samples'], 0)
            self.assertIsNone(result['quality']['pass_rate'])

    def test_input_rejection_and_capacity_failure(self):
        for value in (0, 6, True):
            with self.assertRaises(ValueError):
                runner.validate_request('pc', 'm', 'short', value, 256)
        with patch.object(jobs, 'create', side_effect=ValueError('capacity')):
            with self.assertRaisesRegex(ValueError, 'capacity'):
                server.pair_benchmark_run('pc', 'm')
        self.assertEqual(store.results()['results'][0]['status'], 'failed')

    def test_benchmark_budget_above_8k_passes_unchanged_and_checks_actual_ceiling(self):
        data = runner.validate_request('pc', 'm', 'smoke', 1, 16384)
        row = store.create('pc', 'm', data, 1, 16384)
        calls = []
        def measure(job, client, payload):
            calls.append(payload['max_tokens'])
            case = next(c for c in data['cases'] if c['prompt'] == payload['messages'][0]['content'])
            answer = json.dumps(case['expected']) if isinstance(case['expected'], dict) else case['expected']
            return dict(answer=answer, finish_reason='stop', latency_ms=1, ttft_ms=1,
                        completion_tokens=1, tokens_per_second=1)
        job = self.run_worker(row, data, measure, context=32768)
        self.assertEqual(job.status, 'completed')
        self.assertEqual(calls, [16384] * 7)
        self.assertEqual(job.output_budget['known_ceiling'], 32768)
        self.assertEqual(job.output_budget['requested_max_tokens'], 16384)
        calls.clear()
        too_large = store.create('pc', 'm', data, 1, 32769)
        job = self.run_worker(too_large, data, measure, context=32768)
        self.assertEqual(job.status, 'failed')
        self.assertEqual(calls, [])
        for value in (True, 0, 31):
            with self.assertRaises(ValueError):
                runner.validate_request('pc', 'm', 'smoke', 1, value)

    def test_cancel_stops_a_blocked_http_stream(self):
        entered = threading.Event()
        class SlowStream(httpx.AsyncByteStream):
            async def __aiter__(self):
                entered.set()
                await asyncio.Event().wait()
                yield b''
        original = httpx.AsyncClient
        transport = httpx.MockTransport(lambda request: httpx.Response(200, stream=SlowStream()))
        def client(*args, **kwargs):
            return original(*args, transport=transport, **kwargs)
        job = jobs.Job('pc', 'm')
        class SyncClient:
            base_url = 'http://localhost'
            headers = {}
        def cancel():
            if entered.wait(3):
                jobs.cancel(job.id)
        thread = threading.Thread(target=cancel, daemon=True)
        with patch.dict(jobs._jobs, {job.id: job}), patch.object(runner.httpx, 'AsyncClient', side_effect=client):
            thread.start()
            try:
                with self.assertRaises(asyncio.CancelledError):
                    runner.measure(job, SyncClient(), {})
            finally:
                thread.join(3)
        self.assertTrue(entered.is_set())
        self.assertTrue(job.cancel_event.is_set())
        self.assertIsNone(job.task)
        self.assertIsNone(job.loop)

    def test_unloaded_ambiguous_and_unknown_context_do_not_send_requests(self):
        for instances in ([], [{'id': 'a'}, {'id': 'b'}], [{'id': 'a', 'config': {}}]):
            data = suites.suite('smoke')
            row = store.create('pc', 'm', data, 1, 256)
            job = jobs.Job('pc', 'm')
            with patch.object(management, 'client', return_value=contextlib.nullcontext(object())), \
                 patch.object(management, 'find_model', return_value={'type': 'llm', 'loaded_instances': instances}), \
                 patch.object(runner, 'measure') as measure:
                runner.worker(job, row, data, lambda *args, **kw: contextlib.nullcontext())
                measure.assert_not_called()
            self.assertEqual(job.status, 'failed')
            self.assertEqual(store.results(run_id=row['run_id'])['results'][0]['measured_samples'], 0)


class StreamTests(unittest.IsolatedAsyncioTestCase):
    async def check_stream(self, events, trailer='data: [DONE]\n\n', status=200):
        body = ''.join('data: ' + json.dumps(e) + '\n\n' for e in events) + trailer
        class Job:
            import threading
            cancel_event = threading.Event()
        async with httpx.AsyncClient(base_url='http://localhost', trust_env=False, transport=httpx.MockTransport(
                lambda request: httpx.Response(status, text=body))) as client:
            return await runner.stream(Job(), client, {})

    async def test_sse_usage_first_content_and_reasoning_exclusion(self):
        events = [{'choices': [{'delta': {'reasoning_content': 'PRIVATE'}}]},
                  {'choices': [{'delta': {'content': '30'}}]},
                  {'choices': [{'delta': {}, 'finish_reason': 'stop'}]},
                  {'choices': [], 'usage': {'completion_tokens': 4}}]
        row = await self.check_stream(events)
        self.assertEqual(row['answer'], '30')
        self.assertEqual(row['completion_tokens'], 4)
        self.assertGreaterEqual(row['ttft_ms'], 0)
        self.assertGreaterEqual(row['tokens_per_second'], 0)
        row = await self.check_stream(events[:-1])
        self.assertIsNone(row['tokens_per_second'])

    async def test_sse_truncation_and_errors(self):
        events = [{'choices': [{'delta': {'content': '30'}, 'finish_reason': 'stop'}]}]
        for kwargs, reason in (({'trailer': ''}, 'incomplete_stream'), ({'status': 500}, 'http_error'),
                               ({'trailer': 'data: {broken}\n\n'}, 'invalid_stream')):
            with self.assertRaisesRegex(ValueError, reason):
                await self.check_stream(events, **kwargs)


if __name__ == '__main__':
    unittest.main()
