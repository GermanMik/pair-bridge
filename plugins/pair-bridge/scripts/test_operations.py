import contextlib
import asyncio
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import httpx

import benchmarks
import jobs
import management
import server
import telemetry


class OperationsTests(unittest.TestCase):
    def test_live_gpu_capacity_rejects_cold_load(self):
        candidate = {'key': 'new', 'loaded_instances': []}
        estimate = {'total_bytes': 20 * 2**30, 'gpu_bytes': 19 * 2**30, 'context_length': 8192}
        capacity = {'checked_at': __import__('time').time(),
                    'memory': {'available_bytes': 30 * 2**30},
                    'gpu': {'devices': [{'free_bytes': 2 * 2**30}]}}
        with patch.object(management, 'estimate_memory', return_value=estimate):
            with self.assertRaisesRegex(ValueError, 'GPU memory'):
                management.memory_preflight('pc', [candidate], candidate, 8192, None, capacity)

    def test_benchmark_ranking_is_measured_and_prompt_free(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(benchmarks, 'path', return_value=Path(temp) / 'bench.jsonl'):
            record = benchmarks.save('pc', 'measured', 'code', 8192,
                                     [{'passed': True, 'latency_ms': 100},
                                      {'passed': True, 'latency_ms': 120},
                                      {'passed': False, 'latency_ms': 140}])
            self.assertEqual(record['passed'], 2)
            self.assertNotIn('prompt', (Path(temp) / 'bench.jsonl').read_text())
            inventory = [{'device': 'pc', 'online': True, 'models': [
                {'key': 'measured', 'type': 'llm', 'loaded_instances': [], 'size_bytes': 100},
                {'key': 'coder-untested', 'type': 'llm', 'loaded_instances': [{'id': 'i'}], 'size_bytes': 10}]}]
            self.assertEqual(server.select_model(inventory, task_hint='code')[1]['key'], 'measured')

    def test_stream_collects_message_only_and_progress(self):
        events = [
            {'type': 'chat.start', 'model_instance_id': 'i'},
            {'type': 'prompt_processing.progress', 'progress': .5},
            {'type': 'reasoning.delta', 'content': 'PRIVATE REASONING'},
            {'type': 'message.delta', 'content': 'Hello'},
            {'type': 'chat.end', 'result': {'output': [{'type': 'message', 'content': 'Hello world'}]}}]
        import json
        body = ''.join('event: ' + item['type'] + '\ndata: ' + json.dumps(item) + '\n\n' for item in events)
        client = httpx.AsyncClient(base_url='http://localhost:1234', transport=httpx.MockTransport(
            lambda request: httpx.Response(200, text=body)))
        with tempfile.TemporaryDirectory() as temp, patch.object(jobs, 'journal_path', return_value=Path(temp) / 'jobs.jsonl'), \
             patch.object(server.management, 'engine_for', return_value='lmstudio'):
            job = jobs.Job('pc', 'm')
            self.assertEqual(asyncio.run(server._job_stream(job, client, 'i', 'private prompt', 100)), 'Hello world')
            self.assertNotIn('PRIVATE REASONING', (Path(temp) / 'jobs.jsonl').read_text())
            self.assertNotIn('private prompt', (Path(temp) / 'jobs.jsonl').read_text())
        asyncio.run(client.aclose())

    def test_cancel_blocks_future_deltas_and_recovers_interrupted_job(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(jobs, 'journal_path', return_value=Path(temp) / 'jobs.jsonl'):
            job = jobs.Job('pc', 'm')
            with patch.dict(jobs._jobs, {job.id: job}):
                job.add_text('before')
                self.assertEqual(jobs.cancel(job.id)['status'], 'cancel_requested')
                job.add_text('after')
                self.assertEqual(job.answer, 'before')
            with patch.dict(jobs._jobs, {}, clear=True):
                self.assertEqual(jobs.get(job.id)['status'], 'interrupted_or_restarted')

    def test_windows_telemetry_parsing_and_local_disk_unknown(self):
        payload = '{"memory":{"available_bytes":123,"total_bytes":456},"gpu":{"status":"ok","devices":[{"free_bytes":789}]},"models_disk":null}'
        with patch.object(telemetry, '_run', return_value=payload) as run:
            result = telemetry._windows_over_ssh('alfred', None)
        self.assertEqual(result['gpu']['devices'][0]['free_bytes'], 789)
        self.assertIn('-EncodedCommand', run.call_args.args[0])
        with patch.object(telemetry, '_local_memory', return_value={'available_bytes': 1}), \
             patch.object(telemetry, '_nvidia_gpu', return_value={'status': 'unknown'}):
            result = telemetry.sample({'base_url': 'http://127.0.0.1:1234'})
        self.assertIsNone(result['models_disk'])

    def test_local_windows_memory_sample(self):
        with patch.object(telemetry.platform, 'system', return_value='Windows'), \
             patch.object(telemetry, '_run', return_value='{"total_bytes":34359738368,"available_bytes":17179869184}') as run:
            result = telemetry._local_memory()
        self.assertEqual(result['available_bytes'], 17179869184)
        self.assertEqual(result['kind'], 'windows_free_physical')
        self.assertIn('-EncodedCommand', run.call_args.args[0])


if __name__ == '__main__':
    unittest.main()
