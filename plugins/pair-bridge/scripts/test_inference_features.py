"""Contract, negative and concurrency proof for the MCP expansion."""
import asyncio
import base64
import contextlib
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from mcp import types

import inference_io as io
import jobs
import management
import server


SCHEMA = {'type': 'object', 'properties': {'ok': {'type': 'boolean'}},
          'required': ['ok'], 'additionalProperties': False}
PNG = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII='


class FeatureTests(unittest.TestCase):
    def test_output_budget_uses_actual_instance_and_explicit_model_limits(self):
        item = self.item(max_context_length=101376, loaded_instances=[
            {'id': 'exact-i', 'config': {'context_length': 94976}}])
        budget = io.output_budget(item, 16384, 8192)
        self.assertEqual(budget['known_ceiling'], 94976)
        self.assertEqual(budget['requested_max_tokens'], 16384)
        with self.assertRaisesRegex(ValueError, 'ceiling 94976'):
            io.output_budget(item, 94977)
        item['max_output_tokens'] = 12000
        with self.assertRaisesRegex(ValueError, 'ceiling 12000'):
            io.output_budget(item, 16384)
        self.assertEqual(io.output_budget({'loaded_instances': []}, 16384)['status'], 'unknown')
        for value in [True, '8192', 0, -1]:
            self.assertIsNone(io.output_budget({'max_context_length': value})['known_ceiling'])
        with self.assertRaisesRegex(ValueError, 'ceiling 8192'):
            io.output_budget({'loaded_instances': [], 'max_context_length': 101376}, 16384, 8192)

    def test_large_output_budget_payload_and_reject_before_inference(self):
        item = self.item(max_context_length=65536, loaded_instances=[
            {'id': 'exact-i', 'config': {'context_length': 32768}}], capabilities={'vision': True})
        with patch.object(management, 'client'), patch.object(management, 'find_model', return_value=item), \
                patch.object(management, 'engine_for', return_value='lmstudio'), \
                patch.object(management, 'request', return_value={'choices': [
                    {'message': {'content': 'OK'}, 'finish_reason': 'stop'}]}) as request:
            for invoke in [lambda n: server.pair_ask('model', 'prompt', max_tokens=n, device='pc'),
                           lambda n: server.pair_vision_ask('pc', 'model', 'prompt', [PNG], max_tokens=n)]:
                result = invoke(16384)
                self.assertEqual(request.call_args.args[3]['max_tokens'], 16384)
                self.assertEqual(result['output_budget']['known_ceiling'], 32768)
                request.reset_mock()
                with self.assertRaisesRegex(ValueError, 'ceiling 32768'):
                    invoke(32769)
                request.assert_not_called()

    def test_budget_rejection_precedes_cold_job_load_or_eviction(self):
        item = self.item(loaded_instances=[], max_context_length=65536)
        job = jobs.Job('pc', 'model')
        with patch.object(management, 'client'), patch.object(management, 'models', return_value=[item]), \
                patch.object(management, 'preflight_for_load') as preflight, \
                patch.object(management, 'load_with_auto_unload') as load:
            server._run_job(job, 'prompt', 8192, 16384, False)
        self.assertEqual(job.status, 'failed')
        preflight.assert_not_called()
        load.assert_not_called()

    def test_smart_selection_skips_budget_incompatible_candidate_before_preflight(self):
        small = self.item(key='small', max_output_tokens=4096)
        large = self.item(key='large', max_context_length=65536, loaded_instances=[
            {'id': 'large-i', 'config': {'context_length': 32768}}])
        snapshot = [{'device': 'pc', 'online': True, 'models': [small, large]}]
        with patch.object(server, 'rank_models', return_value=[('pc', small), ('pc', large)]), \
                patch.object(management, 'devices', return_value={'pc': {}}), \
                patch.object(management, 'preflight_for_load') as preflight:
            result = server.select_model_for_memory(snapshot, 4096, 'general', max_tokens=16384)
        self.assertEqual(result[1]['key'], 'large')
        preflight.assert_not_called()

    def test_router_budget_unknown_or_reported_is_never_clamped(self):
        answer = {'choices': [{'message': {'content': 'OK'}, 'finish_reason': 'stop'}]}
        for metadata in [{}, {'max_output_tokens': 12000}]:
            row = dict(id='model', kind_hint='chat_candidate', **metadata)
            with patch.object(server, 'catalog', return_value=[row]), patch.object(server, 'request', return_value=answer) as request:
                if metadata:
                    with self.assertRaisesRegex(ValueError, 'ceiling 12000'):
                        server.pair_ask('model', 'prompt', max_tokens=16384)
                    request.assert_not_called()
                else:
                    result = server.pair_ask('model', 'prompt', max_tokens=16384)
                    self.assertEqual(result['output_budget']['status'], 'unknown')
                    self.assertEqual(request.call_args.args[2]['max_tokens'], 16384)

    def test_inference_schemas_have_no_fixed_output_token_maximum(self):
        async def inspect():
            tools = {tool.name: tool.inputSchema for tool in await server.mcp.list_tools()}
            for name in ['pair_ask', 'pair_smart_ask', 'pair_job_start', 'pair_vision_ask', 'pair_compare']:
                field = tools[name]['properties']['max_tokens']
                self.assertNotIn('maximum', field, name)
                self.assertEqual(field['minimum'], 32)
            schema = tools['pair_batch_start']['$defs']['BatchRequest']['properties']['max_tokens']
            self.assertNotIn('maximum', schema)
        asyncio.run(inspect())
        self.assertEqual(server.BatchRequest(device='pc', model='model', prompt='prompt', max_tokens=16384).max_tokens, 16384)

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        for target, name, value in [(server, 'user_cache_path', Path(tmp.name)),
                                    (jobs, 'journal_path', Path(tmp.name) / 'jobs.jsonl')]:
            patcher = patch.object(target, name, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)
        for name in ('_jobs', '_batches'):
            patcher = patch.object(jobs, name, {})
            patcher.start()
            self.addCleanup(patcher.stop)
        self.journal = Path(tmp.name) / 'jobs.jsonl'

    def item(self, **kw):
        return dict({'key': 'model', 'type': 'llm', 'loaded_instances': [
            {'id': 'exact-i', 'config': {'context_length': 4096}}], 'max_context_length': 8192}, **kw)

    def test_capability_metadata_unknown_is_not_false_or_name_guess(self):
        item = self.item(capabilities={'vision': True, 'trained_for_tool_use': False})
        with patch.object(management, 'client'), patch.object(management, 'find_model', return_value=item), \
                patch.object(management, 'engine_for', return_value='lmstudio'), patch.object(management, 'load_model') as load:
            result = server.pair_model_capabilities('pc', 'vision-in-name')
        self.assertEqual(result['capabilities']['vision']['status'], 'supported')
        self.assertEqual(result['capabilities']['tool_use']['status'], 'unsupported')
        self.assertEqual(result['capabilities']['json_schema']['status'], 'unknown')
        self.assertEqual(result['loaded_contexts'][0]['context_length'], 4096)
        self.assertTrue(result['checked_at'])
        load.assert_not_called()
        result = io.model_capabilities(self.item(type_source='name_hint'), 'unsloth')
        self.assertEqual(result['capabilities']['chat']['status'], 'unknown')

    def test_explicit_unsloth_task_controls_inference_type_consistently(self):
        for task, expected in [('image-text-to-text', 'llm'), ('visual-question-answering', 'llm'),
                               ('sentence-similarity', 'embedding'), ('feature-extraction', 'embedding'),
                               ('text-generation', 'llm')]:
            row = {'id': 'embed-name-hint', 'loaded': True, 'task': task}
            with patch.object(management, 'engine_for', return_value='unsloth'), \
                    patch.object(management, 'request', return_value={'data': [row]}):
                item = management.models(None, 'pc')[0]
            self.assertEqual(item['type'], expected)
            self.assertEqual(io.model_capabilities(item, 'unsloth')['capabilities'][
                'chat' if expected == 'llm' else 'embeddings']['status'], 'supported')

    def test_inventory_keeps_only_explicit_capabilities(self):
        for engine, response in [('lmstudio', {'models': [self.item(capabilities={'vision': True, 'secret': 'private'})]}),
                                 ('unsloth', {'data': [{'id': 'model', 'loaded': True, 'type': 'embedding',
                                                     'capabilities': {'vision': False}}]})]:
            with patch.object(management, 'engine_for', return_value=engine), patch.object(management, 'request', return_value=response):
                item = management.models(None, 'pc')[0]
            self.assertNotIn('secret', item['capabilities'])
            if engine == 'unsloth':
                self.assertEqual(item['type'], 'embedding')
                self.assertEqual(item['type_source'], 'engine_metadata')

    def test_unsloth_task_evidence_does_not_attribute_name_hints_to_engine(self):
        response = {'data': [{'id': 'embed-in-name', 'loaded': False, 'task': 'text-generation'},
                             {'id': 'generic', 'loaded': False, 'task': 'unknown-task'},
                             {'id': 'feature-model', 'loaded': False, 'task': 'feature_extraction'}]}
        with patch.object(management, 'engine_for', return_value='unsloth'), patch.object(management, 'request', return_value=response):
            items = management.models(None, 'pc')
        self.assertEqual(io.model_capabilities(items[0], 'unsloth')['capabilities']['embeddings']['status'], 'unsupported')
        self.assertEqual(io.model_capabilities(items[1], 'unsloth')['capabilities']['chat']['status'], 'unknown')
        self.assertEqual(io.model_capabilities(items[2], 'unsloth')['capabilities']['embeddings']['status'], 'supported')
        self.assertEqual(io.model_capabilities(items[2], 'unsloth')['capabilities']['embeddings']['source'], 'engine.task')

    def test_structured_formats_and_no_network_schema_resolution(self):
        fmt = io.response_format(json_schema=SCHEMA)
        good = {'answer': '{"ok":true}', 'finish_reason': 'stop', 'truncated': False}
        self.assertEqual(io.validate_completion(good, fmt)['structured_output'], {'ok': True})
        for answer in ['private-not-json', '{"ok":"private"}', '{"ok":true,"extra":1}',
                       '{"ok":true,"ok":false}', '{"ok":NaN}']:
            with self.subTest(answer=answer), self.assertRaisesRegex(ValueError, 'invalid structured') as error:
                io.validate_completion(dict(good, answer=answer), fmt)
            self.assertNotIn(answer, str(error.exception))
        with self.assertRaisesRegex(ValueError, 'finish normally'):
            io.validate_completion(dict(good, truncated=True), fmt)
        for schema in [{'$ref': 'https://example.com/private'}, {'$ref': '#/$defs/self'}, {'type': 'wrong'}]:
            with self.assertRaises(ValueError):
                io.response_format(json_schema=schema)
        with self.assertRaises(ValueError):
            io.response_format({'type': 'json_object'}, SCHEMA)
        number_fmt = io.response_format(json_schema={'type': 'number'})
        with self.assertRaisesRegex(ValueError, 'invalid structured'):
            io.validate_completion(dict(good, answer='1e999'), number_fmt)

    def test_ask_passes_schema_and_validates_exact_instance_response(self):
        requests = []
        def handler(request):
            if request.method == 'GET':
                return httpx.Response(200, json={'models': [self.item()]})
            requests.append(json.loads(request.content))
            return httpx.Response(200, json={'model': 'exact-i', 'choices': [
                {'message': {'content': '{"ok":true}'}, 'finish_reason': 'stop'}]})
        @contextlib.contextmanager
        def client(device):
            with httpx.Client(base_url='http://engine', transport=httpx.MockTransport(handler)) as c:
                yield c
        with patch.object(management, 'client', side_effect=client), \
                patch.object(management, 'engine_for', return_value='lmstudio'):
            result = server.pair_ask('model', 'prompt', device='pc', json_schema=SCHEMA)
        self.assertEqual(result['structured_validation'], 'passed')
        self.assertEqual(requests[-1]['model'], 'exact-i')
        self.assertEqual(requests[-1]['response_format']['json_schema']['schema'], SCHEMA)

    def test_invalid_format_fails_before_smart_selection_or_job_start(self):
        with patch.object(server, 'pair_devices') as inventory, patch.object(jobs, 'create') as create:
            with self.assertRaises(ValueError):
                server.pair_smart_ask('prompt', json_schema={'type': 'bad'})
            with self.assertRaises(ValueError):
                server.pair_job_start('pc', 'model', 'prompt', json_schema={'$ref': 'https://example.com'})
            inventory.assert_not_called()
            create.assert_not_called()

    def test_smart_structured_response_preserves_loaded_instance(self):
        item = self.item()
        snapshot = {'devices': [{'device': 'pc', 'online': True, 'models': [item]}]}
        with patch.object(server, 'catalog', return_value=[]), patch.object(server, 'pair_devices', return_value=snapshot), \
                patch.object(management, 'client'), patch.object(management, 'find_model', return_value=item), \
                patch.object(management, 'devices', return_value={'pc': {}}), \
                patch.object(management, 'engine_for', return_value='lmstudio'), \
                patch.object(management, 'unload_model') as unload, \
                patch.object(management, 'request', return_value={'choices': [
                    {'message': {'content': '{"ok":true}'}, 'finish_reason': 'stop'}]}) as request:
            result = server.pair_smart_ask('prompt', model='model', device='pc', context_length=4096, json_schema=SCHEMA)
        self.assertEqual(result['structured_output'], {'ok': True})
        self.assertEqual(result['cleanup'], 'existing_instance_preserved')
        self.assertIn('response_format', request.call_args.args[3])
        unload.assert_not_called()

    def test_structured_job_validates_before_cleanup_and_never_journals_output(self):
        item = self.item()
        fmt = io.response_format(json_schema=SCHEMA)
        for answer, finish, expected in [('{"ok":true}', 'stop', 'completed'),
                                         ('{"ok":true}', 'length', 'failed'),
                                         ('PRIVATE ANSWER', 'stop', 'failed')]:
            async def stream(job, *args):
                job.answer = answer
                job.finish_reason = finish
                return answer
            job = jobs.Job('pc', 'model')
            with patch.object(management, 'client') as client, patch.object(management, 'models', return_value=[item]), \
                    patch.object(management, 'unload_model') as unload, patch.object(server, '_job_stream', side_effect=stream):
                client.return_value.__enter__.return_value.base_url = 'http://engine'
                client.return_value.__enter__.return_value.headers = {}
                server._run_job(job, 'PRIVATE PROMPT', 4096, 128, True, fmt)
            self.assertEqual(job.status, expected)
            unload.assert_not_called()
            if expected == 'completed':
                self.assertEqual(job.snapshot()['structured_output'], {'ok': True})
        self.assertNotIn('PRIVATE', self.journal.read_text())
        self.assertNotIn('structured_output', self.journal.read_text())

    def test_embedding_payload_and_reordered_vectors(self):
        embedding = self.item(type='embedding')
        calls = []
        def request(c, method, route, body):
            calls.append((route, body))
            return {'data': [{'index': 1, 'embedding': [3.0, 4.0]}, {'index': 0, 'embedding': [1.0, 2.0]}]}
        with patch.object(management, 'client'), patch.object(management, 'find_model', return_value=embedding), \
                patch.object(management, 'engine_for', return_value='lmstudio'), patch.object(management, 'request', side_effect=request):
            result = server.pair_embeddings('pc', 'model', ['one', 'two'], expected_dimensions=2)
        self.assertEqual(result['vectors'], [[1.0, 2.0], [3.0, 4.0]])
        self.assertEqual(calls[0], ('/v1/embeddings', {'model': 'exact-i', 'input': ['one', 'two'], 'encoding_format': 'float'}))

    def test_embeddings_fail_closed_on_wrong_models_inputs_and_vectors(self):
        for value in ['', [], ['x'] * 33, ['x' * 8193], ['x' * 8192] * 6]:
            with self.assertRaises(ValueError):
                io.embedding_inputs(value)
        for rows in [[], [{'index': 0, 'embedding': [float('nan')]}],
                     [{'index': 0, 'embedding': [True]}], [{'index': True, 'embedding': [1]}],
                     [{'index': 0, 'embedding': [1]}, {'index': 0, 'embedding': [1]}],
                     [{'index': 0, 'embedding': [1]}, {'index': 1, 'embedding': [1, 2]}]]:
            with self.assertRaises(ValueError):
                io.embeddings({'data': rows}, 2 if len(rows) == 2 else 1)
        with self.assertRaisesRegex(ValueError, 'dimension'):
            io.embeddings({'data': [{'index': 0, 'embedding': [1]}]}, 1, 2)
        with patch.object(management, 'client'), patch.object(management, 'engine_for', return_value='lmstudio'), \
                patch.object(management, 'request') as request:
            for item in [self.item(), self.item(type='embedding', loaded_instances=[])]:
                with patch.object(management, 'find_model', return_value=item), self.assertRaises(ValueError):
                    server.pair_embeddings('pc', 'model', 'text')
            request.assert_not_called()

    def test_vision_requires_confirmed_support_and_uses_multimodal_payload(self):
        item = self.item(capabilities={'vision': True})
        with patch.object(management, 'client'), patch.object(management, 'find_model', return_value=item), \
                patch.object(management, 'engine_for', return_value='lmstudio'), \
                patch.object(management, 'request', return_value={'choices': [
                    {'message': {'content': 'a dot'}, 'finish_reason': 'stop'}]}) as request:
            result = server.pair_vision_ask('pc', 'model', 'describe', [PNG])
        body = request.call_args.args[3]
        self.assertEqual(body['model'], 'exact-i')
        self.assertEqual(body['messages'][0]['content'][1]['image_url']['url'], PNG)
        self.assertEqual(result['image_count'], 1)
        for capability in [None, False]:
            with patch.object(management, 'client'), patch.object(management, 'find_model', return_value=self.item(capabilities={'vision': capability})), \
                    patch.object(management, 'engine_for', return_value='lmstudio'), patch.object(management, 'request') as request:
                with self.assertRaisesRegex(ValueError, 'vision support'):
                    server.pair_vision_ask('pc', 'model', 'describe', [PNG])
                request.assert_not_called()

    def test_vision_rejects_urls_paths_bad_base64_mime_and_oversize_before_client(self):
        urls = ['https://example.com/image.png', 'C:/private.png', 'data:image/png;base64,@@',
                'data:image/jpeg;base64,' + PNG.split(',')[1],
                'data:image/png;base64,' + base64.b64encode(b'x' * (4 * 2**20 + 1)).decode()]
        with patch.object(management, 'client') as client:
            for url in urls:
                with self.assertRaises(ValueError):
                    server.pair_vision_ask('pc', 'model', 'describe', [url])
            client.assert_not_called()

    def test_job_list_redacts_outputs_and_preserves_queue_metadata(self):
        first, second = jobs.Job('pc', 'model'), jobs.Job('pc', 'model')
        jobs._jobs.update({first.id: first, second.id: second})
        first.update(status='running', stage='generating', answer='PRIVATE', structured_output={'secret': 'PRIVATE'})
        result = jobs.listing()
        self.assertEqual(result['active_count'], 2)
        self.assertEqual(result['devices']['pc']['running_job_ids'], [first.id])
        self.assertEqual(result['devices']['pc']['queued_job_ids'], [second.id])
        self.assertNotIn('PRIVATE', json.dumps(result))
        self.assertNotIn('PRIVATE', self.journal.read_text())
        self.assertGreaterEqual(result['jobs'][1]['queue_wait_seconds'], 0)

    def test_cancelled_queue_wait_is_frozen_and_batch_cancel_preserves_terminal(self):
        first, second = jobs.Job('pc', 'm'), jobs.Job('pc', 'm')
        jobs._jobs.update({first.id: first, second.id: second})
        jobs._batches['batch'] = [first.id, second.id]
        first.update(status='completed', stage='completed')
        result = server.pair_batch_cancel('batch')
        self.assertEqual([row['status'] for row in result['jobs']], ['completed', 'cancel_requested'])
        self.assertTrue(second.cancel_event.is_set())
        second.update(status='cancelled', stage='cancelled')
        wait = second.snapshot()['queue_wait_seconds']
        with patch.object(jobs.time, 'monotonic', return_value=second.queued_monotonic + 10000):
            self.assertEqual(second.snapshot()['queue_wait_seconds'], wait)

    def test_batch_validates_whole_input_and_capacity_before_workers(self):
        with patch.object(management, 'devices', return_value={'a': {}, 'b': {}}), patch.object(server, '_run_job') as worker:
            with self.assertRaises(ValueError):
                server.pair_batch_start([{'device': 'a', 'model': 'm', 'prompt': 'valid'},
                                        {'device': 'unknown', 'model': 'm', 'prompt': 'invalid'}])
            self.assertEqual(jobs._jobs, {})
            for _ in range(7):
                job = jobs.Job('a', 'm')
                jobs._jobs[job.id] = job
            with self.assertRaisesRegex(ValueError, 'capacity'):
                server.pair_batch_start([{'device': 'a', 'model': 'm', 'prompt': 'one'},
                                        {'device': 'b', 'model': 'm', 'prompt': 'two'}])
            worker.assert_not_called()
            self.assertEqual(len(jobs._jobs), 7)

    def test_batch_start_all_overlaps_and_independent_failure(self):
        barrier = threading.Barrier(2)
        done = threading.Event()
        seen = {}
        finished = []
        def worker(job, prompt, *args):
            try:
                with server.inference_lock(job.device, wait_seconds=1):
                    job.update(status='running', stage='inference')
                    seen[job.device] = prompt
                    barrier.wait(timeout=3)
                    job.update(status='failed' if job.device == 'b' else 'completed', stage='completed', answer='PRIVATE ANSWER')
            finally:
                finished.append(job.id)
                if len(finished) == 2:
                    done.set()
        with patch.object(management, 'devices', return_value={'a': {}, 'b': {}}), patch.object(server, '_run_job', side_effect=worker):
            batch = server.pair_batch_start([{'device': 'a', 'model': 'm', 'prompt': 'task A'},
                                             {'device': 'b', 'model': 'n', 'prompt': 'task B'}])
            self.assertTrue(done.wait(5))
        result = server.pair_batch_status(batch['batch_id'])
        self.assertTrue(result['finished'])
        self.assertEqual([row['status'] for row in result['jobs']], ['completed', 'failed'])
        self.assertEqual(seen, {'a': 'task A', 'b': 'task B'})
        self.assertNotIn('PRIVATE', self.journal.read_text())
        self.assertEqual(server.pair_batch_cancel(batch['batch_id'])['jobs'][0]['status'], 'completed')

    def test_atomic_admission_for_competing_batches(self):
        barrier = threading.Barrier(2)
        release = threading.Event()
        threads, outcomes = [], []
        def submit():
            barrier.wait(timeout=3)
            try:
                result = jobs.create_batch([{'device': 'pc', 'model': 'm', 'prompt': 'p', 'context_length': 8192,
                                             'max_tokens': 64, 'unload_after': False}] * 5,
                                          lambda job, *args: release.wait(3))
                outcomes.append(result['batch_id'])
            except ValueError:
                outcomes.append('rejected')
        for _ in range(2):
            thread = threading.Thread(target=submit)
            threads.append(thread)
            thread.start()
        for thread in threads:
            thread.join(5)
        release.set()
        self.assertEqual(outcomes.count('rejected'), 1)
        self.assertEqual(len(jobs._jobs), 5)

    def test_structured_stream_payload_finish_and_journal_privacy(self):
        fmt = io.response_format(json_schema=SCHEMA)
        body = ('data: ' + json.dumps({'choices': [{'delta': {'content': '{"ok":true}'}, 'finish_reason': None}]}) + '\n\n' +
                'data: ' + json.dumps({'choices': [{'delta': {}, 'finish_reason': 'stop'}]}) + '\n\ndata: [DONE]\n\n')
        async def run():
            job = jobs.Job('pc', 'model')
            def handle(request):
                self.assertEqual(request.url.path, '/v1/chat/completions')
                self.assertEqual(json.loads(request.content)['response_format'], fmt)
                return httpx.Response(200, content=body.encode())
            async with httpx.AsyncClient(base_url='http://engine', transport=httpx.MockTransport(handle)) as c:
                await server._job_stream(job, c, 'exact-i', 'PRIVATE PROMPT', 128, fmt)
            self.assertEqual(job.finish_reason, 'stop')
            return job
        with patch.object(management, 'engine_for', return_value='lmstudio'):
            job = asyncio.run(run())
        self.assertEqual(io.validate_completion({'answer': job.answer, 'finish_reason': job.finish_reason}, fmt)['structured_output'], {'ok': True})
        self.assertNotIn('PRIVATE', self.journal.read_text())


class FeatureProtocolTests(unittest.IsolatedAsyncioTestCase):
    async def call(self, name, arguments):
        request = types.CallToolRequest(method='tools/call', params=types.CallToolRequestParams(name=name, arguments=arguments))
        return (await server.mcp._mcp_server.request_handlers[types.CallToolRequest](request)).root

    async def test_mcp_batch_and_vision_validation_no_mutations(self):
        with patch.object(management, 'client') as client, patch.object(server, '_run_job') as worker, \
                patch.object(management, 'devices', return_value={'pc': {}}):
            for name, args in [('pair_batch_start', {'requests': []}),
                               ('pair_batch_start', {'requests': [{'device': 'pc', 'model': 'm', 'prompt': 'p', 'max_tokens': -1}]}),
                               ('pair_vision_ask', {'device': 'pc', 'model': 'm', 'prompt': 'p', 'images': ['https://example.com/private']}),
                               ('pair_embeddings', {'device': 'pc', 'model': 'm', 'input': ''}),
                               ('pair_ask', {'model': 'm', 'prompt': 'p', 'json_schema': {'$ref': 'https://example.com/private'}})]:
                result = await self.call(name, args)
                self.assertTrue(result.isError, name)
            client.assert_not_called()
            worker.assert_not_called()

    async def test_mcp_capabilities_serialization_and_job_list_privacy(self):
        item = {'key': 'model', 'type': 'llm', 'capabilities': {'vision': True}, 'loaded_instances': []}
        with patch.object(management, 'client'), patch.object(management, 'find_model', return_value=item), \
                patch.object(management, 'engine_for', return_value='lmstudio'):
            result = await self.call('pair_model_capabilities', {'device': 'pc', 'model': 'model'})
        self.assertFalse(result.isError)
        data = result.structuredContent or json.loads(result.content[0].text)
        self.assertEqual(data['capabilities']['vision']['status'], 'supported')
        with patch.object(jobs, '_jobs', {}):
            result = await self.call('pair_job_list', {})
        self.assertFalse(result.isError)
        data = result.structuredContent or json.loads(result.content[0].text)
        self.assertEqual(data['scope'], 'current_process')


if __name__ == '__main__':
    unittest.main()
