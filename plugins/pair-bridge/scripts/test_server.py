import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import httpx
import server
import jev
import diagnostics


class BridgeTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        patcher = patch.object(server, "user_cache_path", return_value=Path(temp.name))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_http_error_does_not_echo_prompt_or_retry(self):
        with patch.object(server.httpx, 'Client') as factory:
            client = factory.return_value.__enter__.return_value
            client.request.return_value = httpx.Response(400, json={'error': 'PRIVATE PROMPT'})
            with self.assertRaisesRegex(ValueError, '^PAIR returned HTTP 400') as error:
                server.request('POST', '/chat/completions', {'private': 'data'})
            self.assertNotIn('PRIVATE', str(error.exception))
            self.assertEqual(client.request.call_count, 1)

    def test_timeout_no_retry(self):
        with patch.object(server.httpx, 'Client') as factory:
            client = factory.return_value.__enter__.return_value
            client.request.side_effect = httpx.ReadTimeout('timeout')
            with self.assertRaisesRegex(ValueError, 'may still be running'):
                server.request('POST', '/chat/completions', {})
            self.assertEqual(client.request.call_count, 1)

    def test_lock_rejects_overlap(self):
        with server.inference_lock():
            with self.assertRaisesRegex(ValueError, 'Another Codex PAIR request'):
                with server.inference_lock():
                    self.fail('lock was not held')

    def test_device_locks_are_independent(self):
        with server.inference_lock('mac'):
            with server.inference_lock('pc'):
                pass
            with self.assertRaisesRegex(ValueError, 'this target'):
                with server.inference_lock('mac'):
                    self.fail('same device overlapped')

    def test_content_parts_and_truncation(self):
        with patch.object(server, 'catalog', return_value=[{'id': 'model', 'kind_hint': 'chat_candidate'}]), patch.object(server, 'request', return_value={'model': 'model', 'choices': [{'message': {'content': [{'type': 'text', 'text': 'answer'}]}, 'finish_reason': 'length'}]}):
            result = server.pair_ask('model', 'test')
            self.assertEqual(result['answer'], 'answer')
            self.assertTrue(result['truncated'])

    def test_empty_answer_is_failure(self):
        with patch.object(server, 'catalog', return_value=[{'id': 'model', 'kind_hint': 'chat_candidate'}]), patch.object(server, 'request', return_value={'choices': [{'message': {'content': None, 'reasoning_content': 'private reasoning'}, 'finish_reason': 'length'}]}):
            with self.assertRaisesRegex(ValueError, 'no final text'):
                server.pair_ask('model', 'test')

    def test_diagnostic_journal_redacts_prompt_and_survives_partial_line(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(diagnostics, 'journal_path', return_value=Path(tmp) / 'requests.jsonl'), \
             patch.object(server, 'catalog', return_value=[{'id': 'model', 'kind_hint': 'chat_candidate'}]), \
             patch.object(server, 'request', side_effect=ValueError('PAIR request timed out; PRIVATE PROMPT')):
            with self.assertRaisesRegex(ValueError, 'PRIVATE PROMPT'):
                server.pair_ask('model', 'PRIVATE PROMPT')
            raw = (Path(tmp) / 'requests.jsonl').read_text()
            self.assertNotIn('PRIVATE PROMPT', raw)
            self.assertIn('"reason":"timeout"', raw)
            with (Path(tmp) / 'requests.jsonl').open('a') as out:
                out.write('{partial\n')
            self.assertEqual(diagnostics.recent(5)[0]['reason'], 'timeout')

    def test_running_stage_visible_before_request_completes(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(diagnostics, 'journal_path', return_value=Path(tmp) / 'requests.jsonl'), \
             patch.object(server, 'catalog', return_value=[{'id': 'model', 'kind_hint': 'chat_candidate'}]):
            def respond(*_args):
                rows = diagnostics.recent()
                self.assertEqual(rows[-1]['stage'], 'inference')
                self.assertEqual(rows[-1]['status'], 'running')
                return {'model': 'model', 'choices': [{'message': {'content': 'answer'}, 'finish_reason': 'stop'}]}
            with patch.object(server, 'request', side_effect=respond):
                server.pair_ask('model', 'question')
            self.assertEqual(diagnostics.recent()[-1]['status'], 'ok')

    def test_selector_prefers_loaded_installed_chat(self):
        rows = [{'device': 'pc', 'online': False, 'models': [{'key': 'offline', 'type': 'llm', 'loaded_instances': []}]},
                {'device': 'mac', 'online': True, 'models': [
                    {'key': 'embed', 'type': 'embedding', 'loaded_instances': []},
                    {'key': 'cold', 'type': 'llm', 'size_bytes': 1, 'loaded_instances': []},
                    {'key': 'warm', 'type': 'llm', 'size_bytes': 10, 'loaded_instances': [{'id': 'warm-i'}]}]}]
        self.assertEqual(server.select_model(rows)[1]['key'], 'warm')
        with self.assertRaisesRegex(ValueError, 'No suitable installed'):
            server.select_model(rows, model='gpt-oss-20b')

    def test_empty_router_catalog_falls_back_to_installed_devices(self):
        rows = [{'key': 'qwen', 'display_name': 'Qwen', 'type': 'llm', 'size_bytes': 10,
                 'max_context_length': 32768, 'loaded_instances': []}]
        with patch.object(server, 'catalog', return_value=[]), \
             patch.object(server.management, 'devices', return_value={'alfred': {}, 'offline': {}}), \
             patch.object(server.management, 'client'), \
             patch.object(server.management, 'models', side_effect=[rows, ValueError('offline')]):
            result = server.pair_list()
        self.assertEqual(result['source'], 'configured device inventory fallback')
        self.assertEqual(result['models'][0]['id'], 'qwen')
        self.assertEqual(result['models'][0]['device'], 'alfred')
        self.assertTrue(result['models'][0]['installed'])
        self.assertFalse(result['models'][0]['loaded'])
        self.assertEqual(result['device_errors'], [{'device': 'offline', 'status': 'offline'}])

    def test_nonempty_router_catalog_remains_authoritative(self):
        routed = [{'id': 'loaded', 'kind_hint': 'chat_candidate'}]
        with patch.object(server, 'catalog', return_value=routed), \
             patch.object(server.management, 'devices') as devices:
            result = server.pair_list()
        self.assertEqual(result['models'], routed)
        self.assertEqual(result['source'], 'PAIR routing catalog')
        devices.assert_not_called()

    def test_checked_inventory_shows_load_state_and_last_result(self):
        rows = [{'key': 'warm', 'type': 'llm', 'max_context_length': 8192,
                 'loaded_instances': [{'id': 'warm-i'}], 'size_bytes': 100}]
        with patch.object(server.diagnostics, 'recent', return_value=[{'device': 'mac', 'model': 'warm',
              'status': 'error', 'reason': 'timeout'}]):
            result = server.checked_models('mac', rows)[0]
        self.assertEqual(result['availability'], 'online')
        self.assertEqual(result['load_state'], 'loaded')
        self.assertEqual(result['last_request_reason'], 'timeout')
        self.assertEqual(result['max_context_length'], 8192)

    def test_smart_ask_rechecks_type_before_inference(self):
        initial = {'key': 'warm', 'type': 'llm', 'loaded_instances': [{'id': 'warm-i'}]}
        changed = dict(initial, type='embedding')
        snapshot = {'devices': [{'device': 'mac', 'online': True, 'models': [initial]}]}
        with patch.object(server, 'pair_devices', return_value=snapshot), patch.object(server.management, 'client') as factory, \
             patch.object(server.management, 'find_model', return_value=changed), \
             patch.object(server.management, 'request') as req:
            factory.return_value.__enter__.return_value = object()
            with self.assertRaisesRegex(ValueError, 'no longer a chat LLM'):
                server.pair_smart_ask('question')
            req.assert_not_called()

    def test_profile_selection_and_explicit_override(self):
        rows = [{'device': 'mac', 'online': True, 'models': [
            {'key': 'general', 'type': 'llm', 'loaded_instances': [{'id': 'g'}], 'size_bytes': 20,
             'max_context_length': 8192},
            {'key': 'coder', 'type': 'llm', 'loaded_instances': [], 'size_bytes': 30,
             'max_context_length': 16384},
            {'key': 'long', 'type': 'llm', 'loaded_instances': [], 'size_bytes': 60,
             'max_context_length': 131072}]}]
        self.assertEqual(server.select_model(rows, task_hint='code')[1]['key'], 'coder')
        self.assertEqual(server.select_model(rows, task_hint='fast')[1]['key'], 'general')
        self.assertEqual(server.select_model(rows, task_hint='long_context')[1]['key'], 'long')
        self.assertEqual(server.select_model(rows, task_hint='analysis')[1]['key'], 'general')
        self.assertEqual(server.select_model(rows, model='long', task_hint='fast')[1]['key'], 'long')
        with self.assertRaisesRegex(ValueError, 'No suitable installed'):
            server.select_model(rows, model='missing', task_hint='code')

    def test_smart_selector_skips_ranked_model_that_fails_memory_preflight(self):
        inventory = [{'device': 'pc', 'online': True, 'capacity': {'checked_at': 9999999999}, 'models': [
            {'key': 'qwen', 'type': 'llm', 'size_bytes': 10, 'loaded_instances': []},
            {'key': 'ornith', 'type': 'llm', 'size_bytes': 20, 'loaded_instances': []}]}]
        with patch.object(server.management, 'devices', return_value={'pc': {}}), \
             patch.object(server.management, 'memory_preflight', side_effect=[
                 ValueError('Insufficient currently available GPU memory'), {'status': 'estimated'}]):
            device, model, memory, unloaded, rejected = server.select_model_for_memory(inventory, 8192, 'general')
        self.assertEqual((device, model['key']), ('pc', 'ornith'))
        self.assertEqual(memory['status'], 'estimated')
        self.assertEqual(unloaded, [])
        self.assertEqual(len(rejected), 1)

    def test_smart_ask_preserves_existing_instance(self):
        row = {'key': 'warm', 'type': 'llm', 'loaded_instances': [{'id': 'warm-i'}]}
        snapshot = {'devices': [{'device': 'mac', 'online': True, 'models': [row]}]}
        with patch.object(server, 'pair_devices', return_value=snapshot), patch.object(server.management, 'client') as factory, \
             patch.object(server.management, 'find_model', return_value=row), patch.object(server.management, 'request', return_value={
                 'model': 'warm', 'choices': [{'message': {'content': 'answer'}, 'finish_reason': 'stop'}]}) as req:
            factory.return_value.__enter__.return_value = object()
            result = server.pair_smart_ask('question')
        self.assertEqual(result['cleanup'], 'existing_instance_preserved')
        self.assertEqual([x.args[2] for x in req.call_args_list], ['/v1/chat/completions'])

    @patch.object(server.management, 'memory_preflight', return_value={'status': 'estimated'})
    def test_smart_ask_loads_and_unloads_only_owned_instance(self, _preflight):
        cold = {'key': 'cold', 'type': 'llm', 'loaded_instances': []}
        warm = {'key': 'cold', 'type': 'llm', 'loaded_instances': [{'id': 'owned-i'}]}
        snapshot = {'devices': [{'device': 'mac', 'online': True, 'models': [cold]}]}
        responses = [cold, warm, warm, {'key': 'cold', 'type': 'llm', 'loaded_instances': []}]
        prepared = {'rows': [cold], 'candidate': cold, 'memory': {'status': 'estimated'},
                    'auto_unloaded_instances': []}
        def send(_c, _method, route, _body):
            if route == '/api/v1/models/load':
                return {'instance_id': 'owned-i'}
            if route == '/v1/chat/completions':
                return {'model': 'cold', 'choices': [{'message': {'content': 'answer'}, 'finish_reason': 'stop'}]}
            return {}
        with patch.object(server, 'pair_devices', return_value=snapshot), patch.object(server.management, 'client') as factory, \
             patch.object(server.management, 'find_model', side_effect=responses), \
             patch.object(server.management, 'models', side_effect=[[cold], [warm]]), \
             patch.object(server.management, 'devices', return_value={'mac': {}}), \
             patch.object(server.management, 'preflight_for_load', return_value=prepared), \
             patch.object(server.management, 'request', side_effect=send) as req:
            factory.return_value.__enter__.return_value = object()
            result = server.pair_smart_ask('question')
        self.assertEqual(result['cleanup'], 'unloaded')
        self.assertEqual([x.args[2] for x in req.call_args_list],
                         ['/api/v1/models/load', '/v1/chat/completions', '/api/v1/models/unload'])

    @patch.object(server.management, 'memory_preflight', return_value={'status': 'estimated'})
    def test_timeout_retains_new_instance_for_inspection(self, _preflight):
        cold = {'key': 'cold', 'type': 'llm', 'loaded_instances': []}
        warm = {'key': 'cold', 'type': 'llm', 'loaded_instances': [{'id': 'owned-i'}]}
        snapshot = {'devices': [{'device': 'mac', 'online': True, 'models': [cold]}]}
        prepared = {'rows': [cold], 'candidate': cold, 'memory': {'status': 'estimated'},
                    'auto_unloaded_instances': []}
        def send(_c, _method, route, _body):
            if route == '/api/v1/models/load':
                return {'instance_id': 'owned-i'}
            if route == '/v1/chat/completions':
                raise ValueError('Device timed out')
            return {}
        with patch.object(server, 'pair_devices', return_value=snapshot), patch.object(server.management, 'client') as factory, \
             patch.object(server.management, 'find_model', side_effect=[cold, warm]), \
             patch.object(server.management, 'models', side_effect=[[cold], [warm]]), \
             patch.object(server.management, 'devices', return_value={'mac': {}}), \
             patch.object(server.management, 'preflight_for_load', return_value=prepared), \
             patch.object(server.management, 'request', side_effect=send) as req:
            factory.return_value.__enter__.return_value = object()
            with self.assertRaisesRegex(ValueError, 'timed out'):
                server.pair_smart_ask('question')
        self.assertNotIn('/api/v1/models/unload', [x.args[2] for x in req.call_args_list])

    def test_compare_preserves_both_provenances_and_one_failure(self):
        with patch.object(server, 'pair_smart_ask', side_effect=[{'device': 'mac', 'answer': 'A'}, ValueError('offline')]) as ask:
            result = server.pair_compare('question', 'a', 'mac', 'b', 'pc')
        self.assertEqual(ask.call_count, 2)
        self.assertEqual(result['results'][0]['answer'], 'A')
        self.assertEqual(result['results'][1], {'device': 'pc', 'model': 'b', 'error': 'offline'})

    def test_compare_marks_disagreement_unverified(self):
        first = {'device': 'mac', 'answer': '- Input validation is missing at line 12\n- Cache can race on writes'}
        second = {'device': 'pc', 'answer': '- Input validation is missing at line 12\n- Cache writes use a lock'}
        with patch.object(server, 'pair_smart_ask', side_effect=[first, second]):
            result = server.pair_compare('review', 'a', 'mac', 'b', 'pc')
        self.assertEqual(len(result['shared_observations']), 1)
        self.assertEqual(len(result['disputed_observations']), 2)
        self.assertEqual(result['verification_status'], 'requires_codex_source_review')

    def test_download_requires_repeated_exact_model(self):
        with patch.object(server.management, 'devices', return_value={'mac': {}}), patch.object(server.management, 'client') as factory:
            plan = server.pair_download_plan('mac', 'actual/model', 1000000, '/reviewed/models')
            with self.assertRaisesRegex(ValueError, 'Repeat the exact'):
                server.pair_download(plan['plan_id'], 'other/model')
            factory.assert_not_called()

    def test_download_plan_is_one_use_and_exposes_review_fields(self):
        with patch.object(server.management, 'devices', return_value={'mac': {}}), patch.object(server.management, 'client') as factory, \
             patch.object(server.management, 'request', return_value={'job_id': 'job1', 'status': 'downloading', 'total_size_bytes': 1100000}):
            factory.return_value.__enter__.return_value = object()
            plan = server.pair_download_plan('mac', 'actual/model', 1000000, '/reviewed/models')
            self.assertEqual(plan['estimated_disk_and_network_bytes'], 1000000)
            self.assertEqual(plan['source'], 'LM Studio catalog: actual/model')
            self.assertEqual(plan['destination'], '/reviewed/models')
            self.assertEqual(server.pair_download(plan['plan_id'], 'actual/model')['job_id'], 'job1')
            with self.assertRaisesRegex(ValueError, 'missing or expired'):
                server.pair_download(plan['plan_id'], 'actual/model')

    def test_download_plan_rejects_untrusted_url(self):
        with patch.object(server.management, 'devices', return_value={'mac': {}}):
            with self.assertRaisesRegex(ValueError, 'huggingface.co'):
                server.pair_download_plan('mac', 'https://example.com/model', 100, '/models')

    def test_diagnostics_omits_private_router_url_and_model_names(self):
        rows = [{'key': 'private-model', 'type': 'llm', 'loaded_instances': [{'id': 'secret-instance'}]}]
        with patch.object(server.management, 'devices', return_value={'mac': {}}), \
             patch.object(server.management, 'client') as factory, \
             patch.object(server.management, 'models', return_value=rows), \
             patch.object(server, 'catalog', return_value=[{'id': 'private-model'}]):
            factory.return_value.__enter__.return_value = object()
            result = server.pair_diagnose()
        self.assertEqual(result['devices'][0]['loaded_instances'], 1)
        self.assertNotIn('private-model', str(result))
        self.assertNotIn('secret-instance', str(result))


class JevTests(unittest.TestCase):
    def test_external_send_requires_explicit_opt_in(self):
        with patch.dict(jev.os.environ, {'TYPESAFE_API_KEY': 'secret'}), patch.object(jev.httpx, 'Client') as factory:
            with self.assertRaisesRegex(ValueError, 'allow_external'):
                jev.decide('state', 'choose', {'a': 'A', 'b': 'B'}, allow_external=False)
            factory.assert_not_called()

    def test_choice_response_is_validated(self):
        with patch.dict(jev.os.environ, {'TYPESAFE_API_KEY': 'secret'}), patch.object(jev.httpx, 'Client') as factory:
            response = factory.return_value.__enter__.return_value.post.return_value
            response.is_success = True
            response.json.return_value = {'model': 'jev-latest', 'answers': {'decision': {
                'type': 'choice', 'choice': 'a', 'probabilities': {'a': .8, 'b': .2}, 'confidence': .7}}}
            result = jev.decide('state', 'choose', {'a': 'A', 'b': 'B'}, allow_external=True)
            self.assertEqual(result['choice'], 'a')
            self.assertNotIn('secret', str(result))

    def test_score_response_is_validated(self):
        with patch.dict(jev.os.environ, {'TYPESAFE_API_KEY': 'secret'}), patch.object(jev.httpx, 'Client') as factory:
            response = factory.return_value.__enter__.return_value.post.return_value
            response.is_success = True
            response.json.return_value = {'model': 'jev-latest', 'answers': {'decision': {
                'type': 'score', 'score': 1.25, 'legend': {'0': 'low', '1': 'medium', '2': 'high'},
                'probabilities': {'0': .1, '1': .55, '2': .35}, 'confidence': .7}}}
            result = jev.score('state', 'rate', ['low', 'medium', 'high'], allow_external=True)
            self.assertEqual(result['score'], 1.25)
            self.assertEqual(factory.return_value.__enter__.return_value.post.call_args.kwargs['json']['questions']['decision']['type'], 'score')
            self.assertNotIn('secret', str(result))

    def test_score_rejects_wrong_legend(self):
        with patch.dict(jev.os.environ, {'TYPESAFE_API_KEY': 'secret'}), patch.object(jev.httpx, 'Client') as factory:
            response = factory.return_value.__enter__.return_value.post.return_value
            response.is_success = True
            response.json.return_value = {'answers': {'decision': {'type': 'score', 'score': 0,
                'legend': {'0': 'other', '1': 'high'}, 'probabilities': {'0': 1, '1': 0}, 'confidence': 1}}}
            with self.assertRaisesRegex(ValueError, 'legend'):
                jev.score('state', 'rate', ['low', 'high'], allow_external=True)

class ConfigTests(unittest.TestCase):
    def test_new_config_takes_precedence_over_legacy(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(server.Path, 'home', return_value=Path(tmp)), patch.dict(server.os.environ, {}, clear=True):
            (Path(tmp)/'.codex-pair-bridge.json').write_text('{"base_url":"http://localhost:7000/v1"}')
            (Path(tmp)/'.pair-bridge.json').write_text('{"base_url":"http://localhost:8000/v1"}')
            self.assertEqual(server.load_config(), ('http://localhost:8000/v1', None))

    def test_config_file(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(server.Path, 'home', return_value=Path(tmp)), patch.dict(server.os.environ, {}, clear=True):
            (Path(tmp)/'.codex-pair-bridge.json').write_text('{"base_url":"http://localhost:7777/v1/"}')
            self.assertEqual(server.load_config(), ('http://localhost:7777/v1', None))

    def test_reject_embedded_credentials(self):
        with patch.dict(server.os.environ, {'PAIR_BASE_URL':'https://user:secret@example.com/v1'}):
            with self.assertRaisesRegex(ValueError, 'without credentials'):
                server.load_config()

    def test_env_override(self):
        with patch.dict(server.os.environ, {'PAIR_BASE_URL':'https://example.com/v1','PAIR_API_KEY':'test'}):
            self.assertEqual(server.load_config(), ('https://example.com/v1', 'test'))


class ProtocolTests(unittest.IsolatedAsyncioTestCase):
    async def test_mcp_handshake_and_schema(self):
        import sys
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        params = StdioServerParameters(command=sys.executable, args=[str(Path(server.__file__))])
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = (await session.list_tools()).tools
                self.assertEqual({t.name for t in tools}, {'pair_list', 'pair_ask', 'pair_devices', 'pair_load', 'pair_unload', 'pair_memory_plan',
                                                          'pair_smart_ask', 'pair_compare', 'pair_diagnose',
                                                          'pair_download_plan', 'pair_download', 'pair_download_status',
                                                          'pair_decide', 'pair_score', 'pair_benchmark', 'pair_benchmark_results',
                                                          'pair_job_start', 'pair_job_status', 'pair_job_cancel', 'pair_job_recover'})
                result = await session.call_tool('pair_ask', {'model':'model','prompt':'hello','max_tokens':-1})
                self.assertTrue(result.isError)


if __name__ == '__main__':
    unittest.main()
