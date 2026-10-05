import unittest
from unittest.mock import patch
import contextlib
import json
import subprocess
import tempfile
from pathlib import Path
import server
import management

class ManagementTests(unittest.TestCase):
    def scope(self, *_args, **_kwargs):
        return contextlib.nullcontext(object())

    def test_no_download_for_missing_model(self):
        with patch.object(management, 'models', return_value=[]), patch.object(management, 'request') as req:
            with self.assertRaisesRegex(ValueError, 'not installed'):
                management.find_model(object(), 'imaginary')
            req.assert_not_called()

    def test_auto_unload_config_requires_unique_exact_model_keys(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / '.pair-bridge.json'
            device = {'id': 'pc', 'engine': 'unsloth', 'base_url': 'http://127.0.0.1:8888',
                      'auto_unload_models': ['publisher/model-a']}
            path.write_text(json.dumps({'devices': [device]}))
            with patch.object(management.Path, 'home', return_value=Path(temp)):
                self.assertEqual(management.devices()['pc']['auto_unload_models'], ['publisher/model-a'])
                device['auto_unload_models'] = ['publisher/model-a', 'publisher/model-a']
                path.write_text(json.dumps({'devices': [device]}))
                with self.assertRaisesRegex(ValueError, 'unique exact model keys'):
                    management.devices()

    def test_unsloth_device_configuration_and_inventory_normalization(self):
        with patch.object(management, 'devices', return_value={
            'local': {'engine': 'unsloth'}
        }), patch.object(management, 'request', return_value={'data': [
            {'id': 'publisher/chat-model', 'loaded': True, 'context_length': 8192,
             'max_context_length': 32768, 'display_name': 'Chat Model'},
            {'id': 'publisher/embedding-model', 'loaded': False, 'task': 'embedding'},
        ]}) as request:
            rows = management.models(object(), 'local')
        self.assertEqual(request.call_args.args[2], '/v1/models')
        self.assertEqual(rows[0]['type'], 'llm')
        self.assertEqual(rows[0]['loaded_instances'][0]['config']['context_length'], 8192)
        self.assertEqual(rows[0]['max_context_length'], 32768)
        self.assertEqual(rows[1]['type'], 'embedding')
        self.assertEqual(rows[1]['loaded_instances'], [])

    def test_unsloth_inventory_requires_explicit_loaded_state(self):
        with patch.object(management, 'devices', return_value={'local': {'engine': 'unsloth'}}), \
             patch.object(management, 'request', return_value={'data': [{'id': 'model'}]}):
            with self.assertRaisesRegex(ValueError, 'cannot determine loaded state'):
                management.models(object(), 'local')

    def test_load_http_out_of_memory_is_classified_without_leaking_body(self):
        class Response:
            is_success = False
            status_code = 500
            text = 'CUDA out of memory; private model path'
        class Client:
            def request(self, *_args, **_kwargs):
                return Response()
        with self.assertRaisesRegex(ValueError, 'insufficient available memory') as error:
            management.request(Client(), 'POST', '/api/inference/load', {'model_path': 'model'})
        self.assertNotIn('private model path', str(error.exception))

    def test_unsloth_estimate_uses_device_api_and_maps_complete_result(self):
        with patch.object(management, 'devices', return_value={'local': {'engine': 'unsloth'}}), \
             patch.object(management, 'client', return_value=self.scope()), \
             patch.object(management.shutil, 'which', return_value=None), \
             patch.object(management, 'request', return_value={
                 'available': True, 'total_bytes': 21_000, 'gpu_bytes': 19_000,
                 'kv_estimable': True,
             }) as request:
            result = management.estimate_memory('local', 'publisher/model', 8192)
        self.assertEqual(request.call_args.args[2], '/api/inference/estimate-memory')
        self.assertEqual(request.call_args.args[3], {
            'model_path': 'publisher/model', 'n_ctx': 8192, 'max_seq_length': 8192,
        })
        self.assertEqual(result['gpu_bytes'], 19_000)
        self.assertEqual(result['source'], 'Unsloth Studio /api/inference/estimate-memory')

    def test_unsloth_load_and_unload_use_management_routes(self):
        connection = object()
        with patch.object(management, 'engine_for', return_value='unsloth'), \
             patch.object(management, 'request', return_value={}) as request:
            management.load_model(connection, 'local', 'publisher/model', 8192)
            self.assertEqual(request.call_args.args[2], '/api/inference/load')
            self.assertEqual(request.call_args.args[3], {
                'model_path': 'publisher/model', 'n_ctx': 8192,
                'max_seq_length': 8192, 'load_in_4bit': True,
            })
            management.unload_model(connection, 'local', 'publisher/model', 'instance')
            self.assertEqual(request.call_args.args[2], '/api/inference/unload')
            self.assertEqual(request.call_args.args[3], {'model_path': 'publisher/model'})
            self.assertEqual(management.chat_model_id('local', 'publisher/model', 'instance'),
                             'publisher/model')

    def test_unsloth_load_failure_releases_only_allowlisted_exact_instance_then_retries(self):
        cold = {'key': 'candidate', 'loaded_instances': []}
        occupied = {'key': 'permitted', 'loaded_instances': [{'id': 'permitted'}]}
        released = {'key': 'permitted', 'loaded_instances': []}
        connection = object()
        with patch.object(management, 'models', side_effect=[
            [cold, occupied], [cold, released]
        ]), patch.object(management, 'load_model', side_effect=[
            ValueError('Device reports insufficient available memory'), {'status': 'loaded'}
        ]) as load, patch.object(management, 'unload_model') as unload:
            result = management.load_with_auto_unload(
                connection, 'local', 'candidate', 8192, 'llm',
                {'auto_unload_models': ['permitted']})
        self.assertEqual(result['result'], {'status': 'loaded'})
        self.assertEqual(result['auto_unloaded_instances'], [
            {'model': 'permitted', 'instance_id': 'permitted'}
        ])
        unload.assert_called_once_with(connection, 'local', 'permitted', 'permitted')
        self.assertEqual(load.call_count, 2)

    def test_weight_limit_preserves_other_loaded_models(self):
        loaded = {'key': 'other', 'size_bytes': 80, 'loaded_instances': [{'id': 'other-i'}]}
        candidate = {'key': 'new', 'size_bytes': 40, 'loaded_instances': []}
        with self.assertRaisesRegex(ValueError, 'limit would be exceeded'):
            management.ensure_capacity([loaded, candidate], candidate, 100)
        management.ensure_capacity([loaded, candidate], candidate, 120)

    def test_memory_preflight_uses_loaded_context_and_blocks_cap(self):
        loaded = {'key': 'running', 'loaded_instances': [{'id': 'i', 'config': {'context_length': 65536}}]}
        cold = {'key': 'new', 'loaded_instances': []}
        with patch.object(management, 'estimate_memory', side_effect=[
            {'total_bytes': 20, 'gpu_bytes': 20, 'context_length': 8192},
            {'total_bytes': 30, 'gpu_bytes': 30, 'context_length': 65536}]) as estimate:
            result = management.memory_preflight('pc', [loaded, cold], cold, 8192, 60)
        self.assertEqual(result['estimated_total_bytes'], 50)
        self.assertEqual(estimate.call_args_list[1].args, ('pc', 'running', 65536))
        with patch.object(management, 'estimate_memory', side_effect=[
            {'total_bytes': 20, 'gpu_bytes': 20}, {'total_bytes': 30, 'gpu_bytes': 30}]):
            with self.assertRaisesRegex(ValueError, 'estimated memory limit'):
                management.memory_preflight('pc', [loaded, cold], cold, 8192, 49)

    def test_memory_preflight_fails_closed_when_cap_and_context_unknown(self):
        loaded = {'key': 'running', 'loaded_instances': [{'id': 'i'}]}
        cold = {'key': 'new', 'loaded_instances': []}
        with patch.object(management, 'estimate_memory', return_value={'total_bytes': 20, 'gpu_bytes': 20}):
            with self.assertRaisesRegex(ValueError, 'context is unknown'):
                management.memory_preflight('pc', [loaded, cold], cold, 8192, 100)

    def test_unknown_unsloth_estimate_is_reported_and_configured_cap_blocks(self):
        cold = {'key': 'new', 'loaded_instances': []}
        with patch.object(management, 'estimate_memory', side_effect=ValueError(
                'Unsloth Studio cannot estimate this model at the planned context (not_gguf)')):
            result = management.memory_preflight('local', [cold], cold, 8192, None)
            self.assertEqual(result['status'], 'unknown')
            with self.assertRaisesRegex(ValueError, 'cannot estimate'):
                management.memory_preflight('local', [cold], cold, 8192, 100)

    def test_preflight_unloads_only_allowlisted_exact_instances_and_rechecks(self):
        other = {'key': 'permitted', 'loaded_instances': [{'id': 'permitted-i'}]}
        cold = {'key': 'candidate', 'loaded_instances': []}
        after = {'key': 'candidate', 'loaded_instances': []}
        with patch.object(management, 'models', side_effect=[[other, cold], [other, cold], [after]]), \
             patch.object(management, 'memory_preflight', side_effect=[
                 ValueError('Insufficient currently available GPU memory'), {'status': 'estimated'}]) as preflight, \
             patch.object(management, 'request', return_value={}) as request, \
             patch.object(management, 'engine_for', return_value='lmstudio'), \
             patch.object(management, 'estimate_memory', return_value={}) as estimate:
            result = management.preflight_for_load(
                object(), 'pc', 'candidate', 8192, {'auto_unload_models': ['permitted']},
                capacity={'checked_at': 1}, capacity_sampler=lambda: {'checked_at': 2})
        self.assertEqual(result['auto_unloaded_instances'], [{'model': 'permitted', 'instance_id': 'permitted-i'}])
        self.assertEqual(request.call_args.args[3], {'instance_id': 'permitted-i'})
        self.assertEqual(preflight.call_count, 2)
        self.assertEqual(estimate.call_count, 0)

    def test_preflight_without_unload_setting_preserves_loaded_model(self):
        other = {'key': 'other', 'loaded_instances': [{'id': 'other-i'}]}
        cold = {'key': 'candidate', 'loaded_instances': []}
        with patch.object(management, 'models', return_value=[other, cold]), \
             patch.object(management, 'memory_preflight', side_effect=ValueError('Insufficient currently available GPU memory')), \
             patch.object(management, 'request') as request:
            with self.assertRaisesRegex(ValueError, 'Insufficient currently available'):
                management.preflight_for_load(object(), 'pc', 'candidate', 8192, {})
            request.assert_not_called()

    def test_estimator_accepts_cli_stderr_and_target_port(self):
        output = 'Estimated GPU Memory: 19.24 GiB\nEstimated Total Memory: 20.00 GiB\n'
        with patch.object(management, 'devices', return_value={'pc': {'base_url': 'http://127.0.0.1:1234'}}), \
             patch.object(management.shutil, 'which', return_value='/tmp/lms'), \
             patch.object(management.Path, 'is_file', return_value=True), \
             patch.object(management.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, '', output)) as run:
            result = management.estimate_memory('pc', 'installed-model', 8192)
        self.assertEqual(result['total_bytes'], 20 * 2**30)
        self.assertIn('1234', run.call_args.args[0])

    def test_reuse_loaded_without_mutation(self):
        model={'key':'m','loaded_instances':[{'id':'instance'}]}
        with patch.object(server, 'inference_lock', self.scope), patch.object(management, 'client', return_value=self.scope()), patch.object(management, 'models', return_value=[model]), patch.object(management, 'request') as req:
            self.assertEqual(server.pair_load('pc','m')['status'], 'already_loaded')
            req.assert_not_called()

    def test_unload_exact_instance_and_verify(self):
        before=[{'key':'m','loaded_instances':[{'id':'instance'}]}]
        with patch.object(server, 'inference_lock', self.scope), patch.object(management, 'client', return_value=self.scope()), patch.object(management, 'models', side_effect=[before, []]), patch.object(management, 'devices', return_value={'pc': {}}), patch.object(management, 'request', return_value={}) as req:
            self.assertEqual(server.pair_unload('pc','instance')['status'], 'unloaded')
            self.assertEqual(req.call_args.args[3], {'instance_id':'instance'})

    def test_unload_unknown_rejected(self):
        with patch.object(server, 'inference_lock', self.scope), patch.object(management, 'client', return_value=self.scope()), patch.object(management, 'models', return_value=[]), patch.object(management, 'request') as req:
            with self.assertRaisesRegex(ValueError, 'not loaded'):
                server.pair_unload('pc','missing')
            req.assert_not_called()

    @patch.object(management, 'preflight_for_load')
    def test_load_not_confirmed(self, preflight):
        m={'key':'m','type':'llm','loaded_instances':[],'max_context_length':8192}
        preflight.return_value={'memory': {'status':'estimated'}, 'candidate':m, 'auto_unloaded_instances':[]}
        with patch.object(server, 'inference_lock', self.scope), patch.object(management, 'client', return_value=self.scope()), patch.object(management, 'models', return_value=[m]), patch.object(management, 'devices', return_value={'pc': {}}), patch.object(management, 'find_model', return_value=m), patch.object(management, 'request', return_value={}):
            self.assertEqual(server.pair_load('pc','m')['status'],'not_confirmed')

    @patch.object(management, 'preflight_for_load')
    def test_load_reports_engine_eviction_without_unloading_itself(self, preflight):
        other = {'key': 'other', 'type': 'llm', 'size_bytes': 10, 'loaded_instances': [{'id': 'other-i'}]}
        cold = {'key': 'm', 'type': 'llm', 'size_bytes': 20, 'loaded_instances': [], 'max_context_length': 8192}
        warm = dict(cold, loaded_instances=[{'id': 'new-i'}])
        preflight.return_value={'memory': {'status':'estimated'}, 'candidate':cold,
                                'auto_unloaded_instances':[{'model':'other','instance_id':'other-i'}]}
        with patch.object(server, 'inference_lock', self.scope), patch.object(management, 'client', return_value=self.scope()), \
             patch.object(management, 'models', side_effect=[[other, cold], [warm]]), \
             patch.object(management, 'devices', return_value={'pc': {}}), \
             patch.object(management, 'find_model', return_value=warm), \
             patch.object(management, 'request', return_value={'instance_id': 'new-i', 'load_time_seconds': 1.5}) as req:
            result = server.pair_load('pc', 'm')
        self.assertEqual(result['engine_evicted_instances'], [])
        self.assertEqual(result['auto_unloaded_instances'], [{'model':'other','instance_id':'other-i'}])
        self.assertEqual(result['load_time_seconds'], 1.5)
        self.assertEqual(req.call_args.args[2], '/api/v1/models/load')

    def test_device_ask_uses_instance_not_router(self):
        m={'key':'m','type':'llm','loaded_instances':[{'id':'actual-instance'}]}
        with patch.object(server, 'inference_lock', self.scope), patch.object(management, 'client', return_value=self.scope()), patch.object(management, 'devices', return_value={'pc': {}}), patch.object(management, 'find_model', return_value=m), patch.object(management, 'request', return_value={'choices':[{'message':{'content':'ok'}}]}) as req, patch.object(server,'request') as router:
            r=server.pair_ask('m','test',device='pc')
            self.assertEqual(r['device'],'pc')
            self.assertEqual(req.call_args.args[3]['model'],'actual-instance')
            router.assert_not_called()

    def test_offline_device_does_not_hide_online(self):
        with patch.object(management,'devices',return_value={'offline':{},'online':{}}), patch.object(management,'client',side_effect=[ValueError('unreachable'),self.scope()]), patch.object(management,'models',return_value=[]):
            rows=server.pair_devices()['devices']
            self.assertFalse(rows[0]['online'])
            self.assertTrue(rows[1]['online'])

    def test_malformed_loaded_state_rejected(self):
        with patch.object(management,'request',return_value={'models':[{'key':'m'}]}):
            with self.assertRaisesRegex(ValueError,'loaded state'):
                management.models(object())
