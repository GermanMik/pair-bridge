import asyncio
import contextlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
import inference_io as io
import jobs
import server

ADVERTISED = {'components': {'schemas': {'ChatCompletionRequest': {
    'properties': {'enable_thinking': {'type': 'boolean'}}}}}}
ITEM = {'key': 'swift', 'type': 'llm', 'max_context_length': 121344,
        'loaded_instances': [{'id': 'swift', 'config': {'context_length': 121344}}]}


class ThinkingTests(unittest.TestCase):
    def test_control_default_and_guards(self):
        self.assertEqual(io.thinking_options(None, None), {})
        for value in (False, True):
            self.assertEqual(io.thinking_options(value, 'unsloth', ADVERTISED), {'enable_thinking': value})
        for value in (0, 1, 'false', {}):
            with self.assertRaises(ValueError): io.thinking_options(value, 'unsloth', ADVERTISED)
        for engine, schema in [('lmstudio', ADVERTISED), (None, None), ('unsloth', {}), ('unsloth', None)]:
            with self.assertRaises(ValueError): io.thinking_options(False, engine, schema)

    def test_sync_transport_exact_false_and_no_inference_when_unsupported(self):
        with tempfile.TemporaryDirectory() as temp, \
             patch.object(server, 'user_cache_path', return_value=Path(temp)), \
             patch.object(server.management, 'devices', return_value={'unsloth': {'engine': 'unsloth'}}), \
             patch.object(server.telemetry, 'sample', return_value=None), \
             patch.object(server.management, 'engine_for', return_value='unsloth'), \
             patch.object(server.management, 'client', return_value=contextlib.nullcontext(object())), \
             patch.object(server.management, 'find_model', return_value=ITEM), \
             patch.object(server.management, 'chat_model_id', return_value='swift'):
            for value in (False, True, None):
                calls = []
                def request(c, method, path, payload=None):
                    calls.append((method, path, payload))
                    return ADVERTISED if path == '/openapi.json' else {'choices': [
                        {'message': {'content': 'final'}, 'finish_reason': 'stop'}]}
                with patch.object(server.management, 'request', side_effect=request):
                    result = server.pair_ask('swift', 'bounded request', 40000, 'unsloth', enable_thinking=value)
                self.assertEqual(result['answer'], 'final')
                payload = calls[-1][2]
                if value is None:
                    self.assertNotIn('enable_thinking', payload)
                    self.assertEqual(len(calls), 1)
                else:
                    self.assertIs(payload['enable_thinking'], value)
                    self.assertEqual(len(calls), 2)
            with patch.object(server.management, 'request', return_value={}) as request:
                with self.assertRaisesRegex(ValueError, 'advertise'):
                    server.pair_ask('swift', 'bounded', device='unsloth', enable_thinking=False)
                self.assertEqual(request.call_args.args[1:3], ('GET', '/openapi.json'))
                self.assertEqual(request.call_count, 1)

    def test_stream_preserves_final_and_records_reasoning_only_length(self):
        async def run():
            for final in (True, False):
                payloads = []
                def handler(request):
                    payloads.append(json.loads(request.content))
                    events = [{'choices': [{'delta': {'reasoning_content': 'private'}}]}]
                    if final: events.append({'choices': [{'delta': {'content': 'final'}}]})
                    events.append({'choices': [{'delta': {}, 'finish_reason': 'stop' if final else 'length'}]})
                    body = ''.join('data: '+json.dumps(e)+'\n\n' for e in events)+'data: [DONE]\n\n'
                    return httpx.Response(200, text=body)
                with tempfile.TemporaryDirectory() as temp, patch.object(jobs, 'user_cache_path', return_value=Path(temp)):
                    job = jobs.Job('unsloth', 'swift')
                    with patch.object(server.management, 'engine_for', return_value='unsloth'):
                        async with httpx.AsyncClient(base_url='http://fixture', transport=httpx.MockTransport(handler)) as c:
                            if final:
                                answer = await server._job_stream(job, c, 'swift', 'bounded', 40000, enable_thinking=False)
                                self.assertEqual(answer, 'final')
                            else:
                                with self.assertRaisesRegex(ValueError, 'no final text'):
                                    await server._job_stream(job, c, 'swift', 'bounded', 40000, enable_thinking=False)
                                self.assertEqual(job.finish_reason, 'length')
                    self.assertIs(payloads[0]['enable_thinking'], False)
                    self.assertEqual(len(payloads), 1)
                    self.assertNotIn('private', job.answer)
        asyncio.run(run())

    def test_budget_confirmation_boundaries(self):
        for requested in (32, 6144, 39999):
            row = io.output_budget(ITEM, requested)['increase_request']
            self.assertEqual(row['status'], 'confirmation_required')
            self.assertEqual(row['proposed_max_tokens'], 40000)
        for requested in (40000, 40001):
            self.assertEqual(io.output_budget(ITEM, requested)['increase_request']['status'], 'not_needed')
        row = io.output_budget({'max_context_length': 8192}, 2048)['increase_request']
        self.assertEqual(row['status'], 'limit_below_target')
        self.assertIsNone(row['proposed_max_tokens'])
        row = io.output_budget({}, 2048)['increase_request']
        self.assertFalse(row['ceiling_verified'])
        self.assertIsNone(row['known_ceiling'])

    def test_context_confirmation_is_separate_from_generation(self):
        for value in (8192, 39999, 40000, 121344):
            item = dict(ITEM, loaded_instances=[{'id': 'swift', 'config': {'context_length': value}}])
            row = io.output_budget(item, 32)['context_increase_request']
            self.assertEqual(row['requires_confirmation'], value < 40000)
            self.assertEqual(row['reload_required'], value < 40000)
            self.assertEqual(row['current_context_length'], value)
        row = io.output_budget({'max_context_length': 262144}, 64, 8192)['context_increase_request']
        self.assertEqual(row['proposed_context_length'], 40000)
        self.assertFalse(row['reload_required'])
        row = io.output_budget({'max_context_length': 8192}, 64, 8192)['context_increase_request']
        self.assertEqual(row['status'], 'model_limit_below_target')
        self.assertIsNone(row['proposed_context_length'])
        row = io.output_budget({'max_context_length': 121344}, 6144)['context_increase_request']
        self.assertEqual(row['status'], 'unknown')

    def test_capabilities_reports_small_context_even_if_request_exceeds_ceiling(self):
        item = dict(ITEM, loaded_instances=[{'id': 'swift', 'config': {'context_length': 8192}}])
        with patch.object(server.management, 'client', return_value=contextlib.nullcontext(object())), \
             patch.object(server.management, 'find_model', return_value=item), \
             patch.object(server.management, 'engine_for', return_value='unsloth'):
            row = server.pair_model_capabilities('unsloth', 'swift', max_tokens=40000)['output_budget']
        self.assertFalse(row['request_within_known_ceiling'])
        self.assertEqual(row['context_increase_request']['proposed_context_length'], 40000)
        self.assertEqual(row['increase_request']['status'], 'limit_below_target')

    def test_public_schema_and_job_batch_forwarding(self):
        async def schema():
            tools = {t.name: t.inputSchema for t in await server.mcp.list_tools()}
            for name in ('pair_ask', 'pair_smart_ask', 'pair_job_start'):
                self.assertIn('enable_thinking', tools[name]['properties'])
            self.assertIn('enable_thinking', tools['pair_batch_start']['$defs']['BatchRequest']['properties'])
        asyncio.run(schema())
        with patch.object(server.management, 'devices', return_value={'unsloth': {}}), \
             patch.object(server.management, 'engine_for', return_value='unsloth'), \
             patch.object(jobs, 'create') as create:
            server.pair_job_start('unsloth', 'swift', 'bounded', max_tokens=40000, enable_thinking=False)
            self.assertIs(create.call_args.args[-1], False)
        with tempfile.TemporaryDirectory() as temp, patch.object(jobs, 'user_cache_path', return_value=Path(temp)), \
             patch.object(jobs, '_jobs', {}), patch.object(jobs, '_batches', {}), patch.object(jobs, '_start') as start:
            jobs.create_batch([{'device': 'unsloth', 'model': 'swift', 'prompt': 'bounded',
                'context_length': 8192, 'max_tokens': 40000, 'unload_after': False, 'enable_thinking': False}], object())
            self.assertIs(start.call_args.args[2][-1], False)

    def test_smart_rejects_control_before_selection_and_load(self):
        with patch.object(server, 'pair_devices') as inventory, \
             patch.object(server.management, 'client', return_value=contextlib.nullcontext(object())), \
             patch.object(server.management, 'engine_for', return_value='unsloth'), \
             patch.object(server.management, 'request', return_value={}):
            with self.assertRaisesRegex(ValueError, 'advertise'):
                server.pair_smart_ask('bounded', device='unsloth', enable_thinking=False)
            inventory.assert_not_called()
            with self.assertRaisesRegex(ValueError, 'configured Unsloth'):
                server.pair_smart_ask('bounded', enable_thinking=False)
            inventory.assert_not_called()
        for value in (0, 1, 'false'):
            with self.assertRaises(ValueError):
                server.BatchRequest(device='unsloth', model='swift', prompt='p', enable_thinking=value)
