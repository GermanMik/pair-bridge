"""Recovery contracts: cause evidence, consent, cancellation, live capacity and privacy."""
import asyncio
import contextlib
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
import diagnostics
import inference_io
import jobs
import recovery
import server


ITEM = {'key': 'model', 'type': 'llm', 'max_context_length': 120000,
        'loaded_instances': [{'id': 'instance', 'config': {'context_length': 100000}}]}
SCHEMA = {'components': {'schemas': {'ChatCompletionRequest': {
    'properties': {'enable_thinking': {'type': 'boolean'}}}}}}


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        for module, name, value in [(diagnostics, 'journal_path', Path(temp.name) / 'requests.jsonl'),
                                     (jobs, 'journal_path', Path(temp.name) / 'jobs.jsonl'),
                                     (recovery, 'user_cache_path', Path(temp.name))]:
            p = patch.object(module, name, return_value=value)
            p.start()
            self.addCleanup(p.stop)

    def test_empty_length_preserves_usage_without_reasoning_text(self):
        data = {'choices': [{'finish_reason': 'length', 'message': {
            'content': None, 'reasoning_content': 'PRIVATE REASONING'}}],
            'usage': {'prompt_tokens': 100, 'completion_tokens': 6144,
                      'completion_tokens_details': {'reasoning_tokens': 6144}, 'secret': 'SECRET'}}
        with self.assertRaises(recovery.InferenceFailure) as raised:
            server.completion(data, 'model', 'pc', time.monotonic())
        row = raised.exception.metadata
        self.assertEqual(raised.exception.code, 'empty_answer')
        self.assertEqual(row['finish_reason'], 'length')
        self.assertEqual(row['usage']['reasoning_tokens'], 6144)
        self.assertEqual(row['cause'], 'budget_exhausted')
        self.assertTrue(row['reasoning_present'])
        self.assertNotIn('PRIVATE', json.dumps(row))
        self.assertNotIn('SECRET', json.dumps(row))

    def test_tools_refusals_and_invalid_choices_are_not_empty(self):
        for reason, message, expected in [
            ('tool_calls', {'content': None, 'tool_calls': [{'function': 'secret'}]}, 'tool_response'),
            ('stop', {'content': None, 'refusal': 'private refusal'}, 'refusal'),
            ('content_filter', {'content': ''}, 'refusal')]:
            with self.subTest(expected=expected), self.assertRaises(recovery.InferenceFailure) as error:
                server.completion({'choices': [{'finish_reason': reason, 'message': message}]}, 'm', 'pc', time.monotonic())
            self.assertEqual(error.exception.code, expected)
        with self.assertRaises(recovery.InferenceFailure) as error:
            server.completion({'choices': []}, 'm', None, time.monotonic())
        self.assertEqual(error.exception.code, 'invalid_response')

    def test_public_sync_empty_returns_job_without_second_inference(self):
        empty = {'choices': [{'message': {'content': ''}, 'finish_reason': 'stop'}]}
        with patch.object(server, 'catalog', return_value=[{'id': 'model', 'kind_hint': 'chat_candidate'}]), \
                patch.object(server, 'request', return_value=empty) as request, \
                patch.object(jobs, 'create', return_value={'job_id': 'tracked'}) as create:
            result = server.pair_ask('model', 'private prompt', max_tokens=40000)
        self.assertEqual(result['job_id'], 'tracked')
        self.assertEqual(result['status'], 'recovering')
        self.assertEqual(request.call_count, 1)
        self.assertEqual(create.call_count, 1)

    def test_budget_waits_for_change_then_recovers_without_new_load(self):
        async def run():
            job = jobs.Job('pc', 'model')
            job.update(status='running', instance_id='instance', recovery_parameters={'max_tokens': 6144})
            calls = 0
            async def send():
                nonlocal calls
                calls += 1
                if calls == 1:
                    raise recovery.InferenceFailure('empty_answer', recovery.metadata('length', {'completion_tokens': 6144}))
                job.answer = 'final'
                job.finish_reason = 'stop'
                return 'final'
            task = asyncio.create_task(server._retry_job_empty_answer(job, send))
            for _ in range(100):
                if job.stage == 'waiting_for_configuration': break
                await asyncio.sleep(.005)
            self.assertEqual(job.stage, 'waiting_for_configuration')
            self.assertEqual(calls, 1)
            with patch.dict(jobs._jobs, {job.id: job}), \
                    patch.object(server.management, 'client', return_value=contextlib.nullcontext(object())), \
                    patch.object(server.management, 'find_model', return_value=ITEM):
                server.pair_job_reconfigure(job.id, max_tokens=40000)
            self.assertEqual(await asyncio.wait_for(task, 2), 'final')
            self.assertEqual(calls, 2)
            self.assertEqual(job.empty_answer_retries, 1)
            self.assertEqual(recovery.load_profile('pc', 'model'), {'max_tokens': 40000})
        asyncio.run(run())

    def test_waiting_configuration_cancel_does_not_resubmit(self):
        async def run():
            job = jobs.Job('pc', 'model')
            job.recovery_wait_revision = 0
            job.update(status='running')
            calls = []
            async def send(): calls.append(1)
            with patch.dict(jobs._jobs, {job.id: job}):
                task = asyncio.create_task(server._retry_job_empty_answer(job, send))
                await asyncio.sleep(.01)
                jobs.cancel(job.id)
                with self.assertRaises(asyncio.CancelledError):
                    await asyncio.wait_for(task, 1)
            self.assertEqual(calls, [])
        asyncio.run(run())

    def test_measured_usage_exhaustion_waits_without_inventing_finish_reason(self):
        async def run():
            job = jobs.Job('pc', 'model')
            job.recovery_parameters = {'max_tokens': 6144}
            calls = []
            async def send():
                calls.append(1)
                raise recovery.InferenceFailure('empty_answer', recovery.metadata('unknown', {'completion_tokens': 6144}))
            task = asyncio.create_task(server._retry_job_empty_answer(job, send))
            for _ in range(100):
                if job.stage == 'waiting_for_configuration': break
                await asyncio.sleep(.005)
            self.assertEqual(job.stage, 'waiting_for_configuration')
            self.assertEqual(job.attempt_metadata['finish_reason'], 'unknown')
            self.assertEqual(job.attempt_metadata['cause'], 'budget_exhausted')
            task.cancel()
            with self.assertRaises(asyncio.CancelledError): await task
            self.assertEqual(calls, [1])
        asyncio.run(run())

    def test_fresh_capacity_shortage_waits_then_warns_and_sends(self):
        async def run():
            job = jobs.Job('pc', 'model')
            job.recovery_initial_preflight = True
            checks = 0
            async def send():
                self.assertEqual(checks, 2)
                return 'final'
            def before():
                nonlocal checks
                checks += 1
                if checks == 1:
                    raise ValueError('Insufficient available system memory to preserve the 1 GiB free reserve')
                self.assertEqual(job.stage, 'waiting_for_resources')
                return {'status': 'warning', 'warnings': ['Unknown VRAM']}
            self.assertEqual(await server._retry_job_empty_answer(job, send, before), 'final')
            self.assertEqual(job.memory_preflight['status'], 'warning')
        asyncio.run(run())

    def test_retry_checks_original_instance_and_unknown_inventory_warns(self):
        with self.assertRaises(recovery.InferenceFailure):
            server._recovery_preflight(None, None, 'model', None)
        with patch.object(server.management, 'find_model', return_value=ITEM), \
                patch.object(server.management, 'devices', return_value={'pc': {}}), \
                patch.object(server.telemetry, 'sample', return_value={}):
            with self.assertRaises(recovery.InferenceFailure):
                server._recovery_preflight(object(), 'pc', 'model', 'changed')
            result = server._recovery_preflight(object(), 'pc', 'model', 'instance')
            self.assertEqual(result['status'], 'warning')
            with patch.object(server.management, 'find_model', side_effect=ValueError('unreachable')):
                result = server._recovery_preflight(object(), 'pc', 'model', 'instance')
            self.assertEqual(result['status'], 'warning')
            self.assertIn('unconfirmed', result['warnings'][0])

    def test_reconfigure_rejects_unchanged_unsupported_and_prompt_overflow(self):
        job = jobs.Job('pc', 'model')
        job.update(status='running', stage='waiting_for_configuration', instance_id='instance',
                   recovery_parameters={'max_tokens': 6144},
                   attempt_metadata={'usage': {'prompt_tokens': 90000}})
        with patch.dict(jobs._jobs, {job.id: job}), \
                patch.object(server.management, 'client', return_value=contextlib.nullcontext(object())), \
                patch.object(server.management, 'find_model', return_value=ITEM), \
                patch.object(server.management, 'engine_for', return_value='lmstudio'):
            with self.assertRaisesRegex(ValueError, 'unchanged'):
                server.pair_job_reconfigure(job.id, max_tokens=6144)
            with self.assertRaisesRegex(ValueError, 'Prompt plus'):
                server.pair_job_reconfigure(job.id, max_tokens=40000)
            with self.assertRaises(ValueError):
                server.pair_job_reconfigure(job.id, enable_thinking=False)
        self.assertEqual(job.recovery_revision, 0)

    def test_attempt_logs_export_safe_metadata_only(self):
        diagnostics.record_attempt('job', 'pc', 'model', {
            'attempt': 21, 'finish_reason': 'length', 'usage': {'prompt_tokens': 20, 'secret': 'SECRET'},
            'reasoning_present': True, 'reasoning_content': 'PRIVATE', 'prompt': 'PRIVATE'}, 21)
        rows, _ = diagnostics.recent_outcomes()
        self.assertEqual(rows[-1]['attempt_metadata']['attempt'], 21)
        target = diagnostics.journal_path().with_name('export.jsonl')
        diagnostics.export_outcomes(str(target))
        text = target.read_text()
        self.assertNotIn('SECRET', text)
        self.assertNotIn('PRIVATE', text)

    def test_stream_usage_only_chunks_and_native_finish_are_retained(self):
        async def run():
            job = jobs.Job('pc', 'model')
            chunks = [{'choices': [{'delta': {'reasoning_content': 'PRIVATE'}}]},
                      {'choices': [{'delta': {'content': [{'type': 'text', 'text': 'final'}]}, 'finish_reason': 'stop'}]},
                      {'choices': [], 'usage': {'prompt_tokens': 10, 'completion_tokens': 5}}]
            body = ''.join('data: '+json.dumps(x)+'\n\n' for x in chunks)+'data: [DONE]\n\n'
            with patch.object(server.management, 'engine_for', return_value='unsloth'):
                async with httpx.AsyncClient(base_url='http://fixture', transport=httpx.MockTransport(
                        lambda request: httpx.Response(200, text=body))) as c:
                    self.assertEqual(await server._job_stream(job, c, 'instance', 'prompt', 40000), 'final')
            self.assertEqual(job.attempt_metadata['usage']['prompt_tokens'], 10)
            self.assertTrue(job.attempt_metadata['reasoning_present'])
            native = 'event: chat.end\ndata: '+json.dumps({'result': {'finish_reason': 'length',
                'output': [{'type': 'message', 'content': 'partial'}]}})+'\n\n'
            with patch.object(server.management, 'engine_for', return_value='lmstudio'):
                async with httpx.AsyncClient(base_url='http://fixture', transport=httpx.MockTransport(
                        lambda request: httpx.Response(200, text=native))) as c:
                    await server._job_stream(job, c, 'instance', 'prompt', 40000)
            self.assertEqual(job.finish_reason, 'length')
        asyncio.run(run())

    def test_runtime_fingerprint_and_prompt_budget_are_honest(self):
        with patch.object(server.management, 'devices', return_value={}):
            row = server.pair_capabilities()
        self.assertEqual(row['bridge_version'], '0.8.4')
        self.assertEqual(len(row['source_build']), 16)
        self.assertIsNone(row['empty_answer_recovery']['attempt_limit'])
        self.assertIsNone(inference_io.prompt_budget(ITEM, 40000)['request_fits'])
        self.assertFalse(inference_io.prompt_budget(ITEM, 40000, 90000)['request_fits'])

    def test_native_terminal_errors_never_enter_empty_recovery(self):
        async def run():
            for result, code in [
                ({'output': 'invalid'}, 'invalid_response'),
                ({}, 'invalid_response'),
                ({'output': [None]}, 'invalid_response'),
                ({'output': [{'type': 'message', 'content': 5}]}, 'invalid_response'),
                ({'output': [{'type': 'tool_call', 'tool': 'PRIVATE_TOOL'}]}, 'tool_response'),
                ({'output': [{'type': 'invalid_tool_call'}]}, 'tool_response'),
                ({'output': [{'type': 'refusal', 'content': 'PRIVATE'}]}, 'refusal')]:
                with self.subTest(result=result):
                    calls = []
                    body = 'event: chat.end\ndata: '+json.dumps({'result': result})+'\n\n'
                    def handle(request):
                        calls.append(1)
                        return httpx.Response(200, text=body)
                    job = jobs.Job('pc', 'model')
                    with patch.object(server.management, 'engine_for', return_value='lmstudio'):
                        async with httpx.AsyncClient(base_url='http://fixture', transport=httpx.MockTransport(handle)) as c:
                            with self.assertRaises(recovery.InferenceFailure) as raised:
                                await asyncio.wait_for(server._retry_job_empty_answer(job,
                                    lambda: server._job_stream(job, c, 'instance', 'prompt', 40000)), .5)
                    self.assertEqual(raised.exception.code, code)
                    self.assertEqual(calls, [1])
                    self.assertEqual(job.empty_answer_retries, 0)
        asyncio.run(run())

    def test_malformed_sync_content_never_hands_off_or_resubmits(self):
        for content in (7, True, {}, [None], ['text'], [{'type': 'text', 'text': 5}],
                        [{'type': 'unknown', 'text': 'PRIVATE'}]):
            with self.subTest(content=content), \
                    patch.object(server, 'catalog', return_value=[{'id': 'model', 'kind_hint': 'chat_candidate'}]), \
                    patch.object(server, 'request', return_value={'choices': [{
                        'message': {'content': content}, 'finish_reason': 'stop'}],
                        'usage': {'prompt_tokens': 10, 'secret': 'PRIVATE'}}) as request, \
                    patch.object(jobs, 'create') as create:
                with self.assertRaises(recovery.InferenceFailure) as error:
                    server.pair_ask('model', 'PRIVATE_PROMPT')
                self.assertEqual(error.exception.code, 'invalid_response')
                self.assertEqual(error.exception.metadata['usage'], {'prompt_tokens': 10})
                self.assertEqual(request.call_count, 1)
                create.assert_not_called()

    def test_malformed_stream_content_is_terminal_on_first_attempt(self):
        async def run():
            for content in (7, {}, [{'type': 'text', 'text': 5}], [None]):
                calls = []
                body = 'data: '+json.dumps({'choices': [{'delta': {'content': content},
                    'finish_reason': 'stop'}]})+'\n\ndata: [DONE]\n\n'
                def handler(request):
                    calls.append(1)
                    return httpx.Response(200, text=body)
                job = jobs.Job('pc', 'model')
                with patch.object(server.management, 'engine_for', return_value='unsloth'):
                    async with httpx.AsyncClient(base_url='http://fixture', transport=httpx.MockTransport(handler)) as c:
                        with self.assertRaises(recovery.InferenceFailure) as error:
                            await asyncio.wait_for(server._retry_job_empty_answer(job,
                                lambda: server._job_stream(job, c, 'instance', 'prompt', 40000)), .5)
                self.assertEqual(error.exception.code, 'invalid_response')
                self.assertEqual(calls, [1])
                self.assertEqual(job.empty_answer_retries, 0)
        asyncio.run(run())

    def test_reasoning_content_parts_are_never_returned_as_final_text(self):
        data = {'choices': [{'finish_reason': 'stop', 'message': {'content': [
            {'type': 'reasoning', 'text': 'PRIVATE_REASONING'}, {'type': 'text', 'text': 'final'}]}}]}
        row = server.completion(data, 'model', 'pc', time.monotonic())
        self.assertEqual(row['answer'], 'final')
        self.assertTrue(row['attempt_metadata']['reasoning_present'])
        self.assertNotIn('PRIVATE', json.dumps(row))


if __name__ == '__main__': unittest.main()
