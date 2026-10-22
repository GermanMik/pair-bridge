"""Fresh MCP -> real loopback HTTP -> tracked recovery, explicit change and cancellation."""
import asyncio
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
import tempfile
import threading
import unittest

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import server


class RecoveryProtocolTests(unittest.IsolatedAsyncioTestCase):
    async def test_handoff_reconfigure_export_and_cancel_over_fresh_mcp(self):
        seen = []
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args): pass
            def send_json(self, data):
                raw = json.dumps(data).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
            def do_GET(self):
                if self.path == '/openapi.json':
                    self.send_json({'components': {'schemas': {'ChatCompletionRequest': {
                        'properties': {'enable_thinking': {'type': 'boolean'}}}}}})
                elif self.path == '/v1/models':
                    self.send_json({'data': [{'id': 'model', 'type': 'llm', 'loaded': True,
                        'context_length': 100000, 'max_context_length': 120000}]})
                else: self.send_error(404)
            def do_POST(self):
                if self.path != '/v1/chat/completions':
                    self.send_error(400)
                    return
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                seen.append(body)
                ready = body.get('enable_thinking') is False and body['max_tokens'] == 40000
                self.send_json({'model': 'model', 'choices': [{'message': {
                    'content': '{"ok":true}' if ready else None,
                    'reasoning_content': None if ready else 'PRIVATE_REASONING'},
                    'finish_reason': 'stop' if ready else 'length'}],
                    'usage': {'prompt_tokens': 50, 'completion_tokens': 10 if ready else body['max_tokens']}})
        engine = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=engine.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / '.pair-bridge.json').write_text(json.dumps({'base_url': f'http://127.0.0.1:{engine.server_port}/v1', 'devices': [{
                    'id': 'fixture', 'engine': 'unsloth', 'base_url': f'http://127.0.0.1:{engine.server_port}'}]}))
                bootstrap = (
                    f'import sys,time;sys.path.insert(0,{str(Path(server.__file__).parent)!r});import server;'
                    'server.telemetry.sample=lambda _: {"checked_at":time.time(),'
                    '"memory":{"available_bytes":2147483648},'
                    '"gpu":{"status":"ok","devices":[{"free_bytes":2147483648}]}};'
                    'server.mcp.run(transport="stdio")')
                params = StdioServerParameters(command=sys.executable, args=['-c', bootstrap],
                    env=dict(os.environ, USERPROFILE=tmp, HOME=tmp, LOCALAPPDATA=tmp, APPDATA=tmp))
                async with stdio_client(params) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        async def call(name, **args):
                            r = await session.call_tool(name, args)
                            self.assertFalse(r.isError, str(r.content))
                            return r.structuredContent or json.loads(r.content[0].text)
                        cap = await call('pair_capabilities')
                        self.assertEqual(cap['source_build'], server.SOURCE_BUILD)
                        schema = {'type': 'object', 'properties': {'ok': {'type': 'boolean'}}, 'required': ['ok']}
                        pending = await call('pair_ask', model='model',
                                             prompt='PRIVATE_PROMPT', max_tokens=6144, json_schema=schema)
                        self.assertEqual(pending['status'], 'recovering')
                        async def poll(job_id, stages):
                            for _ in range(100):
                                status = await call('pair_job_status', job_id=job_id)
                                if status['stage'] in stages: return status
                                await asyncio.sleep(.02)
                            self.fail('Recovery did not reach expected stage: '+str(status))
                        waiting = await poll(pending['job_id'], {'waiting_for_device'})
                        self.assertEqual(waiting['attempt_metadata']['finish_reason'], 'length')
                        self.assertEqual(len(seen), 1)
                        await call('pair_job_reconfigure', job_id=pending['job_id'], max_tokens=40000,
                                   enable_thinking=False, device='fixture')
                        final = await poll(pending['job_id'], {'completed', 'failed'})
                        self.assertEqual(final['status'], 'completed')
                        self.assertEqual(final['structured_output'], {'ok': True})
                        self.assertEqual(final['empty_answer_retries'], 1)
                        self.assertEqual(len(seen), 2)
                        self.assertEqual(seen[0]['messages'], seen[1]['messages'])
                        export = root / 'support.jsonl'
                        await call('pair_logs_export', output_path=str(export))
                        rows = [json.loads(line) for line in export.read_text().splitlines()]
                        self.assertTrue(any(r.get('attempt_metadata', {}).get('finish_reason') == 'length' for r in rows))
                        pending_cancel = await call('pair_ask', model='model', device='fixture',
                                                    prompt='PRIVATE_CANCEL', max_tokens=6144)
                        await poll(pending_cancel['job_id'], {'waiting_for_configuration'})
                        await call('pair_job_cancel', job_id=pending_cancel['job_id'])
                        cancelled = await poll(pending_cancel['job_id'], {'cancelled'})
                        self.assertEqual(cancelled['status'], 'cancelled')
                        self.assertEqual(len(seen), 3)
                # A fresh process can inspect persisted metadata but never replays the prompt.
                async with stdio_client(params) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        recovered = await session.call_tool('pair_job_status', {'job_id': pending['job_id']})
                        state = recovered.structuredContent or json.loads(recovered.content[0].text)
                        self.assertEqual(state['status'], 'completed')
                        self.assertEqual(state['empty_answer_retries'], 1)
                        self.assertEqual(len(seen), 3)
                for path in root.rglob('*.jsonl'):
                    self.assertNotIn('PRIVATE', path.read_text())
        finally:
            engine.shutdown()
            engine.server_close()
            thread.join(timeout=2)


if __name__ == '__main__': unittest.main()
