"""Fresh stdio MCP process talking to a real loopback HTTP engine fixture."""
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
from test_inference_features import PNG, SCHEMA


class EndToEndFeatureTests(unittest.IsolatedAsyncioTestCase):
    async def test_fresh_mcp_process_http_embeddings_vision_structured_and_batch(self):
        seen = []
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def send_body(self, body, mime='application/json'):
                raw = body.encode() if isinstance(body, str) else json.dumps(body).encode()
                self.send_response(200)
                self.send_header('Content-Type', mime)
                self.send_header('Content-Length', str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self):
                self.send_body({'models': [
                    {'key': 'chat', 'type': 'llm', 'capabilities': {'vision': True},
                     'max_context_length': 8192, 'loaded_instances': [{'id': 'chat-i', 'config': {'context_length': 8192}}]},
                    {'key': 'embed', 'type': 'embedding', 'loaded_instances': [{'id': 'embed-i'}]}]})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                seen.append((self.path, body))
                if self.path == '/v1/embeddings':
                    self.send_body({'data': [{'index': i, 'embedding': [0.25, 0.75]} for i in range(len(body['input']))]})
                elif self.path == '/v1/chat/completions':
                    if body.get('stream'):
                        events = [{'choices': [{'delta': {'content': '{"ok":true}'}, 'finish_reason': None}]},
                                  {'choices': [{'delta': {}, 'finish_reason': 'stop'}]}]
                        self.send_body(''.join('data: ' + json.dumps(row) + '\n\n' for row in events) + 'data: [DONE]\n\n', 'text/event-stream')
                    else:
                        self.send_body({'model': 'chat-i', 'choices': [{'message': {'content': '{"ok":true}'}, 'finish_reason': 'stop'}]})
                elif self.path == '/api/v1/chat':
                    self.send_body('event: message.delta\ndata: {"type":"message.delta","content":"plain answer"}\n\n'
                                   'event: chat.end\ndata: {"type":"chat.end","result":{"output":[{"type":"message","content":"plain answer"}]}}\n\n', 'text/event-stream')
                else:
                    self.send_error(400)

        engine = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=engine.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                config = {'devices': [{'id': 'fixture', 'base_url': f'http://127.0.0.1:{engine.server_port}'}]}
                (root / '.pair-bridge.json').write_text(json.dumps(config))
                env = dict(os.environ, USERPROFILE=tmp, HOME=tmp, LOCALAPPDATA=tmp, APPDATA=tmp)
                params = StdioServerParameters(command=sys.executable, args=[str(Path(server.__file__))], env=env)
                async with stdio_client(params) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        async def call(name, **args):
                            result = await session.call_tool(name, args)
                            self.assertFalse(result.isError, str(result.content))
                            return result.structuredContent or json.loads(result.content[0].text)
                        cap = await call('pair_model_capabilities', device='fixture', model='chat')
                        self.assertEqual(cap['capabilities']['vision']['status'], 'supported')
                        embed = await call('pair_embeddings', device='fixture', model='embed', input=['one', 'two'], expected_dimensions=2)
                        self.assertEqual(embed['vectors'], [[0.25, 0.75], [0.25, 0.75]])
                        ask = await call('pair_ask', device='fixture', model='chat', prompt='PRIVATE ASK', json_schema=SCHEMA)
                        self.assertEqual(ask['structured_output'], {'ok': True})
                        vision = await call('pair_vision_ask', device='fixture', model='chat', prompt='PRIVATE VISION', images=[PNG], json_schema=SCHEMA)
                        self.assertEqual(vision['structured_output'], {'ok': True})
                        batch = await call('pair_batch_start', requests=[
                            {'device': 'fixture', 'model': 'chat', 'prompt': 'PRIVATE JOB', 'json_schema': SCHEMA},
                            {'device': 'fixture', 'model': 'chat', 'prompt': 'PRIVATE PLAIN JOB'}])
                        for _ in range(50):
                            status = await call('pair_batch_status', batch_id=batch['batch_id'])
                            if status['finished']:
                                break
                            await asyncio.sleep(.05)
                        self.assertTrue(status['finished'])
                        self.assertEqual([job['status'] for job in status['jobs']], ['completed', 'completed'])
                        self.assertEqual(status['jobs'][0]['structured_output'], {'ok': True})
                        listing = await call('pair_job_list', include_terminal=True)
                        self.assertNotIn('PRIVATE', json.dumps(listing))
                        await call('pair_batch_cancel', batch_id=batch['batch_id'])
                for journal in root.rglob('*.jsonl'):
                    self.assertNotIn('PRIVATE', journal.read_text())
                self.assertTrue(any(path == '/v1/embeddings' and body['model'] == 'embed-i' for path, body in seen))
                self.assertTrue(any(body.get('response_format') for _, body in seen))
                self.assertFalse(any('load' in path or 'unload' in path for path, _ in seen))
        finally:
            engine.shutdown()
            engine.server_close()
            thread.join(2)


if __name__ == '__main__':
    unittest.main()
