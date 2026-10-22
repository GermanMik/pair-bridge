"""Real stdio MCP and HTTP fixture: no installed model or user cache is touched."""
import asyncio
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import benchmark_suites
import server


class BenchmarkProtocolTests(unittest.IsolatedAsyncioTestCase):
    async def test_run_query_compare_and_restart_over_mcp(self):
        cases = benchmark_suites.suite('short')['cases']
        expected = {case['prompt']: (json.dumps(case['expected']) if isinstance(case['expected'], dict)
                                     else case['expected']) for case in cases}
        seen = []
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def respond(self, body, mime='application/json'):
                raw = body.encode() if isinstance(body, str) else json.dumps(body).encode()
                self.send_response(200)
                self.send_header('Content-Type', mime)
                self.send_header('Content-Length', str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self):
                seen.append((self.path, None))
                self.respond({'models': [{'key': 'model', 'type': 'llm', 'max_context_length': 8192,
                                          'loaded_instances': [{'id': 'exact-i', 'config': {'context_length': 8192}}]}]})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                seen.append((self.path, body['model']))
                if self.path != '/v1/chat/completions' or body['model'] != 'exact-i':
                    self.send_error(400)
                    return
                answer = expected[body['messages'][0]['content']]
                events = [{'choices': [{'delta': {'reasoning_content': 'PRIVATE_REASONING'}}]},
                          {'choices': [{'delta': {'content': answer}, 'finish_reason': None}]},
                          {'choices': [{'delta': {}, 'finish_reason': 'stop'}]},
                          {'choices': [], 'usage': {'completion_tokens': 8}}]
                self.respond(''.join('data: ' + json.dumps(e) + '\n\n' for e in events) +
                             'data: [DONE]\n\n', 'text/event-stream')

        engine = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=engine.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / '.pair-bridge.json').write_text(json.dumps({'devices': [
                    {'id': 'fixture', 'base_url': f'http://127.0.0.1:{engine.server_port}'}]}))
                # Force a test-local platformdirs root, including child processes on Windows.
                bootstrap = ("import sys,runpy,pathlib,platformdirs; "
                             "sys.path.insert(0,sys.argv[1]); "
                             "platformdirs.user_cache_path=lambda *a,**k:pathlib.Path(sys.argv[2]); "
                             "runpy.run_path(str(pathlib.Path(sys.argv[1])/'server.py'),run_name='__main__')")
                params = StdioServerParameters(command=sys.executable,
                                               args=['-c', bootstrap, str(Path(server.__file__).parent), tmp],
                                               env=dict(os.environ, USERPROFILE=tmp, HOME=tmp))
                async def call(session, name, **args):
                    response = await session.call_tool(name, args)
                    self.assertFalse(response.isError, str(response.content))
                    return response.structuredContent or json.loads(response.content[0].text)
                run_ids = []
                async with stdio_client(params) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        catalog = await call(session, 'pair_benchmark_suites')
                        self.assertEqual([s['case_count'] for s in catalog['suites']], [6, 20, 100])
                        self.assertEqual((await call(session, 'pair_benchmark_results'))['status'], 'not_measured')
                        for _ in range(2):
                            run = await call(session, 'pair_benchmark_run', device='fixture', model='model',
                                             suite='short', repeats=1, max_tokens=256)
                            run_ids.append(run['run_id'])
                            deadline = time.monotonic() + 30
                            while time.monotonic() < deadline:
                                status = await call(session, 'pair_job_status', job_id=run['job_id'])
                                if status['status'] in ('completed', 'failed', 'cancelled'):
                                    break
                                await asyncio.sleep(.02)
                            self.assertEqual(status['status'], 'completed', status)
                            self.assertEqual(status['structured_output']['quality']['passed'], 20)
                            rows = await call(session, 'pair_benchmark_results', run_id=run['run_id'])
                            result = rows['results'][0]
                            self.assertEqual(result['quality']['pass_rate'], 1)
                            self.assertEqual(result['instance_id'], 'exact-i')
                            self.assertEqual(result['speed']['median_tokens_per_second']['samples'], 20)
                        comparison = await call(session, 'pair_benchmark_compare', run_ids=run_ids)
                        self.assertIn('| model / fixture | short / 1 | 20/20 |', comparison['table_markdown'])
                async with stdio_client(params) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        rows = await call(session, 'pair_benchmark_results', run_id=run_ids[0])
                        self.assertEqual(rows['results'][0]['status'], 'completed')
                self.assertEqual(sum(path == '/v1/chat/completions' for path, _ in seen), 42)
                self.assertFalse(any('load' in path or 'unload' in path for path, _ in seen))
                db = (root / 'benchmark-runs.sqlite3').read_bytes()
                self.assertNotIn(b'PRIVATE_REASONING', db)
                self.assertNotIn(cases[0]['prompt'].encode(), db)
                self.assertNotIn(b'"prompt":', db)
                self.assertNotIn(b'"answer":', db)
        finally:
            engine.shutdown()
            engine.server_close()
            thread.join(2)


if __name__ == '__main__':
    unittest.main()
