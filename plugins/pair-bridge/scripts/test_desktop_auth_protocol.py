"""Authenticated sync/async MCP fixture: no user credential or model is touched."""
import asyncio,json,os,sys,tempfile,threading,unittest
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from mcp import ClientSession,StdioServerParameters
from mcp.client.stdio import stdio_client
import server


class DesktopAuthProtocolTests(unittest.IsolatedAsyncioTestCase):
    async def test_desktop_token_reaches_async_job_and_is_not_journaled(self):
        seen=[]
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def respond(self,body,mime='application/json'):
                data=(json.dumps(body) if isinstance(body,dict) else body).encode()
                self.send_response(200);self.send_header('Content-Type',mime)
                self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
            def authorized(self):return self.headers.get('Authorization')=='Bearer fixture-short-lived-token'
            def do_GET(self):
                if self.path!='/v1/models' or not self.authorized():self.send_error(401);return
                self.respond({'data':[{'id':'fixture/chat','loaded':True,'type':'llm','context_length':32768}]})
            def do_POST(self):
                body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                if self.path=='/api/auth/desktop-login':
                    if body!={'secret':'fixture-desktop-secret'}:self.send_error(401);return
                    self.respond({'access_token':'fixture-short-lived-token'});return
                if self.path!='/v1/chat/completions' or not self.authorized():self.send_error(401);return
                seen.append({'authorized':True,'max_tokens':body['max_tokens']})
                self.respond('data: {"choices":[{"delta":{"content":"fixture answer"},"finish_reason":null}]}\n\n'
                             'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n','text/event-stream')
        engine=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        thread=threading.Thread(target=engine.serve_forever,daemon=True);thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);secret=root/'desktop';secret.write_text('fixture-desktop-secret')
                original={'devices':[{'id':'fixture','engine':'unsloth','base_url':f'http://127.0.0.1:{engine.server_port}','desktop_secret_file':str(secret)}]}
                config=root/'.pair-bridge.json';config.write_text(json.dumps(original))
                env=dict(os.environ,HOME=tmp,USERPROFILE=tmp,LOCALAPPDATA=tmp,APPDATA=tmp,PAIR_CLUSTER_DIR=str(root/'cluster'))
                params=StdioServerParameters(command=sys.executable,args=[str(Path(server.__file__))],env=env)
                with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as errlog:
                    async with stdio_client(params, errlog=errlog) as (read,write):
                        async with ClientSession(read,write) as session:
                            await session.initialize()
                            async def call(tool,**args):
                                result=await session.call_tool(tool,args)
                                self.assertFalse(result.isError,str(result.content))
                                return result.structuredContent or json.loads(next(x.text for x in result.content if x.type=='text'))
                            start=await call('pair_job_start',device='fixture',model='fixture/chat',prompt='fixture request',max_tokens=16384)
                            for _ in range(80):
                                result=await call('pair_job_status',job_id=start['job_id'])
                                if result['status'] in ('completed','failed','cancelled'):break
                                await asyncio.sleep(.05)
                            self.assertEqual(result['status'],'completed')
                    errlog.seek(0)
                    stderr=errlog.read()
                    self.assertNotIn('Task was destroyed',stderr)
                    self.assertNotIn('was never awaited',stderr)
                    self.assertNotIn('fixture-desktop-secret',stderr)
                    self.assertNotIn('fixture-short-lived-token',stderr)
                self.assertEqual(seen,[{'authorized':True,'max_tokens':16384}])
                self.assertEqual(json.loads(config.read_text()),original)
                for p in root.rglob('*.jsonl'):
                    self.assertNotIn('fixture-short-lived-token',p.read_text())
                    self.assertNotIn('fixture-desktop-secret',p.read_text())
        finally:engine.shutdown();engine.server_close();thread.join(2)


if __name__=='__main__':unittest.main()
