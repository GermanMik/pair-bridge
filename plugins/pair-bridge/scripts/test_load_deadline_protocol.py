"""Cold load through actual FastMCP sync-tool event loop; isolated HTTP fixture."""
import asyncio,json,os,sys,tempfile,threading,time,unittest
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from mcp import ClientSession,StdioServerParameters
from mcp.client.stdio import stdio_client
import server

class LoadDeadlineProtocolTests(unittest.IsolatedAsyncioTestCase):
    async def test_cold_load_deadline_through_fresh_mcp_releases_device(self):
        requests=[]
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_GET(self):
                data=json.dumps({'data':[{'id':'fixture/chat','loaded':False,'type':'llm'}]}).encode()
                self.send_response(200);self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
            def do_POST(self):
                body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                requests.append(self.path)
                if self.path!='/api/inference/load':self.send_error(400);return
                try:
                    self.connection.sendall(b'HTTP/1.0 200 OK\r\nX-Padding: ')
                    for _ in range(40):self.connection.sendall(b'x');time.sleep(.05)
                except OSError:pass
        engine=ThreadingHTTPServer(('127.0.0.1',0),Handler);engine.daemon_threads=True
        thread=threading.Thread(target=engine.serve_forever,daemon=True);thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp,tempfile.TemporaryFile(mode='w+',encoding='utf-8') as stderr:
                root=Path(tmp);config={'devices':[{'id':'fixture','engine':'unsloth','base_url':f'http://127.0.0.1:{engine.server_port}','load_timeout_seconds':1}]}
                (root/'.pair-bridge.json').write_text(json.dumps(config))
                env=dict(os.environ,HOME=tmp,USERPROFILE=tmp,APPDATA=tmp,LOCALAPPDATA=tmp,PAIR_CLUSTER_DIR=str(root/'cluster'))
                params=StdioServerParameters(command=sys.executable,args=[str(Path(server.__file__))],env=env)
                async with stdio_client(params,errlog=stderr) as (read,write):
                    async with ClientSession(read,write) as session:
                        await session.initialize();start=time.monotonic()
                        result=await session.call_tool('pair_load',{'device':'fixture','model':'fixture/chat','context_length':8192})
                        self.assertTrue(result.isError);self.assertIn('total deadline',' '.join(x.text for x in result.content if x.type=='text'))
                        self.assertLess(time.monotonic()-start,3)
                        self.assertNotIn('asyncio.run() cannot be called',' '.join(x.text for x in result.content if x.type=='text'))
                        # Read inventory after timeout; no retry or unload performed.
                        after=await session.call_tool('pair_list',{'device':'fixture'});self.assertFalse(after.isError)
                stderr.seek(0);logs=stderr.read();self.assertNotIn('was never awaited',logs)
                self.assertEqual(requests,['/api/inference/estimate-memory','/api/inference/load'])
                self.assertEqual(json.loads((root/'.pair-bridge.json').read_text()),config)
        finally:engine.shutdown();engine.server_close();thread.join(2)

if __name__=='__main__':unittest.main()
