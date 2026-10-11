"""Real loopback slow-response regression: no user engine/model/config touched."""
import json,subprocess,sys,tempfile,threading,time,unittest
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from unittest.mock import patch
import httpx,management,server,diagnostics

class LoadDeadlineTests(unittest.TestCase):
    def test_continuous_slow_headers_and_body_do_not_extend_total_deadline(self):
        for mode in ('headers','body'):
            with self.subTest(mode=mode):
                observed=[]
                class Handler(BaseHTTPRequestHandler):
                    def log_message(self,*args):pass
                    def do_POST(self):
                        body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                        observed.append((self.path,self.headers.get('Authorization'),body))
                        try:
                            if mode=='headers':
                                self.connection.sendall(b'HTTP/1.0 200 OK\r\nX-Padding: ')
                                for _ in range(30):self.connection.sendall(b'x');time.sleep(.03)
                                self.connection.sendall(b'\r\nContent-Length: 2\r\n\r\n{}')
                            else:
                                self.send_response(200);self.send_header('Content-Length','32');self.end_headers()
                                for _ in range(30):self.wfile.write(b' ');self.wfile.flush();time.sleep(.03)
                                self.wfile.write(b'{}');self.wfile.flush()
                        except (BrokenPipeError,ConnectionResetError,OSError):pass
                engine=ThreadingHTTPServer(('127.0.0.1',0),Handler);engine.daemon_threads=True
                thread=threading.Thread(target=engine.serve_forever,daemon=True);thread.start()
                try:
                    with tempfile.TemporaryDirectory() as tmp, patch.object(server,'user_cache_path',return_value=Path(tmp)), patch.object(management,'devices',return_value={'fixture':{'engine':'unsloth','load_timeout_seconds':.15}}):
                        with httpx.Client(base_url=f'http://127.0.0.1:{engine.server_port}',headers={'Authorization':'Bearer fixture-token'},trust_env=False) as client:
                            start=time.monotonic()
                            with self.assertRaisesRegex(ValueError,'total deadline') as error:
                                with server.inference_lock('fixture'):management.load_model(client,'fixture','fixture/model',8192)
                            elapsed=time.monotonic()-start
                            self.assertLess(elapsed,1.5);self.assertEqual(diagnostics.reason(error.exception),'load_timeout')
                            self.assertNotIn('fixture-token',str(error.exception))
                        code='import sys;from filelock import FileLock\nwith FileLock(sys.argv[1],timeout=0):pass'
                        result=subprocess.run([sys.executable,'-c',code,str(Path(tmp)/'device-fixture.lock')],capture_output=True,timeout=5)
                        self.assertEqual(result.returncode,0,result.stderr)
                    self.assertEqual(len(observed),1,'No automatic retry')
                    self.assertEqual(observed[0][0],'/api/inference/load');self.assertEqual(observed[0][1],'Bearer fixture-token')
                finally:engine.shutdown();engine.server_close();thread.join(2)

    def test_public_timeout_configuration_is_validated(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(management.Path,'home',return_value=Path(tmp)):
            p=Path(tmp)/'.pair-bridge.json'
            for value in (0,-1,True,.1,3601,'180'):
                p.write_text(json.dumps({'devices':[{'id':'fixture','engine':'unsloth','base_url':'http://127.0.0.1:8888','load_timeout_seconds':value}]}))
                with self.subTest(value=value),self.assertRaisesRegex(ValueError,'load_timeout_seconds'):management.devices()
            for value in (1,180,3600):
                p.write_text(json.dumps({'devices':[{'id':'fixture','engine':'lmstudio','base_url':'http://127.0.0.1:1234','load_timeout_seconds':value}]}))
                self.assertEqual(management.devices()['fixture']['load_timeout_seconds'],value)

    def test_both_load_routes_use_deadline_but_unload_does_not(self):
        for engine,route in [('unsloth','/api/inference/load'),('lmstudio','/api/v1/models/load')]:
            with patch.object(management,'devices',return_value={'fixture':{'engine':engine,'load_timeout_seconds':321}}),patch.object(management,'request',return_value={}) as request:
                management.load_model(object(),'fixture','fixture/model',8192)
                self.assertEqual(request.call_args.args[2],route);self.assertEqual(request.call_args.kwargs,{'total_timeout':321})

    def test_disabled_download_calls_never_contact_engines_or_repository(self):
        with patch.object(management,'client') as client,patch.object(server.download_review,'model_files') as repository,patch.object(management,'request') as request:
            for fn,args in [(server.pair_download_plan,('fixture','model')),(server.pair_download,('old-plan','model')),(server.pair_download_status,('fixture','old-job'))]:
                with self.assertRaisesRegex(ValueError,'downloads are disabled'):fn(*args)
            client.assert_not_called();repository.assert_not_called();request.assert_not_called()
        self.assertEqual(diagnostics.reason(ValueError('Another Codex PAIR request is running on this target.')),'device_busy')

    def test_bounded_response_keeps_validation_and_refuses_redirects(self):
        real_async=httpx.AsyncClient
        cases=[(200,b'{"status":"loaded"}',None),(302,b'fixture-secret','HTTP 302'),(401,b'fixture-secret','HTTP 401'),(200,b'fixture-secret','invalid JSON'),(200,b'x'*(1024*1024+1),'size limit')]
        with httpx.Client(base_url='http://127.0.0.1:1',headers={'Authorization':'Bearer fixture-token'}) as client:
            for status,body,error in cases:
                calls=[]
                def reply(req):
                    calls.append(req.url.path);self.assertEqual(req.headers['Authorization'],'Bearer fixture-token')
                    return httpx.Response(status,content=body,headers={'Location':'http://127.0.0.1:2/other'})
                def factory(**kw):
                    self.assertFalse(kw['follow_redirects']);self.assertFalse(kw['trust_env'])
                    return real_async(**kw,transport=httpx.MockTransport(reply))
                with self.subTest(status=status,size=len(body)),patch.object(management.httpx,'AsyncClient',side_effect=factory):
                    if error:
                        with self.assertRaisesRegex(ValueError,error) as exc:management.request(client,'POST','/api/inference/load',{},total_timeout=1)
                        self.assertNotIn('fixture-secret',str(exc.exception))
                    else:self.assertEqual(management.request(client,'POST','/api/inference/load',{},total_timeout=1),{'status':'loaded'})
                self.assertEqual(calls,['/api/inference/load'])

if __name__=='__main__':unittest.main()
