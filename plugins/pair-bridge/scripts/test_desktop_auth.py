import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import httpx
import management


class DesktopAuthTests(unittest.TestCase):
    def test_loopback_only_configuration(self):
        with tempfile.TemporaryDirectory() as temp:
            config = Path(temp) / '.pair-bridge.json'
            row = {'id': 'local', 'engine': 'unsloth', 'base_url': 'http://127.0.0.1:8888',
                   'desktop_secret_file': str(Path(temp) / 'desktop')}
            with patch.object(management.Path, 'home', return_value=Path(temp)):
                config.write_text(json.dumps({'devices': [row]}))
                self.assertEqual(management.devices()['local']['engine'], 'unsloth')
                for override in ({'base_url': 'http://example.com:8888'},
                                 {'engine': 'lmstudio'}, {'ssh_host': 'remote'},
                                 {'api_key_env': 'TEST_API_KEY'}, {'desktop_secret_file': 'relative'}):
                    config.write_text(json.dumps({'devices': [dict(row, **override)]}))
                    with self.assertRaisesRegex(ValueError, 'desktop_secret_file'):
                        management.devices()

    def test_existing_secret_exchange_and_auth_errors(self):
        with tempfile.TemporaryDirectory() as temp:
            secret_file = Path(temp) / 'desktop'
            secret_file.write_text('fake-desktop-secret')
            config = Path(temp) / '.pair-bridge.json'
            config.write_text(json.dumps({'devices': [{'id': 'local', 'engine': 'unsloth',
                'base_url': 'http://127.0.0.1:8888', 'desktop_secret_file': str(secret_file)}]}))
            calls = []
            def handler(req):
                calls.append(req.url.path)
                if req.url.path == '/api/auth/desktop-login':
                    self.assertEqual(json.loads(req.content), {'secret': 'fake-desktop-secret'})
                    return httpx.Response(200, json={'access_token': 'fake-short-lived-token'})
                self.assertEqual(req.headers['Authorization'], 'Bearer fake-short-lived-token')
                return httpx.Response(200, json={'data': []})
            real_client = httpx.Client
            with patch.object(management.Path, 'home', return_value=Path(temp)), \
                 patch.object(management.httpx, 'Client', side_effect=lambda **kwargs: real_client(
                     **kwargs, transport=httpx.MockTransport(handler))):
                for _ in range(2):
                    with management.client('local') as connection:
                        self.assertEqual(management.models(connection, 'local'), [])
                        self.assertFalse(connection.follow_redirects)
            self.assertEqual(calls, ['/api/auth/desktop-login', '/v1/models'] * 2)
            self.assertEqual(secret_file.read_text(), 'fake-desktop-secret')
            with patch.object(management.Path, 'home', return_value=Path(temp)), \
                 patch.object(management.httpx, 'Client', side_effect=lambda **kwargs: real_client(
                     **kwargs, transport=httpx.MockTransport(lambda req: httpx.Response(
                         401, text='fake-desktop-secret')))):
                with self.assertRaisesRegex(ValueError, 'HTTP 401') as error:
                    with management.client('local'):
                        self.fail('Authentication should stop the operation')
                self.assertNotIn('fake-desktop-secret', str(error.exception))

    def test_invalid_credentials_and_tokens_are_sanitized(self):
        with tempfile.TemporaryDirectory() as temp:
            secret=Path(temp)/'desktop'
            config=Path(temp)/'.pair-bridge.json'
            config.write_text(json.dumps({'devices':[{'id':'local','engine':'unsloth','base_url':'http://127.0.0.1:8888','desktop_secret_file':str(secret)}]}))
            real=httpx.Client
            with patch.object(management.Path,'home',return_value=Path(temp)):
                for value in ('', 'fixture-secret'*1000):
                    secret.write_text(value)
                    with self.subTest(credential_length=len(value)), self.assertRaisesRegex(ValueError,'credential'):
                        with management.client('local'):self.fail('Invalid credential accepted')
                secret.unlink()
                with self.assertRaisesRegex(ValueError,'unreadable'):
                    with management.client('local'):self.fail('Missing credential accepted')
                secret.write_text('fixture-secret')
                for token in (None, 123, '', 'bad\r\nheader', 'bad\x00token', 'токен', 'x'*8193):
                    with self.subTest(token_type=type(token).__name__), patch.object(management.httpx,'Client',side_effect=lambda **kw:real(**kw,transport=httpx.MockTransport(lambda req:httpx.Response(200,json={'access_token':token})))):
                        with self.assertRaisesRegex(ValueError,'valid access token') as error:
                            with management.client('local'):self.fail('Invalid token accepted')
                        self.assertNotIn('fixture-secret',str(error.exception))
