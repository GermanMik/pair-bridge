# /// script
# requires-python = ">=3.11,<3.15"
# dependencies = ["httpx==0.28.1"]
# ///
"""OpenAI-compatible gateway exposing PAIR and configured local models to Hermes."""
from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import httpx
import management


def configured_pair_url() -> str:
    if os.environ.get('PAIR_BASE_URL'):
        return os.environ['PAIR_BASE_URL'].rstrip('/')
    for name in ('.pair-bridge.json', '.codex-pair-bridge.json'):
        path = Path.home() / name
        if path.exists():
            data = json.loads(path.read_text())
            if not isinstance(data, dict):
                raise ValueError('PAIR config must be a JSON object')
            return str(data.get('base_url', 'http://127.0.0.1:1234/v1')).rstrip('/')
    return 'http://127.0.0.1:1234/v1'


PAIR_URL = configured_pair_url()
PAIR_KEY = os.environ.get('PAIR_API_KEY')
GATEWAY_KEY = os.environ.get('HERMES_PAIR_API_KEY')
HOST = os.environ.get('HERMES_PAIR_HOST', '127.0.0.1')
PORT = int(os.environ.get('HERMES_PAIR_PORT', '8765'))
PREFIX = 'device/'


def chat_candidate(model_id: str) -> bool:
    name = model_id.lower()
    return not any(term in name for term in (
        'embed', 'embedding', 'rerank', 'dflash', 'draft', 'text-to-image',
        'image-generation', 'qwen-image',
    ))


def pair_request(method: str, route: str, body: dict | None = None) -> dict:
    headers = {'Authorization': 'Bearer ' + PAIR_KEY} if PAIR_KEY else {}
    with httpx.Client(timeout=180, trust_env=False, follow_redirects=False, headers=headers) as client:
        response = client.request(method, PAIR_URL + route, json=body)
    if not response.is_success:
        raise ValueError(f'PAIR returned HTTP {response.status_code}')
    result = response.json()
    if not isinstance(result, dict):
        raise ValueError('PAIR returned invalid JSON')
    return result


def models() -> list[dict]:
    result = []
    # Preserve PAIR router IDs for normal routed inference.
    try:
        data = pair_request('GET', '/models').get('data', [])
        for item in data:
            if (isinstance(item, dict) and isinstance(item.get('id'), str)
                    and chat_candidate(item['id'])):
                result.append({'id': item['id'], 'object': 'model', 'owned_by': 'PAIR'})
    except (ValueError, httpx.HTTPError):
        pass
    # Device-qualified IDs avoid collisions and permit exact host selection.
    for device in management.devices():
        try:
            with management.client(device) as client:
                engine = management.engine_for(device)
                for item in management.models(client, device):
                    if (item.get('type') == 'llm' and isinstance(item.get('key'), str)
                            and chat_candidate(item['key'])
                            and (engine != 'unsloth' or item['loaded_instances'])):
                        result.append({'id': PREFIX + device + '/' + item['key'],
                                       'object': 'model', 'owned_by': 'PAIR device ' + device})
        except (ValueError, httpx.HTTPError):
            continue
    # Router models take precedence if an upstream happens to share a device ID.
    unique = {item['id']: item for item in result}
    return list(unique.values())


class Handler(BaseHTTPRequestHandler):
    server_version = 'PAIR-Hermes-Gateway/1.0'

    def log_message(self, fmt, *args):
        # Avoid logging request bodies or prompt data.
        sys.stderr.write('%s %s\n' % (self.log_date_time_string(), fmt % args))

    def send_json(self, status: int, data: dict):
        raw = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def send_completion(self, data: dict, requested_model: str, stream: bool):
        data = dict(data)
        data['model'] = requested_model
        if not stream:
            return self.send_json(200, data)
        choices = data.get('choices')
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            return self.send_json(502, {'error': {'message': 'Upstream returned no completion choices', 'type': 'upstream_error'}})
        choice = choices[0]
        message = choice.get('message') or {}
        if not isinstance(message, dict):
            message = {}
        created = data.get('created')
        completion_id = data.get('id', 'chatcmpl-pair-bridge')
        chunks = [
            {'id': completion_id, 'object': 'chat.completion.chunk', 'created': created,
             'model': requested_model, 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]}
        ]
        delta = {}
        content = message.get('content')
        if isinstance(content, str) and content:
            delta['content'] = content
        reasoning = message.get('reasoning_content')
        if isinstance(reasoning, str) and reasoning:
            delta['reasoning_content'] = reasoning
        if delta:
            chunks.append({'id': completion_id, 'object': 'chat.completion.chunk', 'created': created,
                           'model': requested_model, 'choices': [{'index': 0, 'delta': delta, 'finish_reason': None}]})
        chunks.append({'id': completion_id, 'object': 'chat.completion.chunk', 'created': created,
                       'model': requested_model, 'choices': [{'index': 0, 'delta': {},
                       'finish_reason': choice.get('finish_reason') or 'stop'}]})
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream; charset=utf-8')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Connection', 'close')
        self.end_headers()
        for chunk in chunks:
            self.wfile.write(b'data: ' + json.dumps(chunk, ensure_ascii=False).encode() + b'\n\n')
        if isinstance(data.get('usage'), dict):
            usage = {'id': completion_id, 'object': 'chat.completion.chunk', 'created': created,
                     'model': requested_model, 'choices': [], 'usage': data['usage']}
            self.wfile.write(b'data: ' + json.dumps(usage, ensure_ascii=False).encode() + b'\n\n')
        self.wfile.write(b'data: [DONE]\n\n')

    def authorized(self) -> bool:
        return not GATEWAY_KEY or self.headers.get('Authorization') == 'Bearer ' + GATEWAY_KEY

    def do_GET(self):
        if not self.authorized():
            return self.send_json(401, {'error': {'message': 'Unauthorized', 'type': 'authentication_error'}})
        if urlsplit(self.path).path != '/v1/models':
            return self.send_json(404, {'error': {'message': 'Not found', 'type': 'not_found'}})
        try:
            self.send_json(200, {'object': 'list', 'data': models()})
        except Exception:
            self.send_json(502, {'error': {'message': 'Could not read model catalog', 'type': 'upstream_error'}})

    def do_POST(self):
        if not self.authorized():
            return self.send_json(401, {'error': {'message': 'Unauthorized', 'type': 'authentication_error'}})
        if urlsplit(self.path).path != '/v1/chat/completions':
            return self.send_json(404, {'error': {'message': 'Not found', 'type': 'not_found'}})
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 2_000_000:
                raise ValueError('Invalid request size')
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict) or not isinstance(body.get('model'), str):
                raise ValueError('A model ID is required')
            model = body['model']
            wants_stream = body.get('stream') is True
            # The upstream call is non-streaming; stream its completed result to OpenAI clients.
            body['stream'] = False
            if model.startswith(PREFIX):
                rest = model[len(PREFIX):]
                device, sep, model_id = rest.partition('/')
                if not sep or device not in management.devices() or not model_id:
                    raise ValueError('Use an exact model ID returned by /v1/models')
                with management.client(device) as client:
                    if management.engine_for(device) == 'unsloth':
                        selected = management.find_model(client, model_id, device)
                        if len(selected['loaded_instances']) != 1:
                            raise ValueError('Load this Unsloth model through PAIR Bridge first so memory preflight can run')
                        body['model'] = management.chat_model_id(
                            device, model_id, selected['loaded_instances'][0]['id'])
                    else:
                        # LM Studio may reuse or JIT-load an installed model.
                        body['model'] = model_id
                    upstream = management.request(client, 'POST', '/v1/chat/completions', body)
                return self.send_completion(upstream, model, wants_stream)
            # The request body and model ID pass through to PAIR router unchanged.
            data = pair_request('POST', '/chat/completions', body)
            return self.send_completion(data, model, wants_stream)
        except (ValueError, httpx.HTTPError) as exc:
            return self.send_json(400 if isinstance(exc, ValueError) else 502,
                                  {'error': {'message': str(exc), 'type': 'upstream_error'}})
        except Exception:
            return self.send_json(502, {'error': {'message': 'Upstream request failed', 'type': 'upstream_error'}})


if __name__ == '__main__':
    parts = urlsplit(PAIR_URL)
    if parts.scheme not in ('http', 'https') or not parts.hostname or parts.username or parts.password:
        raise SystemExit('PAIR_BASE_URL must be an HTTP(S) URL')
    if HOST not in ('127.0.0.1', 'localhost', '::1') and not GATEWAY_KEY:
        raise SystemExit('Set HERMES_PAIR_API_KEY before binding beyond loopback')
    print(f'PAIR Hermes gateway listening on {HOST}:{PORT}', file=sys.stderr)
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
