"""Typed inference outcomes and local, explicit recovery choices. No prompts are persisted."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import uuid
from platformdirs import user_cache_path


class InferenceFailure(ValueError):
    def __init__(self, code, metadata=None):
        self.code = code
        self.metadata = safe_metadata(metadata or {})
        if code in ('tool_response', 'refusal', 'invalid_response', 'target_changed'):
            self.metadata['cause'] = code
        super().__init__('Model returned no final text' if code == 'empty_answer' else {
            'tool_response': 'Model returned a tool call instead of a final answer',
            'refusal': 'Model declined the request',
            'invalid_response': 'Model returned an invalid completion response',
            'target_changed': 'The original loaded instance changed; inspect device state',
        }.get(code, 'Model inference failed'))


FINISH_REASONS = {'stop', 'length', 'tool_calls', 'function_call', 'content_filter', 'unknown'}
USAGE_FIELDS = ('prompt_tokens', 'completion_tokens', 'total_tokens', 'reasoning_tokens')


def safe_usage(value):
    if not isinstance(value, dict):
        return {}
    result = {k: value[k] for k in USAGE_FIELDS
              if isinstance(value.get(k), int) and not isinstance(value[k], bool) and value[k] >= 0}
    details = value.get('completion_tokens_details')
    if isinstance(details, dict):
        count = details.get('reasoning_tokens')
        if isinstance(count, int) and not isinstance(count, bool) and count >= 0:
            result['reasoning_tokens'] = count
    return result


def safe_metadata(value):
    if not isinstance(value, dict):
        return {}
    result = {}
    reason = value.get('finish_reason')
    if reason is not None:
        result['finish_reason'] = reason if isinstance(reason, str) and reason in FINISH_REASONS else 'unknown'
    result['usage'] = safe_usage(value.get('usage'))
    for key in ('attempt', 'reasoning_characters', 'max_tokens'):
        n = value.get(key)
        if isinstance(n, int) and not isinstance(n, bool) and n >= 0:
            result[key] = n
    for key in ('reasoning_present', 'stream_completed', 'enable_thinking'):
        if type(value.get(key)) is bool:
            result[key] = value[key]
    cause = value.get('cause')
    if cause in ('budget_exhausted', 'reasoning_only', 'empty_completed', 'target_changed', 'completed',
                 'tool_response', 'refusal', 'invalid_response'):
        result['cause'] = cause
    return result


def metadata(reason=None, usage=None, reasoning_characters=0, stream_completed=True):
    row = safe_metadata({'finish_reason': reason, 'usage': usage,
                         'reasoning_characters': reasoning_characters,
                         'reasoning_present': reasoning_characters > 0,
                         'stream_completed': stream_completed})
    if reason == 'length':
        row['cause'] = 'budget_exhausted'
    elif reasoning_characters or row['usage'].get('reasoning_tokens', 0):
        row['cause'] = 'reasoning_only'
        row['reasoning_present'] = True
    else:
        row['cause'] = 'empty_completed'
    return row


def check_final(text, row, message=None):
    message = message if isinstance(message, dict) else {}
    reason = row.get('finish_reason')
    if message.get('refusal') or reason == 'content_filter':
        raise InferenceFailure('refusal', row)
    if message.get('tool_calls') or message.get('function_call') or reason in ('tool_calls', 'function_call'):
        raise InferenceFailure('tool_response', row)
    if not isinstance(text, str) or not text.strip():
        raise InferenceFailure('empty_answer', row)
    if reason != 'length':
        row['cause'] = 'completed'
    return text


def content_text(value, separator='\n'):
    """Validate response content parts; malformed content is never a retryable empty."""
    if value is None:
        return '', 0, {}
    if isinstance(value, str):
        return value, 0, {}
    if not isinstance(value, list):
        raise InferenceFailure('invalid_response')
    parts, reasoning, special = [], 0, {}
    for part in value:
        if not isinstance(part, dict):
            raise InferenceFailure('invalid_response')
        kind = part.get('type')
        if kind in (None, 'text', 'output_text'):
            if not isinstance(part.get('text'), str):
                raise InferenceFailure('invalid_response')
            parts.append(part['text'])
        elif kind in ('reasoning', 'reasoning_content'):
            text = part.get('text', part.get('content'))
            if not isinstance(text, str):
                raise InferenceFailure('invalid_response')
            reasoning += len(text)
        elif kind == 'refusal':
            special['refusal'] = True
        elif kind in ('tool_call', 'tool_calls', 'function_call'):
            special['tool_calls'] = True
        else:
            raise InferenceFailure('invalid_response')
    return separator.join(parts), reasoning, special


def delay(attempt):
    # Jitter spreads clients without imposing an attempt or total-time ceiling.
    return min(30.0, (2 ** min(max(attempt - 1, 0), 5)) * random.uniform(.8, 1.2))


def budget_exhausted(row):
    tokens = row.get('max_tokens')
    used = safe_usage(row.get('usage')).get('completion_tokens')
    return row.get('finish_reason') == 'length' or (
        type(tokens) is int and type(used) is int and used >= tokens)


def profile_path(device, model):
    key = hashlib.sha256(json.dumps([device, model]).encode()).hexdigest()
    return user_cache_path('pair-bridge', appauthor=False) / 'recovery-profiles' / (key + '.json')


def load_profile(device, model):
    try:
        row = json.loads(profile_path(device, model).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}
    if not isinstance(row, dict):
        return {}
    result = {}
    if type(row.get('enable_thinking')) is bool:
        result['enable_thinking'] = row['enable_thinking']
    if type(row.get('max_tokens')) is int and row['max_tokens'] >= 32:
        result['max_tokens'] = row['max_tokens']
    return result


def save_profile(device, model, parameters):
    target = profile_path(device, model)
    target.parent.mkdir(parents=True, exist_ok=True)
    row = {k: parameters[k] for k in ('max_tokens', 'enable_thinking') if k in parameters}
    temporary = target.with_name(uuid.uuid4().hex + '.tmp')
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(row, stream)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
