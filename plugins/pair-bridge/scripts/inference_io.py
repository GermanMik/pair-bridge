"""Bounded local validation for model metadata and OpenAI inference payloads.

Never fetch references, images or files. Error messages omit user/model content.
"""
from __future__ import annotations

import base64
import binascii
import copy
import json
import math
import re
import struct
from datetime import datetime, timezone

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError


def _positive_int(value):
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def output_budget(item: dict, requested: int | None = None,
                  planned_context_length: int | None = None) -> dict:
    """Known ceilings only; the engine accounts for prompt/reasoning tokens."""
    limits = []
    def add(value, source):
        if _positive_int(value):
            limits.append({'tokens': value, 'source': source})
    add(item.get('max_output_tokens'), 'engine.max_output_tokens')
    add(item.get('max_context_length'), 'engine.max_context_length')
    instances = item.get('loaded_instances') or []
    if len(instances) == 1:
        config = instances[0].get('config') or {}
        add(config.get('context_length'), 'loaded_instance.context_length')
        add(config.get('max_output_tokens'), 'loaded_instance.max_output_tokens')
    elif not instances:
        add(planned_context_length, 'requested_load.context_length')
    upper = min((row['tokens'] for row in limits), default=None)
    if requested is not None:
        if not _positive_int(requested) or requested < 32:
            raise ValueError('max_tokens must be an integer >= 32')
        if upper is not None and requested > upper:
            raise ValueError(f'Requested max_tokens={requested} exceeds known model/instance ceiling {upper}; inspect pair_model_capabilities')
    return {'requested_max_tokens': requested, 'known_ceiling': upper,
            'status': 'known' if limits else 'unknown', 'limits': limits,
            'remaining_context': 'engine_validated_including_prompt_and_reasoning'}


def model_capabilities(item: dict, engine: str) -> dict:
    """Only explicit metadata establishes positive model support."""
    raw = item.get('capabilities')
    raw = raw if isinstance(raw, dict) else {}
    kind = item.get('metadata_type', item.get('type'))
    reliable_type = item.get('type_source', 'engine_metadata') == 'engine_metadata'
    type_source = 'engine.task' if 'metadata_type' in item else 'engine.type'
    def state(value, source):
        return {'status': 'supported' if value is True else 'unsupported' if value is False else 'unknown',
                'source': source if isinstance(value, bool) else 'not_reported'}
    def flag(name):
        return state(raw.get(name), 'engine.capabilities.' + name)
    features = {
        'chat': state(kind == 'llm' if reliable_type and kind in ('llm', 'embedding', 'draft', 'other') else None, type_source),
        'embeddings': state(kind == 'embedding' if reliable_type and kind in ('llm', 'embedding', 'draft', 'other') else None, type_source),
        'vision': flag('vision'),
        'json_schema': flag('json_schema'),
        'tool_use': flag('trained_for_tool_use'),
    }
    maximum = item.get('max_context_length')
    instances = item.get('loaded_instances', [])
    return {'engine': engine, 'model': item['key'],
            'checked_at': datetime.now(timezone.utc).isoformat(),
            'capabilities': features,
            'max_context_length': maximum if _positive_int(maximum) else None,
            'context_source': 'engine.max_context_length' if _positive_int(maximum) else 'not_reported',
            'output_budget': output_budget(item),
            'loaded_contexts': [{'instance_id': row['id'],
                                 'context_length': (row.get('config') or {}).get('context_length')
                                 if _positive_int((row.get('config') or {}).get('context_length')) else None}
                                for row in instances],
            'notice': 'Metadata is not a successful inference probe. Unknown is not unsupported.'}


def _finite_json(value):
    raise ValueError('Non-finite JSON numbers are not allowed')


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON object keys are not allowed')
        result[key] = value
    return result


def _finite_float(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError('Non-finite JSON number')
    return result


def response_format(response_format: dict | None = None, json_schema: dict | None = None) -> dict | None:
    if response_format is not None and json_schema is not None:
        raise ValueError('Use response_format or json_schema, not both')
    if json_schema is not None:
        response_format = {'type': 'json_schema', 'json_schema': {'name': 'pair_response', 'strict': True, 'schema': json_schema}}
    if response_format is None:
        return None
    try:
        encoded = json.dumps(response_format, allow_nan=False)
        if len(encoded.encode()) > 32768:
            raise ValueError('Response format exceeds 32 KiB')
        fmt = json.loads(encoded)
    except (TypeError, OverflowError, RecursionError, ValueError) as exc:
        raise ValueError('Invalid or oversized response format') from exc
    if not isinstance(fmt, dict):
        raise ValueError('Response format must be an object')
    if fmt.get('type') == 'json_object' and set(fmt) == {'type'}:
        return fmt
    if fmt.get('type') != 'json_schema' or set(fmt) != {'type', 'json_schema'}:
        raise ValueError('Response format must be json_object or json_schema')
    spec = fmt['json_schema']
    if (not isinstance(spec, dict) or set(spec) - {'name', 'schema', 'strict', 'description'} or
            not isinstance(spec.get('name'), str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', spec['name']) or
            'strict' in spec and not isinstance(spec['strict'], bool) or
            'description' in spec and not isinstance(spec['description'], str) or
            not isinstance(spec.get('schema'), dict)):
        raise ValueError('Invalid json_schema definition')
    schema = spec['schema']
    # Reject references: guarantees no network resolution or recursive schema work.
    # Inline definitions instead; this deliberately bounded contract is documented.
    def walk(node, depth=0):
        if depth > 24:
            raise ValueError('JSON schema nesting exceeds 24 levels')
        if isinstance(node, dict):
            if any(key in node for key in ('$ref', '$dynamicRef', '$recursiveRef')):
                raise ValueError('JSON schema references are unsupported; inline definitions')
            for child in node.values():
                walk(child, depth + 1)
        elif isinstance(node, list):
            for child in node:
                walk(child, depth + 1)
    walk(schema)
    if '$schema' in schema and schema['$schema'] not in (
            'https://json-schema.org/draft/2020-12/schema', 'http://json-schema.org/draft/2020-12/schema'):
        raise ValueError('Only JSON Schema draft 2020-12 is supported')
    try:
        Draft202012Validator.check_schema(schema)
    except (SchemaError, RecursionError) as exc:
        raise ValueError('Invalid JSON schema') from exc
    return copy.deepcopy(fmt)


def validate_completion(result: dict, fmt: dict | None) -> dict:
    if fmt is None:
        return result
    if result.get('truncated') or result.get('finish_reason') != 'stop':
        raise ValueError('Structured response did not finish normally; no retry was made')
    answer = result.get('answer')
    if not isinstance(answer, str) or len(answer) > 48000:
        raise ValueError('Structured response exceeded output limit')
    try:
        value = json.loads(answer, parse_constant=_finite_json, parse_float=_finite_float, object_pairs_hook=_unique_object)
        if fmt['type'] == 'json_object':
            if not isinstance(value, dict):
                raise ValueError('Expected a JSON object')
        else:
            Draft202012Validator(fmt['json_schema']['schema']).validate(value)
    except (ValueError, ValidationError, RecursionError, TypeError) as exc:
        raise ValueError('Model returned invalid structured output; no retry was made') from exc
    return dict(result, structured_output=value, structured_validation='passed')


def require_structured(item: dict, fmt: dict | None):
    if fmt is not None and fmt['type'] == 'json_schema' and (item.get('capabilities') or {}).get('json_schema') is False:
        raise ValueError('Engine metadata reports JSON Schema unsupported for this model')


def _finite_number(value):
    try:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    except OverflowError:
        return False


def embedding_inputs(value: str | list[str]) -> list[str]:
    rows = [value] if isinstance(value, str) else value
    if (not isinstance(rows, list) or not 1 <= len(rows) <= 32 or
            any(not isinstance(row, str) or not row.strip() or len(row) > 8192 for row in rows) or
            sum(len(row) for row in rows) > 48000):
        raise ValueError('Embeddings require 1-32 nonblank texts, at most 8192 characters each and 48000 total')
    return list(rows)


def embeddings(data: dict, count: int, expected_dimensions: int | None = None) -> dict:
    rows = data.get('data')
    if not isinstance(rows, list) or len(rows) != count:
        raise ValueError('Embedding response must contain one vector per input')
    vectors = {}
    dimension = None
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError('Invalid embedding response')
        index, vector = row.get('index'), row.get('embedding')
        if (not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < count or index in vectors or
                not isinstance(vector, list) or not 1 <= len(vector) <= 16384 or
                any(not _finite_number(x) for x in vector)):
            raise ValueError('Invalid embedding index or vector')
        dimension = dimension or len(vector)
        if len(vector) != dimension or expected_dimensions is not None and len(vector) != expected_dimensions:
            raise ValueError('Embedding dimension mismatch')
        vectors[index] = vector
    return {'vectors': [vectors[index] for index in range(count)], 'dimensions': dimension,
            'count': count, 'reported_model': data.get('model'), 'usage': data.get('usage')}


def vision_content(prompt: str, images: list[str]) -> list[dict]:
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 48000:
        raise ValueError('Provide a nonblank vision prompt of at most 48000 characters')
    if not isinstance(images, list) or not 1 <= len(images) <= 4:
        raise ValueError('Provide 1-4 inline image data URLs')
    total = 0
    content = [{'type': 'text', 'text': prompt}]
    for url in images:
        if not isinstance(url, str) or len(url) > 5_592_500:
            raise ValueError('Image exceeds the 4 MiB decoded limit')
        match = re.fullmatch(r'data:image/(png|jpeg|webp);base64,([A-Za-z0-9+/]*={0,2})', url)
        if not match:
            raise ValueError('Only inline base64 PNG, JPEG or WebP images are accepted')
        try:
            raw = base64.b64decode(match[2], validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError('Invalid image base64') from exc
        total += len(raw)
        if not 1 <= len(raw) <= 4 * 2**20 or total > 12 * 2**20:
            raise ValueError('Images exceed decoded size limits (4 MiB each, 12 MiB total)')
        kind = match[1]
        valid = False
        if kind == 'png' and len(raw) >= 45 and raw[:16] == b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR':
            width, height = struct.unpack('>II', raw[16:24])
            valid = 0 < width <= 8192 and 0 < height <= 8192 and raw[-12:] == b'\x00\x00\x00\x00IEND\xaeB`\x82'
        elif kind == 'jpeg':
            valid = len(raw) >= 16 and raw[:3] == b'\xff\xd8\xff' and raw[-2:] == b'\xff\xd9'
        elif kind == 'webp':
            valid = (len(raw) >= 20 and raw[:4] == b'RIFF' and raw[8:12] == b'WEBP' and
                     struct.unpack('<I', raw[4:8])[0] == len(raw) - 8 and raw[12:16] in (b'VP8 ', b'VP8L', b'VP8X'))
        if not valid:
            raise ValueError('Image bytes do not match a supported image header/container')
        content.append({'type': 'image_url', 'image_url': {'url': url}})
    return content
