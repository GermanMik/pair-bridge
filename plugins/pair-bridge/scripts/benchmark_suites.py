"""Versioned, objective local suites. Never execute model-generated code."""
from __future__ import annotations

import hashlib
import json

VERSION = '1'
SIZES = {'smoke': 6, 'short': 20, 'full': 100}
CATEGORIES = ('arithmetic', 'logic', 'json', 'code', 'long_context')


def _cases() -> list[dict]:
    rows = []
    for n in range(10):
        for category in CATEGORIES:
            for lang in ('en', 'ru'):
                prefix = ('Return only the answer, no explanation. ' if lang == 'en' else
                          'Верни только ответ, без пояснений. ')
                a, b = n + 7, n + 3
                validator = 'exact'
                if category == 'arithmetic':
                    question, expected = f'({a} + {b}) * 3 - {n} = ?', str((a + b) * 3 - n)
                elif category == 'logic':
                    question = (f'A > B, B > C, C > D. Which symbol is greatest? Trial {n}.' if lang == 'en' else
                                f'A > B, B > C, C > D. Какой символ наибольший? Вариант {n}.')
                    # Alternate order so "A" is not a fixed-answer benchmark.
                    symbols = ['A', 'B', 'C', 'D']
                    symbols = symbols[n % 4:] + symbols[:n % 4]
                    for old, new in zip('ABCD', 'WXYZ'):
                        question = question.replace(old, new)
                    for old, new in zip('WXYZ', symbols):
                        question = question.replace(old, new)
                    expected = symbols[0]
                elif category == 'json':
                    question = (f'Return a JSON object with exactly count={a} (integer), ok=true (boolean).' if lang == 'en' else
                                f'Верни JSON-объект: только count={a} (целое число), ok=true (логическое значение).')
                    expected, validator = {'count': a, 'ok': True}, 'json_exact'
                elif category == 'code':
                    code = f'values = [{n}, {n+1}, {n+2}, {n+3}]\nprint(sum(x * 2 for x in values if x % 2 == 0))'
                    question = ('What does this Python program print?\n' if lang == 'en' else
                                'Что выведет эта программа Python?\n') + code
                    expected = str(sum(x * 2 for x in range(n, n + 4) if x % 2 == 0))
                else:
                    # ~10k characters, repeated distractors and a different key on each trial.
                    key, expected = f'target_{n}', f'VALUE_{n+37}'
                    lines = [f'record_{i}: ignored_{(i * 7 + n) % 991}' for i in range(420)]
                    lines.insert(25 + n * 37, f'{key}: {expected}')
                    question = ('Find the exact value for ' if lang == 'en' else 'Найди точное значение для ')
                    question += key + ':\n' + '\n'.join(lines)
                rows.append({'id': f'{category}-{n+1:02d}-{lang}', 'category': category,
                             'language': lang, 'prompt': prefix + question,
                             'validator': validator, 'expected': expected})
    return rows


def suite(name: str) -> dict:
    if name not in SIZES:
        raise ValueError('Unknown benchmark suite; use smoke, short or full')
    cases = _cases()[:SIZES[name]]
    fingerprint = hashlib.sha256(json.dumps(cases, sort_keys=True, ensure_ascii=False,
                                             separators=(',', ':')).encode()).hexdigest()
    return {'suite': name, 'suite_version': VERSION, 'suite_fingerprint': fingerprint, 'cases': cases}


def catalog() -> dict:
    rows = []
    for name in SIZES:
        data = suite(name)
        cases = data.pop('cases')
        rows.append(dict(data, case_count=len(cases),
                         categories={cat: sum(c['category'] == cat for c in cases) for cat in CATEGORIES
                                     if any(c['category'] == cat for c in cases)},
                         languages={lang: sum(c['language'] == lang for c in cases) for lang in ('ru', 'en')},
                         routing_eligible=name != 'smoke'))
    return {'suites': rows, 'nesting': 'smoke is a prefix of short; short is a prefix of full',
            'validation': 'Exact answers and strict JSON equality; code comprehension, no generated code execution.',
            'limitations': 'Synthetic local suite, not a general intelligence score. Long-context retrieval uses about 10k characters.'}


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON key')
        result[key] = value
    return result


def evaluate(case: dict, answer: str, finish_reason: str) -> str:
    if finish_reason != 'stop':
        return 'format_error'
    if case['validator'] == 'exact':
        return 'pass' if answer.strip() == case['expected'] else 'mismatch'
    try:
        def invalid_constant(value):
            raise ValueError('Non-finite JSON')
        actual = json.loads(answer, object_pairs_hook=_object, parse_constant=invalid_constant)
    except (ValueError, TypeError, RecursionError):
        return 'format_error'
    if (not isinstance(actual, dict) or set(actual) != {'count', 'ok'} or
            type(actual['count']) is not int or type(actual['ok']) is not bool):
        return 'format_error'
    return 'pass' if actual == case['expected'] else 'mismatch'
