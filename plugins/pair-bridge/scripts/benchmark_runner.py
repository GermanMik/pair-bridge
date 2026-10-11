"""Bounded streaming measurements on an exact already-loaded instance."""
from __future__ import annotations

import asyncio
import json
import math
import time

import httpx
import benchmark_store
import benchmark_suites
import inference_io
import management


def validate_request(device, model, suite, repeats, max_tokens, request_timeout_seconds=180, enable_thinking=None):
    if (not isinstance(device, str) or not device or len(device) > 64 or
            not isinstance(model, str) or not model or len(model) > 512):
        raise ValueError('Use exact configured device and model keys')
    if type(repeats) is not int or not 1 <= repeats <= 5:
        raise ValueError('Benchmark repeats must be between 1 and 5')
    if type(max_tokens) is not int or max_tokens < 32:
        raise ValueError('Benchmark max_tokens must be an integer >= 32')
    if type(request_timeout_seconds) is not int or not 30 <= request_timeout_seconds <= 1200:
        raise ValueError('Benchmark request_timeout_seconds must be an integer 30-1200')
    if enable_thinking is not None and type(enable_thinking) is not bool:
        raise ValueError('Benchmark enable_thinking must be a boolean or null')
    return benchmark_suites.suite(suite)


async def stream(job, c, payload):
    started = time.monotonic()
    first_at = None
    text, finish, usage, ended = '', None, None, False
    # wait_for bounds total request time, including a server that never closes the stream.
    async with c.stream('POST', '/v1/chat/completions', json=payload) as response:
        if not response.is_success:
            raise ValueError('http_error')
        async for line in response.aiter_lines():
            if job.cancel_event.is_set():
                raise asyncio.CancelledError()
            if not line.startswith('data:'):
                continue
            event = line[5:].strip()
            if not event:
                continue
            if event == '[DONE]':
                ended = True
                break
            if len(event) > 1_000_000:
                raise ValueError('invalid_stream')
            try:
                data = json.loads(event)
            except (ValueError, RecursionError) as exc:
                raise ValueError('invalid_stream') from exc
            if not isinstance(data, dict) or 'error' in data:
                raise ValueError('invalid_stream')
            if isinstance(data.get('usage'), dict):
                usage = data['usage'].get('completion_tokens')
            choices = data.get('choices')
            if not isinstance(choices, list) or not choices:
                continue
            choice = choices[0]
            if not isinstance(choice, dict):
                raise ValueError('invalid_stream')
            if choice.get('finish_reason') is not None:
                finish = choice['finish_reason']
            delta = choice.get('delta')
            content = delta.get('content') if isinstance(delta, dict) else None
            if isinstance(content, str) and content:
                if first_at is None:
                    first_at = time.monotonic()
                text += content
                if len(text) > 48000:
                    raise ValueError('output_limit')
        if not ended or finish is None:
            raise ValueError('incomplete_stream')
        if not text.strip():
            raise ValueError('empty_response')
    elapsed = time.monotonic() - started
    tokens = usage if type(usage) is int and usage >= 0 else None
    # completion_tokens can include reasoning. This is usage / end-to-end seconds,
    # explicitly NOT inferred token counts or pure decoder speed.
    throughput = tokens / elapsed if tokens is not None and elapsed > 0 else None
    return {'answer': text, 'finish_reason': finish, 'latency_ms': round(elapsed * 1000, 3),
            'ttft_ms': round((first_at - started) * 1000, 3) if first_at is not None else None,
            'completion_tokens': tokens, 'tokens_per_second': round(throughput, 3) if throughput is not None else None}


def measure(job, client, payload):
    timeout = getattr(job, 'benchmark_request_timeout_seconds', 180)
    async def run():
        async with httpx.AsyncClient(base_url=str(client.base_url), headers=dict(client.headers),
                                     timeout=timeout, trust_env=False, follow_redirects=False) as c:
            return await asyncio.wait_for(stream(job, c, payload), timeout=timeout)
    with asyncio.Runner() as runner:
        loop = runner.get_loop()
        task = loop.create_task(run())
        with job.lock:
            job.loop, job.task = loop, task
            if job.cancel_event.is_set():
                task.cancel()
        try:
            return runner.run(task_wait(task))
        finally:
            with job.lock:
                job.loop, job.task = None, None


async def task_wait(task):
    return await task


def error_code(exc):
    if isinstance(exc, (TimeoutError, httpx.TimeoutException)):
        return 'timeout'
    if isinstance(exc, httpx.RequestError):
        return 'transport_error'
    if str(exc) in ('http_error', 'invalid_stream', 'output_limit', 'incomplete_stream', 'empty_response'):
        return str(exc)
    return 'runtime_error'


def worker(job, row, data, lock_factory):
    """Uses job lifecycle/locks; retains no prompts or responses in job metadata."""
    try:
        row['job_id'] = job.id
        benchmark_store.persist(row)
        with lock_factory(row['device'], wait_seconds=30), management.client(row['device']) as c:
            if job.cancel_event.is_set():
                raise asyncio.CancelledError()
            selected = management.find_model(c, row['model'], row['device'])
            if selected.get('type') != 'llm' or len(selected.get('loaded_instances', [])) != 1:
                raise ValueError('Benchmark requires exactly one already-loaded chat instance')
            instance = selected['loaded_instances'][0]
            config = instance.get('config') or {}
            context = config.get('context_length')
            if type(context) is not int or context < 512:
                raise ValueError('Loaded context is unknown')
            budget = inference_io.output_budget(selected, row['max_tokens'])
            thinking = row.get('enable_thinking')
            if thinking is not None:
                if management.engine_for(row['device']) != 'unsloth':
                    raise ValueError('Explicit thinking control is not supported for this engine')
                schema = management.request(c, 'GET', '/openapi.json')
                fields = schema.get('components', {}).get('schemas', {}).get('ChatCompletionRequest', {}).get('properties', {})
                if 'enable_thinking' not in fields:
                    raise ValueError('Engine does not advertise explicit thinking control')
            job.benchmark_request_timeout_seconds = row.get('request_timeout_seconds', 180)
            # Whitelist known scalar load settings. Never copy arbitrary config, URLs or auth.
            row['loaded_config'] = {key: config[key] for key in
                                    ('context_length', 'gpu_offload_ratio', 'flash_attention', 'eval_batch_size')
                                    if type(config.get(key)) in (int, float, bool) and
                                    (not isinstance(config[key], float) or math.isfinite(config[key]))}
            row.update(context_length=context, instance_id=instance['id'], engine=management.engine_for(row['device']),
                       status='running', job_id=job.id)
            benchmark_store.persist(row)
            job.update(status='running', stage='warmup', instance_id=instance['id'], progress=0,
                       output_budget=budget)
            base = {'model': management.chat_model_id(row['device'], row['model'], instance['id']),
                    'temperature': 0, 'max_tokens': row['max_tokens'], 'stream': True,
                    'stream_options': {'include_usage': True}}
            if thinking is not None:
                base['enable_thinking'] = thinking
            def payload(case):
                return dict(base, messages=[{'role': 'user', 'content': case['prompt']}])
            # One warmup, excluded from measurements. Failure aborts, no automatic retry.
            measure(job, c, payload(data['cases'][0]))
            for repeat in range(row['repeats']):
                for case in data['cases']:
                    if job.cancel_event.is_set():
                        raise asyncio.CancelledError()
                    job.update(stage='benchmark', progress=len(row['samples']) / row['expected_samples'])
                    started = time.monotonic()
                    sample = {'case_id': case['id'], 'category': case['category'], 'language': case['language'],
                              'repeat': repeat + 1, 'ttft_ms': None, 'completion_tokens': None,
                              'tokens_per_second': None, 'latency_ms': None}
                    try:
                        result = measure(job, c, payload(case))
                        sample['status'] = benchmark_suites.evaluate(case, result['answer'], result['finish_reason'])
                        for key in ('latency_ms', 'ttft_ms', 'completion_tokens', 'tokens_per_second'):
                            sample[key] = result[key]
                    except (ValueError, TimeoutError, httpx.HTTPError) as exc:
                        sample.update(status=error_code(exc), latency_ms=round((time.monotonic() - started) * 1000, 3))
                    row['samples'].append(sample)
                    benchmark_store.persist(row)
                    # Transport failure may still be computing. Stop this run; never issue next case.
                    if sample['status'] not in ('pass', 'mismatch', 'format_error'):
                        row['error_code'] = sample['status']
                        raise ValueError('Benchmark transport failed; inspect engine before another run')
            # Serialize terminal publication with jobs.cancel: an accepted cancellation
            # must not be overwritten by completed between the last check and persist.
            with job.lock:
                if job.cancel_event.is_set():
                    raise asyncio.CancelledError()
                row['status'] = 'completed'
                benchmark_store.persist(row)
                job.update(status='completed', stage='completed', progress=1,
                           structured_output=benchmark_store.summarize(row))
    except asyncio.CancelledError:
        row.update(status='cancelled', error_code='cancelled')
        try:
            benchmark_store.persist(row)
        finally:
            job.update(status='cancelled', stage='cancelled', error_code='cancelled')
    except Exception as exc:
        cancelled = job.cancel_event.is_set()
        row.update(status='cancelled' if cancelled else 'failed',
                   error_code='cancelled' if cancelled else row.get('error_code') or error_code(exc))
        try:
            benchmark_store.persist(row)
        finally:
            job.update(status=row['status'], stage=row['status'], error_code=row['error_code'])
