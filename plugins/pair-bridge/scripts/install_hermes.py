#!/usr/bin/env python3
"""One-command local setup for the PAIR Bridge Hermes provider."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request


ROOT = Path(__file__).resolve().parents[3]
PROXY = ROOT / 'plugins' / 'pair-bridge' / 'scripts' / 'hermes_proxy.py'
URL = 'http://127.0.0.1:8765/v1'
PROVIDER = 'pair-bridge'


def fail(message: str) -> "NoReturn":
    raise SystemExit('PAIR Bridge Hermes setup: ' + message)


def main() -> None:
    uv = shutil.which('uv')
    hermes = shutil.which('hermes')
    if not uv:
        fail('uv was not found on PATH. Install uv, then run this command again.')
    if not hermes:
        fail('Hermes was not found on PATH. Install Hermes, then run this command again.')

    home = Path.home()
    pair_config = home / '.pair-bridge.json'
    if not pair_config.exists():
        pair_config = home / '.codex-pair-bridge.json'
    if not pair_config.exists():
        fail('PAIR Bridge config is missing. Configure PAIR Bridge first.')
    try:
        config = json.loads(pair_config.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        fail('Could not read PAIR Bridge configuration: ' + str(exc))
    if not isinstance(config, dict):
        fail('PAIR Bridge configuration must be a JSON object.')

    env = os.environ.copy()
    env.pop('HERMES_PAIR_API_KEY', None)  # loopback-only; do not require a secret or write one to Hermes config
    env.setdefault('PAIR_BASE_URL', config.get('base_url', 'http://127.0.0.1:1234/v1'))
    env.setdefault('HERMES_PAIR_HOST', '127.0.0.1')
    env.setdefault('HERMES_PAIR_PORT', '8765')
    cache = home / '.cache' / 'pair-bridge'
    if sys.platform == 'win32':
        cache = home / 'AppData' / 'Local' / 'pair-bridge'
    cache.mkdir(parents=True, exist_ok=True)
    log_path = cache / 'hermes-gateway.log'

    # Reuse a currently running gateway. Do not start a duplicate process.
    def get_models() -> bool:
        try:
            with urllib.request.urlopen(URL + '/models', timeout=2) as response:
                return response.status == 200
        except (OSError, urllib.error.URLError):
            return False

    process = None
    if not get_models():
        log = log_path.open('ab')
        flags = subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS if sys.platform == 'win32' else 0
        process = subprocess.Popen([uv, 'run', '--script', str(PROXY)], cwd=ROOT, env=env,
                                   stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                   start_new_session=sys.platform != 'win32', creationflags=flags)
        log.close()
        for _ in range(40):
            if get_models():
                break
            if process.poll() is not None:
                fail(f'PAIR gateway exited during startup. See {log_path}')
            time.sleep(.25)
        else:
            # Only stop the process this invocation owns; reused gateways stay intact.
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            fail(f'PAIR gateway did not start. See {log_path}')

    # Use Hermes' own config writer so existing YAML settings are preserved.
    settings = {
        f'providers.{PROVIDER}.api': URL,
        f'providers.{PROVIDER}.transport': 'openai_chat',
        f'providers.{PROVIDER}.discover_models': 'true',
    }
    try:
        for key, value in settings.items():
            result = subprocess.run([hermes, 'config', 'set', key, value], text=True,
                                    capture_output=True, timeout=30)
            if result.returncode:
                fail('Hermes could not save its provider configuration: ' +
                     (result.stderr.strip() or result.stdout.strip() or key))
    except (OSError, subprocess.TimeoutExpired) as exc:
        fail('Could not run the Hermes config command: ' + str(exc))

    print('PAIR Bridge is connected to Hermes.')
    print('Use `hermes model` or `/model` to select a model from the pair-bridge provider.')
    print('Device models have IDs like `device/pc/model-key`; PAIR router models keep their usual IDs.')
    print(f'Gateway log: {log_path}')


if __name__ == '__main__':
    main()
