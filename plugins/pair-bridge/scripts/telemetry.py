"""Best-effort, timestamped device capacity samples; no credentials or prompts."""
from __future__ import annotations

import json
import platform
import base64
from pathlib import Path
import re
import shutil
import subprocess
import time


def _run(args: list[str], timeout: int = 10) -> str:
    result = subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False)
    if result.returncode:
        raise ValueError('Device telemetry command failed')
    return result.stdout.strip()


def _local_memory() -> dict:
    system = platform.system()
    if system == 'Darwin':
        total = int(_run(['sysctl', '-n', 'hw.memsize']))
        output = _run(['vm_stat'])
        page = re.search(r'page size of (\d+) bytes', output)
        if not page:
            raise ValueError('Cannot parse macOS memory counters')
        counters = {name: int(value.replace('.', '')) for name, value in
                    re.findall(r'^Pages (free|inactive|speculative):\s*([\d.]+)', output, re.MULTILINE)}
        if not all(name in counters for name in ('free', 'inactive', 'speculative')):
            raise ValueError('Cannot parse macOS memory counters')
        available = sum(counters.get(name, 0) for name in ('free', 'inactive', 'speculative')) * int(page.group(1))
        return {'total_bytes': total, 'available_bytes': min(total, available),
                'kind': 'macos_vm_estimate', 'gpu_memory': 'unified_with_system_memory'}
    if system == 'Linux':
        values = {key: int(value) * 1024 for key, value in
                  re.findall(r'^(MemTotal|MemAvailable):\s*(\d+) kB', Path('/proc/meminfo').read_text(), re.MULTILINE)}
        return {'total_bytes': values['MemTotal'], 'available_bytes': values['MemAvailable'], 'kind': 'linux_memavailable'}
    if system == 'Windows':
        script = ('$os = Get-CimInstance Win32_OperatingSystem; '
                  '@{total_bytes=([int64]$os.TotalVisibleMemorySize)*1024; '
                  'available_bytes=([int64]$os.FreePhysicalMemory)*1024} | ConvertTo-Json -Compress')
        encoded = base64.b64encode(script.encode('utf-16le')).decode('ascii')
        data = json.loads(_run(['powershell.exe', '-NoProfile', '-NonInteractive', '-EncodedCommand', encoded]))
        if not isinstance(data, dict) or not all(isinstance(data.get(key), int) and data[key] >= 0
                                                  for key in ('total_bytes', 'available_bytes')):
            raise ValueError('Cannot parse Windows memory counters')
        return {'total_bytes': data['total_bytes'], 'available_bytes': data['available_bytes'],
                'kind': 'windows_free_physical'}
    return {'status': 'unknown', 'reason': 'No local telemetry adapter for this operating system'}


def _nvidia_gpu() -> dict:
    try:
        output = _run(['nvidia-smi', '--query-gpu=memory.free,memory.total', '--format=csv,noheader,nounits'])
        pairs = [line.split(',') for line in output.splitlines()]
        devices = [{'free_bytes': int(pair[0].strip()) * 2**20, 'total_bytes': int(pair[1].strip()) * 2**20}
                   for pair in pairs if len(pair) == 2]
        return {'status': 'ok', 'devices': devices} if devices else {'status': 'unknown'}
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return {'status': 'unknown', 'reason': 'nvidia-smi unavailable'}


def _windows_over_ssh(host: str, models_path: str | None) -> dict:
    # Fixed read-only PowerShell script; the optional path is a single quoted literal.
    literal = "'" + (models_path or '').replace("'", "''") + "'"
    script = f"""
$os = Get-CimInstance Win32_OperatingSystem
$g = @()
try {{
  $rows = @(nvidia-smi --query-gpu=memory.free,memory.total --format=csv,noheader,nounits)
  foreach ($row in $rows) {{
    $x = $row -split ','
    if ($x.Count -eq 2) {{ $g += @{{free_bytes=([int64]$x[0].Trim())*1MB; total_bytes=([int64]$x[1].Trim())*1MB}} }}
  }}
}} catch {{}}
$p = {literal}
$disk = $null
$systemDrive = $env:SystemDrive
$systemDisk = $null
try {{
  $sd = Get-CimInstance Win32_LogicalDisk -Filter ("DeviceID='" + $systemDrive + "'")
  if ($sd) {{ $systemDisk = @{{free_bytes=[int64]$sd.FreeSpace; total_bytes=[int64]$sd.Size}} }}
}} catch {{}}
if ($p) {{
  try {{
    $root = [System.IO.Path]::GetPathRoot($p)
    $drive = $root.TrimEnd('\\')
    $d = Get-CimInstance Win32_LogicalDisk -Filter ("DeviceID='" + $drive + "'")
    if ($d) {{ $disk = @{{free_bytes=[int64]$d.FreeSpace; total_bytes=[int64]$d.Size}} }}
  }} catch {{}}
}}
@{{memory=@{{total_bytes=([int64]$os.TotalVisibleMemorySize)*1024; available_bytes=([int64]$os.FreePhysicalMemory)*1024; kind='windows_free_physical'}}; gpu=@{{status=$(if($g.Count){{'ok'}}else{{'unknown'}}); devices=$g}}; system_disk=$systemDisk; models_disk=$disk}} | ConvertTo-Json -Depth 6 -Compress
"""
    encoded = __import__('base64').b64encode(script.encode('utf-16le')).decode('ascii')
    output = _run(['ssh', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
                   '-o', 'ConnectTimeout=10', host, 'powershell.exe', '-NoProfile',
                   '-NonInteractive', '-EncodedCommand', encoded], timeout=20)
    try:
        return json.loads(output)
    except ValueError as exc:
        raise ValueError('Remote device returned invalid telemetry') from exc


def sample(device: dict) -> dict:
    """Return partial measurements rather than inventing unavailable capacity."""
    result = {'checked_at': time.time(), 'status': 'partial'}
    try:
        if device.get('ssh_host'):
            result.update(_windows_over_ssh(device['ssh_host'], device.get('models_path')))
        elif device.get('base_url', '').startswith(('http://127.0.0.1:', 'http://localhost:')):
            result['memory'] = _local_memory()
            result['gpu'] = _nvidia_gpu()
            system_usage = shutil.disk_usage(Path.home())
            result['system_disk'] = {'free_bytes': system_usage.free, 'total_bytes': system_usage.total}
            path = device.get('models_path')
            if path:
                usage = shutil.disk_usage(Path(path).expanduser())
                result['models_disk'] = {'free_bytes': usage.free, 'total_bytes': usage.total}
            else:
                result['models_disk'] = None
        else:
            result['reason'] = 'Direct remote endpoint has no OS telemetry adapter'
            return result
        result['status'] = 'ok' if isinstance(result.get('memory'), dict) and 'available_bytes' in result['memory'] else 'partial'
    except (OSError, ValueError, KeyError, subprocess.TimeoutExpired) as exc:
        result['reason'] = type(exc).__name__ + ': telemetry unavailable'
    return result
