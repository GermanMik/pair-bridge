"""Deterministic concurrency proof without loading real models."""
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import jobs
import server


class ConcurrencyTests(unittest.TestCase):
    def test_background_jobs_on_distinct_devices_overlap(self):
        barrier = threading.Barrier(2)
        finished = threading.Event()
        errors = []
        count = []
        prompts = {}

        def worker(job, *args):
            try:
                prompts[job.device] = args[0]
                with server.inference_lock(job.device, wait_seconds=1):
                    job.update(status='running')
                    barrier.wait(timeout=3)
                    job.update(status='completed')
            except Exception as exc:
                errors.append(exc)
                job.update(status='failed')
            finally:
                count.append(job.id)
                if len(count) == 2:
                    finished.set()

        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(server, 'user_cache_path', return_value=Path(tmp)), \
                patch.object(jobs, 'journal_path', return_value=Path(tmp) / 'jobs.jsonl'), \
                patch.object(jobs, '_jobs', {}), \
                patch.object(server.management, 'devices', return_value={'pc_a': {}, 'pc_b': {}}), \
                patch.object(server, '_run_job', side_effect=worker):
            first = server.pair_job_start('pc_a', 'model', 'task A')
            second = server.pair_job_start('pc_b', 'model', 'task B')
            self.assertTrue(finished.wait(5), 'Both background workers must finish')
            self.assertEqual(errors, [])
            self.assertEqual(prompts, {'pc_a': 'task A', 'pc_b': 'task B'})
            self.assertEqual(server.pair_job_status(first['job_id'])['status'], 'completed')
            self.assertEqual(server.pair_job_status(second['job_id'])['status'], 'completed')

    def test_same_device_lock_excludes_other_process_but_other_device_is_free(self):
        code = (
            'import sys; from filelock import FileLock,Timeout; '
            'lock=FileLock(sys.argv[1],timeout=0)\n'
            'try: lock.acquire()\n'
            'except Timeout: sys.exit(3)\n'
            'lock.release()\n'
        )
        with tempfile.TemporaryDirectory() as tmp, patch.object(server, 'user_cache_path', return_value=Path(tmp)):
            with server.inference_lock('pc_a'):
                for device, expected in [('pc_a', 3), ('pc_b', 0)]:
                    result = subprocess.run([sys.executable, '-c', code, str(Path(tmp) / ('device-' + device + '.lock'))],
                                            capture_output=True, timeout=10)
                    self.assertEqual(result.returncode, expected, result.stderr.decode(errors='replace'))

    def test_capability_discovery_does_not_contact_or_load_engines(self):
        with patch.object(server.management, 'devices', return_value={'pc_a': {}, 'pc_b': {}}), \
                patch.object(server.management, 'client') as client:
            result = server.pair_capabilities()
            client.assert_not_called()
            self.assertEqual(result['configured_devices'], ['pc_a', 'pc_b'])
            self.assertEqual(result['max_active_jobs_per_process'], jobs.MAX_ACTIVE_JOBS)
            self.assertFalse(result['router_parallel_host_guarantee'])
            self.assertTrue(result['individual_tasks_per_target'])
