import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import diagnostics
import jobs


class PortableDiagnosticTests(unittest.TestCase):
    def test_outcomes_redact_private_fields_and_export_without_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(diagnostics, 'journal_path', return_value=Path(tmp) / 'requests.jsonl'):
            diagnostics.record_outcome({'timestamp': 1, 'request_id': 'ok', 'operation': 'pair_ask',
                'device': 'pc', 'model': 'model', 'status': 'completed', 'answer': 'safe answer',
                'empty_answer_retries': 251, 'prompt': 'PRIVATE PROMPT', 'api_key': 'SECRET'})
            diagnostics.record_outcome({'timestamp': 2, 'request_id': 'err', 'operation': 'pair_ask',
                'device': 'pc', 'model': 'model', 'status': 'error', 'error_code': 'timeout',
                'error': diagnostics.safe_error('timeout'), 'raw_error': 'PRIVATE PROMPT SECRET'})
            rows, status = diagnostics.recent_outcomes(10)
            self.assertEqual(status, 'ok')
            self.assertEqual(rows[0]['empty_answer_retries'], 251)
            text = diagnostics.outcome_path().read_text(encoding='utf-8')
            self.assertNotIn('PRIVATE PROMPT', text)
            self.assertNotIn('SECRET', text)
            target = Path(tmp) / 'support.jsonl'
            result = diagnostics.export_outcomes(str(target), 10)
            self.assertEqual(result['records'], 2)
            with self.assertRaises(FileExistsError):
                diagnostics.export_outcomes(str(target), 10)

    def test_last_successful_model_skips_newer_failures(self):
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(diagnostics, 'journal_path', return_value=Path(tmp) / 'requests.jsonl'):
            path = diagnostics.outcome_path()
            rows = [{'status': 'completed', 'model': 'working', 'device': 'pc', 'answer': 'ok'}]
            rows.extend({'status': 'error', 'model': 'broken', 'device': 'pc', 'error_code': 'empty_answer'}
                        for _ in range(251))
            path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
            self.assertEqual(diagnostics.last_successful_model('pc'), {'device': 'pc', 'model': 'working'})

    def test_job_result_is_logged_only_at_completion(self):
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(jobs, '_append'), \
             patch.object(diagnostics, 'journal_path', return_value=Path(tmp) / 'requests.jsonl'):
            job = jobs.Job('pc', 'model')
            job.add_text('answer')
            warning = 'Fresh RAM/VRAM telemetry is unavailable; the 1 GiB free-memory reserve could not be confirmed.'
            job.update(status='completed', stage='completed', finish_reason='stop', empty_answer_retries=3,
                       memory_preflight={'status': 'estimated', 'warnings': [warning]})
            self.assertEqual(job.snapshot()['memory_preflight']['warnings'], [warning])
            rows, status = diagnostics.recent_outcomes()
            self.assertEqual(status, 'ok')
            self.assertEqual(rows[-1]['answer'], 'answer')
            self.assertEqual(rows[-1]['empty_answer_retries'], 3)


if __name__ == '__main__':
    unittest.main()
