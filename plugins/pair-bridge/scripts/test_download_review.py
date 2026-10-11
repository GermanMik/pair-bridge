import unittest
from unittest.mock import patch

import download_review
import server


class DownloadReviewTests(unittest.TestCase):
    def test_one_exact_hugging_face_quantization(self):
        info = {'sha': 'revision123', 'siblings': [
            {'rfilename': 'model-Q4_K_M.gguf', 'size': 1000},
            {'rfilename': 'model-Q8_0.gguf', 'size': 2000}]}
        with patch.object(download_review.httpx, 'Client') as factory:
            factory.return_value.__enter__.return_value.get.return_value.json.return_value = info
            result = download_review.model_files('https://huggingface.co/owner/repo', 'Q4_K_M')
        self.assertEqual(result['status'], 'verified_file')
        self.assertEqual(result['size_bytes'], 1000)
        self.assertEqual(result['revision'], 'revision123')

    def test_ambiguous_repository_does_not_claim_exact_size(self):
        info = {'sha': 'revision123', 'siblings': [
            {'rfilename': 'a.gguf', 'size': 1000}, {'rfilename': 'b.gguf', 'size': 2000}]}
        with patch.object(download_review.httpx, 'Client') as factory:
            factory.return_value.__enter__.return_value.get.return_value.json.return_value = info
            result = download_review.model_files('https://huggingface.co/owner/repo', None)
        self.assertEqual(result['status'], 'ambiguous')

    def test_download_plan_uses_verified_size_and_rejects_low_space(self):
        with patch.object(server.management, "client") as client, patch.object(server.download_review, "model_files") as metadata:
            with self.assertRaisesRegex(ValueError, "downloads are disabled"):
                server.pair_download_plan("fixture", "fixture/model")
            client.assert_not_called()
            metadata.assert_not_called()

    def test_download_rechecks_revision_before_start(self):
        with patch.object(server.management, "client") as client, patch.object(server.download_review, "model_files") as metadata:
            with self.assertRaisesRegex(ValueError, "downloads are disabled"):
                server.pair_download_plan("fixture", "fixture/model")
            client.assert_not_called()
            metadata.assert_not_called()


if __name__ == '__main__':
    unittest.main()
