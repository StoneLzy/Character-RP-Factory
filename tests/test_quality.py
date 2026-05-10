import unittest

from crpf.quality import score_samples


class QualityTests(unittest.TestCase):
    def test_score_samples_splits_good_and_review(self):
        samples = [
            {
                "sample_id": "1",
                "context": "{user}: よろしくお願いします。",
                "context_line_count": "1",
                "response": "当然よ。",
            },
            {
                "sample_id": "2",
                "context": "",
                "context_line_count": "0",
                "response": "……",
            },
        ]

        good, review = score_samples(samples, min_response_chars=2, max_response_chars=50, min_quality_score=70)

        self.assertEqual(len(review), 2)
        self.assertEqual(len(good), 1)
        self.assertEqual(review[0]["keep"], "yes")
        self.assertIn("punctuation_only", review[1]["quality_reason"])


if __name__ == "__main__":
    unittest.main()
