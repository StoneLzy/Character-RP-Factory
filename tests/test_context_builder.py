import unittest

from crpf.context_builder import build_context_samples


class ContextBuilderTests(unittest.TestCase):
    def test_build_context_uses_full_cleaned_dialogue(self):
        rows = [
            {
                "source_file": "CSV/pstory/hski/a.csv",
                "source_row": "1",
                "source_kind": "pstory",
                "chapter": "pstory/hski",
                "scene": "a",
                "speaker": "{user}",
                "line_id": "1",
                "text": "よろしくお願いします。",
                "translation": "请多多指教。",
            },
            {
                "source_file": "CSV/pstory/hski/a.csv",
                "source_row": "2",
                "source_kind": "pstory",
                "chapter": "pstory/hski",
                "scene": "a",
                "speaker": "",
                "line_id": "select",
                "text": "かわいいから",
                "translation": "因为很可爱",
            },
            {
                "source_file": "CSV/pstory/hski/a.csv",
                "source_row": "3",
                "source_kind": "pstory",
                "chapter": "pstory/hski",
                "scene": "a",
                "speaker": "咲季",
                "line_id": "3",
                "text": "当然よ。",
                "translation": "当然啦。",
            },
        ]

        samples = build_context_samples(rows, ("咲季",), previous_lines=2)

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["response"], "当然よ。")
        self.assertIn("{user}: よろしくお願いします。", samples[0]["context"])
        self.assertIn("选项: かわいいから", samples[0]["context"])
        self.assertIn("制作人: 请多多指教。", samples[0]["context_translation"])
        self.assertIn("选项: 因为很可爱", samples[0]["context_translation"])
        self.assertEqual(samples[0]["context_line_count"], 2)


if __name__ == "__main__":
    unittest.main()
