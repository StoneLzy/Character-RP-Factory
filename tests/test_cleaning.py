import unittest

from crpf.cleaning import clean_rows, filter_character_lines, normalize_text
from crpf.config import DEFAULT_COLUMN_ALIASES


class CleaningTests(unittest.TestCase):
    def test_normalize_text_removes_bom_and_literal_newlines(self):
        self.assertEqual(normalize_text("\ufeffあ\\nい　 う"), "あ\nい う")

    def test_clean_rows_standardizes_and_drops_noise(self):
        rows = [
            {"id": "1", "name": "咲季", "text": "こんにちは", "trans": "你好"},
            {"id": "2", "name": "info", "text": "adv.txt", "trans": ""},
            {"id": "3", "name": "译者", "text": "", "trans": ""},
        ]
        cleaned = clean_rows(rows, DEFAULT_COLUMN_ALIASES, excluded_speakers=("info", "译者"), bad_patterns=())
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(cleaned[0]["speaker"], "咲季")
        self.assertEqual(cleaned[0]["translation"], "你好")

    def test_filter_character_lines_uses_target_names_and_length(self):
        rows = [
            {"speaker": "咲季", "text": "当然よ。", "translation": ""},
            {"speaker": "{user}", "text": "よろしく", "translation": ""},
            {"speaker": "咲季", "text": "あ", "translation": ""},
        ]
        selected = filter_character_lines(rows, ("咲季",), min_chars=2, max_chars=20)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["text"], "当然よ。")


if __name__ == "__main__":
    unittest.main()
