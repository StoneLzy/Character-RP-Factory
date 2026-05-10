import json
import tempfile
import unittest
from pathlib import Path

from crpf.exporters import export_jsonl, sanitize_chinese_text


class ExporterTests(unittest.TestCase):
    def test_export_chatml_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "train.jsonl"
            count = export_jsonl(
                [
                    {
                        "sample_id": "1",
                        "context": "{user}: よろしくお願いします。",
                        "context_translation": "制作人: 请多多指教。",
                        "response": "当然よ。",
                        "response_translation": "当然啦。",
                        "quality_score": "100",
                        "keep": "yes",
                    }
                ],
                output,
                export_format="chatml",
                max_context_chars=100,
                language="ja",
            )

            self.assertEqual(count, 1)
            record = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(record["messages"][-1]["content"], "当然よ。")

    def test_export_zh_chatml_uses_translation_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "train_zh.jsonl"
            count = export_jsonl(
                [
                    {
                        "sample_id": "1",
                        "context": "{user}: よろしくお願いします。",
                        "context_translation": "制作人: 请多多指教。",
                        "response": "当然よ。",
                        "response_translation": "当然啦。",
                        "quality_score": "100",
                        "keep": "yes",
                    }
                ],
                output,
                export_format="chatml",
                max_context_chars=100,
                language="zh",
            )

            self.assertEqual(count, 1)
            record = json.loads(output.read_text(encoding="utf-8"))
            self.assertIn("简体中文", record["messages"][0]["content"])
            self.assertIn("制作人: 请多多指教。", record["messages"][1]["content"])
            self.assertEqual(record["messages"][-1]["content"], "当然啦。")

    def test_export_instruction_jsonl_skips_no_keep(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "train.jsonl"
            count = export_jsonl(
                [
                    {"context": "a", "response": "b", "keep": "no"},
                    {"context": "a", "response": "c", "keep": "yes"},
                ],
                output,
                export_format="instruction",
                max_context_chars=100,
                language="ja",
            )

            self.assertEqual(count, 1)
            record = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(record["output"], "c")

    def test_export_zh_skips_records_with_remaining_kana(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "train_zh.jsonl"
            count = export_jsonl(
                [
                    {
                        "context_translation": "女の子: 你好",
                        "response_translation": "当然啦。",
                        "keep": "yes",
                    },
                    {
                        "context_translation": "制作人: 你好",
                        "response_translation": "当然啦。",
                        "keep": "yes",
                    },
                ],
                output,
                export_format="chatml",
                max_context_chars=100,
                language="zh",
            )

            self.assertEqual(count, 1)

    def test_sanitize_chinese_text_removes_ruby_markup(self):
        self.assertEqual(sanitize_chinese_text("<r\\=ライバル>是敌人</r>哦。"), "是敌人哦。")


if __name__ == "__main__":
    unittest.main()
