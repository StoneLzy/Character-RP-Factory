import json
import tempfile
import unittest
from pathlib import Path

from crpf.source_trace import build_source_trace, normalize_scene_id


class SourceTraceTests(unittest.TestCase):
    def test_build_source_trace_reads_scene_card_summary_and_dialogue(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            rag_docs = root / "data" / "rag_docs"
            outputs = root / "outputs"
            (rag_docs / "scenes").mkdir(parents=True)
            outputs.mkdir()
            (rag_docs / "scenes" / "scene-00001_demo.md").write_text(
                "# scene-00001 demo\n\n## 场景理解摘要\n\n- 摘要：测试场景。",
                encoding="utf-8",
            )
            (outputs / "scene_summaries.jsonl").write_text(
                json.dumps(
                    {
                        "scene_id": "scene-00001",
                        "title": "测试标题",
                        "summary": "测试摘要",
                        "source_file": "CSV/demo.csv",
                        "key_evidence": [{"source_row": "1", "speaker": "咲季", "text_zh": "当然。"}],
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            (outputs / "scene_chunks.jsonl").write_text(
                json.dumps(
                    {
                        "scene_id": "scene-00001",
                        "dialogue_zh": [
                            {
                                "source_row": "1",
                                "speaker": "咲季",
                                "text_zh": "当然。",
                                "text_ja": "もちろん。",
                            }
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            trace = build_source_trace("scene-00001", rag_docs, outputs)

            self.assertEqual(trace["scene_id"], "scene-00001")
            self.assertIn("测试场景", trace["scene_card"])
            self.assertEqual(trace["summary"]["title"], "测试标题")
            self.assertEqual(trace["dialogue_lines"][0]["speaker"], "咲季")

    def test_normalize_scene_id_accepts_source_path(self):
        self.assertEqual(normalize_scene_id("scenes/scene-00123_demo.md"), "scene-00123")
        self.assertEqual(normalize_scene_id("bad"), "")


if __name__ == "__main__":
    unittest.main()
