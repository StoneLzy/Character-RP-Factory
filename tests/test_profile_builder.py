import tempfile
import unittest
from pathlib import Path

from crpf.profile_builder import RAG_DOC_NAMES, build_rag_docs


class ProfileBuilderTests(unittest.TestCase):
    def test_build_rag_docs_writes_expected_templates(self):
        rows = [
            {
                "sample_id": "00000001",
                "source_file": "CSV/pstory/hski/a.csv",
                "source_row": "2",
                "source_kind": "pstory",
                "chapter": "pstory/hski",
                "scene": "a",
                "context_translation": "制作人: 请多多指教。",
                "response_translation": "当然。我可是未来的顶级偶像！",
                "response": "当然よ。未来のトップアイドルなんだから！",
                "quality_score": "100",
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            paths = build_rag_docs(rows, Path(tmp))

            self.assertEqual(sorted(path.name for path in paths), sorted(RAG_DOC_NAMES))
            content = (Path(tmp) / "character_profile.md").read_text(encoding="utf-8")
            self.assertIn("TODO", content)
            self.assertIn("基础统计", content)
            self.assertIn("未来的顶级偶像", content)


if __name__ == "__main__":
    unittest.main()
