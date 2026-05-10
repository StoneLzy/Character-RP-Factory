import json
import tempfile
import unittest
from pathlib import Path

from crpf.rag_summary_builder import build_raw_rag_summaries


class RagSummaryBuilderTests(unittest.TestCase):
    def test_build_raw_rag_summaries_from_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "CSV"
            source = root / "pstory" / "hski"
            source.mkdir(parents=True)
            (source / "scene.csv").write_text(
                "id,name,text,trans\n"
                "1,{user},よろしくお願いします。,请多多指教。\n"
                "2,咲季,当然よ。,当然啦。\n"
                "3,佑芽,お姉ちゃん！,姐姐！\n",
                encoding="utf-8",
            )
            output = Path(tmp) / "outputs"
            rag_docs = Path(tmp) / "data" / "rag_docs"

            written = build_raw_rag_summaries(root, output, rag_docs)

            self.assertTrue(written["scene_chunks"].exists())
            chunk = json.loads(written["scene_chunks"].read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(chunk["source_kind"], "pstory")
            self.assertIn("saki_speaker", chunk["related_reason"])

            self.assertTrue((output / "topic_summaries" / "plot_summary.md").exists())
            self.assertTrue((rag_docs / "team_story.md").exists())


if __name__ == "__main__":
    unittest.main()
