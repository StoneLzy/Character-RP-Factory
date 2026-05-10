import tempfile
import unittest
from pathlib import Path

from crpf.rag_index import (
    RagChunk,
    build_rag_chunks,
    cosine_similarity,
    discover_rag_markdown_files,
    query_simple_vector_index,
    write_simple_vector_index,
)


class RagIndexTests(unittest.TestCase):
    def test_discovers_summary_docs_and_scene_cards(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "character_profile.md").write_text("# 画像\n\n内容", encoding="utf-8")
            (root / "index.md").write_text("# 索引\n\n不入库", encoding="utf-8")
            (root / "scenes").mkdir()
            (root / "scenes" / "scene-00001_demo.md").write_text("# scene-00001 demo", encoding="utf-8")

            paths = [path.relative_to(root).as_posix() for path in discover_rag_markdown_files(root)]

        self.assertEqual(paths, ["character_profile.md", "scenes/scene-00001_demo.md"])

    def test_builds_scene_card_chunk_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "scenes").mkdir()
            (root / "scenes" / "scene-00012_demo.md").write_text(
                "\n".join(
                    [
                        "# scene-00012 demo",
                        "",
                        "## 元数据",
                        "",
                        "- 来源：`CSV/cidol/hski/demo.csv:1-10`",
                        "- 章节：`cidol/hski`",
                        "- 场景：`demo`",
                        "- 主题：剧情推进、人物关系",
                        "- 咲季出场状态：direct",
                        "",
                        "## 场景理解摘要",
                        "",
                        "- 摘要：咲季要求制作人准备原创歌曲。",
                    ]
                ),
                encoding="utf-8",
            )

            chunks = build_rag_chunks(root)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].id, "scene-00012::card")
        self.assertEqual(chunks[0].metadata["doc_type"], "scene_card")
        self.assertEqual(chunks[0].metadata["scene_id"], "scene-00012")
        self.assertEqual(chunks[0].metadata["saki_presence"], "direct")
        self.assertIn("CSV/cidol/hski/demo.csv", chunks[0].metadata["source_csv"])

    def test_builds_summary_chunks_with_scene_refs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "relationships.md").write_text(
                "\n".join(
                    [
                        "# 关系",
                        "",
                        "## 制作人",
                        "",
                        "- 咲季信任制作人。证据：scene-00012, scene-00013。",
                        "",
                        "## 佑芽",
                        "",
                        "- 佑芽是咲季的妹妹。证据：scene-00100。",
                    ]
                ),
                encoding="utf-8",
            )

            chunks = build_rag_chunks(root, chunk_size=80, chunk_overlap=0)

        self.assertGreaterEqual(len(chunks), 2)
        scene_refs = " ".join(chunk.metadata["scene_ids"] for chunk in chunks)
        self.assertIn("scene-00012", scene_refs)
        self.assertIn("scene-00100", scene_refs)
        self.assertTrue(all(chunk.metadata["topic"] == "relationships" for chunk in chunks))

    def test_simple_vector_index_queries_by_cosine_similarity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            chunks = [
                RagChunk("a", "咲季和佑芽是姐妹。", {"source_path": "a.md", "topic": "relationships"}),
                RagChunk("b", "初星学园有演唱会。", {"source_path": "b.md", "topic": "worldbuilding"}),
            ]
            write_simple_vector_index(root, "test", chunks, [[1.0, 0.0], [0.0, 1.0]])

            results = query_simple_vector_index([0.9, 0.1], root, "test", top_k=1)

        self.assertEqual(results[0].metadata["topic"], "relationships")
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [1.0, 0.0]), 1.0)


if __name__ == "__main__":
    unittest.main()
