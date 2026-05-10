import unittest

from crpf.rag_chat import build_rag_prompt, build_sources, format_sources, parse_scene_ids
from crpf.rag_index import RagSearchResult


class RagChatTests(unittest.TestCase):
    def test_parse_scene_ids_handles_json_and_plain_text(self):
        self.assertEqual(parse_scene_ids('["scene-00001", "scene-00002"]'), ("scene-00001", "scene-00002"))
        self.assertEqual(parse_scene_ids("scene-00003, scene-00004"), ("scene-00003", "scene-00004"))
        self.assertEqual(parse_scene_ids(""), ())

    def test_build_sources_and_prompt_include_source_labels(self):
        contexts = [
            RagSearchResult(
                rank=1,
                distance=0.12,
                text="咲季是花海佑芽的姐姐，也把佑芽视为重要竞争对手。",
                metadata={
                    "source_path": "relationships.md",
                    "topic": "relationships",
                    "scene_ids": '["scene-00173"]',
                },
            )
        ]

        sources = tuple(build_sources(contexts))
        prompt = build_rag_prompt("咲季和佑芽是什么关系？", contexts, sources)

        self.assertIn("[S1]", prompt)
        self.assertIn("relationships.md", prompt)
        self.assertIn("scene-00173", prompt)
        self.assertIn("只能根据", prompt)
        self.assertIn("- [S1] relationships.md", format_sources(sources))


if __name__ == "__main__":
    unittest.main()
