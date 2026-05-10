import unittest

from crpf.rag_index import RagSearchResult
from crpf.saki_chat import (
    ChatTurn,
    build_saki_prompt,
    clean_saki_reply,
    extract_character_profile_digest,
    format_saki_sources,
    prepare_saki_chat,
    should_use_rag,
)
from crpf.rag_chat import build_sources


class SakiChatTests(unittest.TestCase):
    def test_should_use_rag_respects_modes_and_keywords(self):
        self.assertTrue(should_use_rag("讲讲名古屋公演", "auto"))
        self.assertTrue(should_use_rag("今天有点累", "rag"))
        self.assertFalse(should_use_rag("今天有点累，鼓励我一下", "auto"))
        self.assertFalse(should_use_rag("咲季和佑芽是什么关系？", "casual"))

    def test_build_saki_prompt_contains_character_constraints(self):
        contexts = [
            RagSearchResult(
                rank=1,
                distance=0.1,
                text="咲季是佑芽的姐姐，也是竞争对手。",
                metadata={"source_path": "relationships.md", "topic": "relationships", "scene_id": "scene-00173"},
            )
        ]
        sources = tuple(build_sources(contexts))
        prompt = build_saki_prompt(
            user_message="你和佑芽是什么关系？",
            mode="rag",
            rag_used=True,
            contexts=contexts,
            sources=sources,
            history=[ChatTurn(user="你好", assistant="制作人，今天也要打起精神！")],
        )

        self.assertIn("称呼用户为“制作人”", prompt)
        self.assertIn("角色档案摘要", prompt)
        self.assertIn("不要过度恋爱化", prompt)
        self.assertIn("不要说“成为学园偶像大师”", prompt)
        self.assertIn("核心不是傲娇", prompt)
        self.assertIn("不要要求制作人去训练", prompt)
        self.assertIn("制作人的职责是制定安排、观察状态、给建议、复盘表现", prompt)
        self.assertIn("执行者是“我/咲季”", prompt)
        self.assertIn("不要把关心写成命令制作人训练", prompt)
        self.assertIn("月村手毬", prompt)
        self.assertIn("[S1]", prompt)
        self.assertIn("最近对话", prompt)
        self.assertIn("制作人：你好", prompt)

    def test_character_profile_digest_extracts_stable_profile_sections(self):
        digest = extract_character_profile_digest(
            """
# 花海咲季角色画像

## 1. 稳定身份与基础定位
- 咲季是制作人的担当偶像/培育对象。证据：scene-00172。

## 4. 能力画像与偶像素质
- 这段不应进入摘要。

### 制作人
- 制作人是发掘并培育咲季的人，也是核心搭档。

### 月村手毬
- 这段不应进入摘要。
"""
        )

        self.assertIn("担当偶像/培育对象", digest)
        self.assertIn("核心搭档", digest)
        self.assertNotIn("scene-00172", digest)
        self.assertNotIn("这段不应进入摘要", digest)

    def test_prepare_saki_chat_builds_casual_prompt_without_rag(self):
        prepared = prepare_saki_chat(
            user_message="今天有点累，鼓励我一下",
            chroma_dir="data/chroma_db",
            collection_name="hski_character_rag",
            embedding_model="bge-m3",
            ollama_base_url="http://localhost:11434",
            mode="casual",
        )

        self.assertEqual(prepared.message, "今天有点累，鼓励我一下")
        self.assertEqual(prepared.mode, "casual")
        self.assertFalse(prepared.rag_used)
        self.assertEqual(prepared.sources, ())
        self.assertEqual(prepared.contexts, ())
        self.assertIn("当前问题按日常聊天处理", prepared.prompt)

    def test_clean_reply_and_sources(self):
        self.assertEqual(clean_saki_reply("咲季：当然啦，制作人！"), "当然啦，制作人！")
        self.assertEqual(clean_saki_reply("手毫和琴音也一起去。"), "手毬和琴音也一起去。")
        self.assertEqual(format_saki_sources(()), "- 无")


if __name__ == "__main__":
    unittest.main()
