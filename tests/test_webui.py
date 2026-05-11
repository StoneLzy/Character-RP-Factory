import unittest

from crpf.config import ProjectConfig, RagConfig
from crpf.webui import WebUISettings, parse_backend, parse_chat_history, parse_chat_mode, parse_top_k, render_index_html


class WebUITests(unittest.TestCase):
    def test_render_index_html_contains_api_hooks_and_models(self):
        cfg = ProjectConfig(rag=RagConfig(top_k=4))
        settings = WebUISettings(
            config=cfg,
            embedding_model="bge-m3",
            embedding_provider="ollama",
            embedding_base_url="http://localhost:11434",
            embedding_api_key_env="",
            chat_model="qwen3.5:9b",
            chat_provider="ollama",
            chat_base_url="http://localhost:11434",
            chat_api_key_env="",
            collection_name="hski_character_rag",
            backend="auto",
        )

        html = render_index_html(settings)

        self.assertIn("/api/ask", html)
        self.assertIn("/api/query", html)
        self.assertIn("/api/chat-saki", html)
        self.assertIn("/api/chat-saki-stream", html)
        self.assertIn("/api/tts/saki", html)
        self.assertIn("data-play-tts", html)
        self.assertIn("播放语音", html)
        self.assertIn("renderMarkdown", html)
        self.assertIn("renderInlineMarkdown", html)
        self.assertIn("<pre><code", html)
        self.assertIn("replace(/\\r\\n/g, '\\n').split('\\n')", html)
        self.assertIn("/api/conversations", html)
        self.assertIn("/api/trace", html)
        self.assertIn("/assets/saki-avatar.jpg", html)
        self.assertIn("CRPF Saki", html)
        self.assertIn("历史聊天", html)
        self.assertIn("咲季聊天", html)
        self.assertIn("知识问答", html)
        self.assertIn("chatMode", html)
        self.assertIn("bge-m3", html)
        self.assertIn("qwen3.5:9b", html)
        self.assertIn("讲讲名古屋公演", html)
        self.assertIn("event.isComposing", html)
        self.assertIn("scrollChatToBottom", html)
        self.assertIn('id="sourcesDetails"', html)
        self.assertIn('id="contextsDetails"', html)
        self.assertIn("has-new", html)
        self.assertIn("markDetailNotice", html)
        self.assertIn("原文追溯", html)
        self.assertIn("data-trace-scene-id", html)
        self.assertIn("sceneIdList", html)
        self.assertIn("renderTraceButtons", html)
        self.assertIn("dialogue-card", html)
        self.assertIn("dialogue-list", html)
        self.assertNotIn("<details open>", html)

    def test_parse_top_k_and_backend_validate_ranges(self):
        self.assertEqual(parse_top_k("", 4), 4)
        self.assertEqual(parse_top_k("6", 4), 6)
        self.assertEqual(parse_backend("", "auto"), "auto")
        self.assertEqual(parse_backend("chroma", "auto"), "chroma")
        self.assertEqual(parse_chat_mode("", "auto"), "auto")
        self.assertEqual(parse_chat_mode("casual", "auto"), "casual")
        with self.assertRaises(ValueError):
            parse_top_k("0", 4)
        with self.assertRaises(ValueError):
            parse_backend("sqlite", "auto")
        with self.assertRaises(ValueError):
            parse_chat_mode("story", "auto")

    def test_parse_chat_history_keeps_valid_recent_turns(self):
        history = parse_chat_history(
            [
                {"user": "你好", "assistant": "制作人，今天也要打起精神！"},
                {"user": "", "assistant": "缺用户"},
                "bad",
            ]
        )

        self.assertEqual(len(history), 1)
        self.assertEqual(history[0].user, "你好")


if __name__ == "__main__":
    unittest.main()
