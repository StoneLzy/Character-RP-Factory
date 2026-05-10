import unittest

from crpf.config import ProjectConfig, RagConfig
from crpf.webui import WebUISettings, parse_backend, parse_top_k, render_index_html


class WebUITests(unittest.TestCase):
    def test_render_index_html_contains_api_hooks_and_models(self):
        cfg = ProjectConfig(rag=RagConfig(top_k=4))
        settings = WebUISettings(
            config=cfg,
            embedding_model="bge-m3",
            chat_model="qwen3.5:9b",
            collection_name="hski_character_rag",
            backend="auto",
        )

        html = render_index_html(settings)

        self.assertIn("/api/ask", html)
        self.assertIn("/api/query", html)
        self.assertIn("bge-m3", html)
        self.assertIn("qwen3.5:9b", html)
        self.assertIn("咲季为什么害怕输给佑芽？", html)

    def test_parse_top_k_and_backend_validate_ranges(self):
        self.assertEqual(parse_top_k("", 4), 4)
        self.assertEqual(parse_top_k("6", 4), 6)
        self.assertEqual(parse_backend("", "auto"), "auto")
        self.assertEqual(parse_backend("chroma", "auto"), "chroma")
        with self.assertRaises(ValueError):
            parse_top_k("0", 4)
        with self.assertRaises(ValueError):
            parse_backend("sqlite", "auto")


if __name__ == "__main__":
    unittest.main()
