import json
import os
import unittest
from unittest.mock import patch

from crpf.providers import ModelProviderConfig, complete_chat, embed_texts, normalize_provider, stream_chat


class FakeResponse:
    def __init__(self, payload: dict | None = None, lines: list[bytes] | None = None):
        self.payload = payload or {}
        self.lines = lines or []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self.payload, ensure_ascii=False).encode("utf-8")

    def __iter__(self):
        return iter(self.lines)


class ProviderTests(unittest.TestCase):
    def test_normalize_provider_accepts_api_aliases(self):
        self.assertEqual(normalize_provider("ollama"), "ollama")
        self.assertEqual(normalize_provider("openai-compatible"), "openai_compatible")
        self.assertEqual(normalize_provider("api"), "openai_compatible")
        with self.assertRaises(ValueError):
            normalize_provider("made_up")

    def test_openai_compatible_chat_uses_chat_completions_shape(self):
        config = ModelProviderConfig(
            provider="openai_compatible",
            model="test-chat",
            base_url="https://api.example.test/v1",
            api_key_env="TEST_PROVIDER_KEY",
        )
        response = FakeResponse({"choices": [{"message": {"content": "  回答完成  "}}]})

        with patch.dict(os.environ, {"TEST_PROVIDER_KEY": "secret"}, clear=False):
            with patch("urllib.request.urlopen", return_value=response) as urlopen:
                content = complete_chat(config, "问题", "系统", timeout=3)

        request = urlopen.call_args.args[0]
        body = json.loads(request.data.decode("utf-8"))
        self.assertEqual(content, "回答完成")
        self.assertEqual(request.full_url, "https://api.example.test/v1/chat/completions")
        self.assertEqual(request.headers.get("Authorization"), "Bearer secret")
        self.assertEqual(body["model"], "test-chat")
        self.assertFalse(body["stream"])

    def test_openai_compatible_stream_yields_deltas(self):
        config = ModelProviderConfig(
            provider="openai_compatible",
            model="test-chat",
            base_url="https://api.example.test/v1",
        )
        response = FakeResponse(
            lines=[
                'data: {"choices":[{"delta":{"content":"你"}}]}\n\n'.encode("utf-8"),
                'data: {"choices":[{"delta":{"content":"好"}}]}\n\n'.encode("utf-8"),
                b"data: [DONE]\n\n",
            ]
        )

        with patch("urllib.request.urlopen", return_value=response):
            parts = list(stream_chat(config, "问题", "系统", timeout=3))

        self.assertEqual(parts, ["你", "好"])

    def test_openai_compatible_embeddings_keep_response_order(self):
        config = ModelProviderConfig(
            provider="openai_compatible",
            model="test-embed",
            base_url="https://api.example.test/v1",
        )
        response = FakeResponse(
            {
                "data": [
                    {"index": 1, "embedding": [0.3, 0.4]},
                    {"index": 0, "embedding": [0.1, 0.2]},
                ]
            }
        )

        with patch("urllib.request.urlopen", return_value=response) as urlopen:
            embeddings = embed_texts(config, ["甲", "乙"], batch_size=2, timeout=3)

        request = urlopen.call_args.args[0]
        body = json.loads(request.data.decode("utf-8"))
        self.assertEqual(request.full_url, "https://api.example.test/v1/embeddings")
        self.assertEqual(body["input"], ["甲", "乙"])
        self.assertEqual(embeddings, [[0.1, 0.2], [0.3, 0.4]])


if __name__ == "__main__":
    unittest.main()
