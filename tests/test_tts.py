import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from crpf.config import ProjectConfig, TTSConfig, load_config
from crpf.tts import (
    clean_translated_voice_text,
    normalize_tts_text,
    synthesize_saki_tts,
    tts_cache_filename,
    translation_cache_path,
)


class TTSTests(unittest.TestCase):
    def test_normalize_tts_text_compacts_and_limits(self):
        self.assertEqual(normalize_tts_text("  制作人\n今天  加油  ", 6), "制作人 今天")

    def test_cache_filename_is_stable_wav_name(self):
        config = TTSConfig(media_type="wav")

        filename = tts_cache_filename("制作人，今天也要打起精神来。", config)

        self.assertTrue(filename.startswith("saki_"))
        self.assertTrue(filename.endswith(".wav"))
        self.assertEqual(filename, tts_cache_filename("制作人，今天也要打起精神来。", config))

    def test_clean_translated_voice_text_removes_labels_and_quotes(self):
        self.assertEqual(
            clean_translated_voice_text("日本語：「プロデューサー、今日も頑張るわよ！」", 80),
            "プロデューサー、今日も頑張るわよ！",
        )

    def test_synthesize_saki_tts_writes_project_cache_and_reuses_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "GPT-SoVITS" / "outputs"
            config = ProjectConfig(
                tts=TTSConfig(
                    output_dir=output_dir,
                    ref_audio_path=Path(tmp) / "ref.wav",
                    timeout_seconds=3,
                    text_lang="zh",
                    translate_to_japanese=False,
                )
            )
            fake_response = Mock()
            fake_response.__enter__ = Mock(return_value=fake_response)
            fake_response.__exit__ = Mock(return_value=False)
            fake_response.read.return_value = b"RIFF" + (b"\0" * 128)

            with patch("urllib.request.urlopen", return_value=fake_response) as urlopen:
                first = synthesize_saki_tts(config, "制作人，今天也要打起精神来。")
                second = synthesize_saki_tts(config, "制作人，今天也要打起精神来。")

            self.assertTrue(first.path.exists())
            self.assertEqual(first.path.parent, output_dir)
            self.assertFalse(first.cached)
            self.assertTrue(second.cached)
            self.assertFalse(first.translated)
            self.assertEqual(first.path, second.path)
            self.assertEqual(urlopen.call_count, 1)
            self.assertTrue(first.metadata_path.exists())

    def test_synthesize_saki_tts_translates_to_japanese_before_voice(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "GPT-SoVITS" / "outputs"
            config = ProjectConfig(
                tts=TTSConfig(
                    output_dir=output_dir,
                    ref_audio_path=Path(tmp) / "ref.wav",
                    timeout_seconds=3,
                    text_lang="ja",
                    translate_to_japanese=True,
                    translation_model="qwen3.5:9b",
                )
            )
            translation_path = translation_cache_path("制作人，今天也要打起精神来。", config)
            translation_path.parent.mkdir(parents=True, exist_ok=True)
            translation_path.write_text(
                '{"voice_text": "プロデューサー、今日も元気出していくわよ！"}',
                encoding="utf-8",
            )
            fake_response = Mock()
            fake_response.__enter__ = Mock(return_value=fake_response)
            fake_response.__exit__ = Mock(return_value=False)
            fake_response.read.return_value = b"RIFF" + (b"\0" * 128)

            with patch("urllib.request.urlopen", return_value=fake_response) as urlopen:
                result = synthesize_saki_tts(config, "制作人，今天也要打起精神来。")

            body = urlopen.call_args.args[0].data.decode("utf-8")
            self.assertIn("プロデューサー、今日も元気出していくわよ！", body)
            self.assertIn('"text_lang": "ja"', body)
            self.assertTrue(result.translated)
            self.assertEqual(result.voice_text, "プロデューサー、今日も元気出していくわよ！")

    def test_load_config_parses_tts_section(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text(
                """
llm:
  provider: openai_compatible
  model: api-chat
  base_url: https://api.example.test/v1
  api_key_env: TEST_CHAT_KEY
embedding:
  provider: openai_compatible
  model: api-embed
  base_url: https://api.example.test/v1
  api_key_env: TEST_EMBED_KEY
tts:
  enabled: true
  provider: gpt_sovits
  base_url: http://127.0.0.1:9880
  output_dir: GPT-SoVITS/outputs
  ref_audio_path: GPT-SoVITS/5-wav32k/sud_vo_adv_dear_hski_001_hski-033.wav
  prompt_lang: ja
  text_lang: ja
  translate_to_japanese: true
  translation_provider: ollama
  translation_model: qwen3.5:9b
  translation_api_key_env: TEST_LLM_KEY
  speed_factor: 1.0
""".strip(),
                encoding="utf-8",
            )

            config = load_config(path)

            self.assertEqual(config.llm.provider, "openai_compatible")
            self.assertEqual(config.llm.model, "api-chat")
            self.assertEqual(config.llm.api_key_env, "TEST_CHAT_KEY")
            self.assertEqual(config.embedding.provider, "openai_compatible")
            self.assertEqual(config.embedding.model, "api-embed")
            self.assertEqual(config.embedding.api_key_env, "TEST_EMBED_KEY")
            self.assertTrue(config.tts.enabled)
            self.assertEqual(config.tts.provider, "gpt_sovits")
            self.assertEqual(config.tts.text_lang, "ja")
            self.assertTrue(config.tts.translate_to_japanese)
            self.assertEqual(config.tts.translation_provider, "ollama")
            self.assertEqual(config.tts.translation_model, "qwen3.5:9b")
            self.assertEqual(config.tts.translation_api_key_env, "TEST_LLM_KEY")
            self.assertEqual(config.tts.output_dir, Path("GPT-SoVITS/outputs"))


if __name__ == "__main__":
    unittest.main()
