from __future__ import annotations

import hashlib
import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .config import ProjectConfig, TTSConfig


@dataclass(frozen=True)
class TTSResult:
    path: Path
    cached: bool
    voice_text: str
    translated: bool
    metadata_path: Path


def synthesize_saki_tts(config: ProjectConfig, text: str) -> TTSResult:
    clean_text = normalize_tts_text(text, config.tts.max_text_chars)
    if not clean_text:
        raise ValueError("text is required")
    if not config.tts.enabled:
        raise ValueError("TTS is disabled in config")
    if config.tts.provider != "gpt_sovits":
        raise ValueError(f"Unsupported TTS provider: {config.tts.provider}")

    output_dir = config.tts.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    voice_text, translated = prepare_voice_text(config, clean_text, output_dir)
    audio_path = output_dir / tts_cache_filename(voice_text, config.tts)
    metadata_path = audio_path.with_suffix(".json")
    if audio_path.exists() and audio_path.stat().st_size > 44:
        write_tts_metadata(metadata_path, clean_text, voice_text, translated, config)
        return TTSResult(path=audio_path, cached=True, voice_text=voice_text, translated=translated, metadata_path=metadata_path)

    audio_bytes = request_gpt_sovits(config.tts, voice_text)
    if len(audio_bytes) <= 44:
        raise ValueError("GPT-SoVITS returned an empty audio file")
    tmp_path = audio_path.with_suffix(audio_path.suffix + ".tmp")
    tmp_path.write_bytes(audio_bytes)
    tmp_path.replace(audio_path)
    write_tts_metadata(metadata_path, clean_text, voice_text, translated, config)
    return TTSResult(path=audio_path, cached=False, voice_text=voice_text, translated=translated, metadata_path=metadata_path)


def normalize_tts_text(text: str, max_chars: int) -> str:
    clean = re.sub(r"\s+", " ", text).strip()
    if max_chars > 0 and len(clean) > max_chars:
        clean = clean[:max_chars].rstrip()
    return clean


def tts_cache_filename(text: str, config: TTSConfig) -> str:
    key = "\n".join(
        [
            text,
            config.provider,
            config.base_url,
            str(config.ref_audio_path),
            config.prompt_text,
            config.prompt_lang,
            config.text_lang,
            config.text_split_method,
            config.media_type,
            str(config.batch_size),
            str(config.speed_factor),
        ]
    )
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
    extension = config.media_type.strip(".").lower() or "wav"
    return f"saki_{digest}.{extension}"


def prepare_voice_text(config: ProjectConfig, text: str, output_dir: Path) -> tuple[str, bool]:
    if should_translate_to_japanese(config.tts):
        return translate_voice_text_to_japanese(config, text, output_dir), True
    return text, False


def should_translate_to_japanese(config: TTSConfig) -> bool:
    return config.translate_to_japanese and config.text_lang.lower() in {"ja", "jp", "japanese"}


def translate_voice_text_to_japanese(config: ProjectConfig, text: str, output_dir: Path) -> str:
    cache_path = translation_cache_path(text, config)
    if cache_path.exists():
        try:
            raw = json.loads(cache_path.read_text(encoding="utf-8"))
            cached_text = normalize_tts_text(str(raw.get("voice_text", "")), config.tts.max_text_chars)
            if cached_text:
                return cached_text
        except (OSError, json.JSONDecodeError):
            pass

    model = config.tts.translation_model or config.rag.chat_model
    base_url = config.tts.translation_base_url or config.rag.ollama_base_url
    voice_text = request_ollama_voice_translation(
        text=text,
        model=model,
        ollama_base_url=base_url,
        timeout=config.tts.translation_timeout_seconds,
        max_chars=config.tts.max_text_chars,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {
                "source_text": text,
                "voice_text": voice_text,
                "translation_model": model,
                "translation_base_url": base_url,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return voice_text


def translation_cache_path(text: str, config: ProjectConfig) -> Path:
    model = config.tts.translation_model or config.rag.chat_model
    base_url = config.tts.translation_base_url or config.rag.ollama_base_url
    key = "\n".join([text, model, base_url, "saki-ja-voice-v1"])
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
    return config.tts.output_dir / f"saki_voice_text_{digest}.json"


def request_ollama_voice_translation(
    text: str,
    model: str,
    ollama_base_url: str,
    timeout: int,
    max_chars: int,
) -> str:
    url = ollama_base_url.rstrip("/") + "/api/chat"
    prompt = f"""把下面这段中文回复改写成适合语音合成的日语台词。

要求：
- 只输出日语台词本身，不要解释，不要加引号。
- 说话人是《学园偶像大师》的花海咲季。
- 称呼用户为「プロデューサー」。
- 保持咲季积极、认真、自信、向前冲的语气；不要写成傲娇。
- 不要改变事实，不要增加新剧情。
- 如果中文里提到训练，训练对象应是咲季/偶像侧，不要变成让プロデューサー去训练。
- 口语自然，适合朗读；过长时可以压缩，但不要漏掉核心意思。

中文回复：
{text}
"""
    payload = {
        "model": model,
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0.15,
            "top_p": 0.9,
            "num_ctx": 4096,
            "num_predict": max(120, min(700, max_chars * 2)),
        },
        "messages": [
            {
                "role": "system",
                "content": "あなたは日中翻訳者です。花海咲季らしい自然な日本語台詞だけを出力します。",
            },
            {"role": "user", "content": prompt},
        ],
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=timeout) as response:
            raw = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise ValueError(f"Ollama voice translation failed: {exc}") from exc

    content = str(raw.get("message", {}).get("content", "")).strip()
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    content = clean_translated_voice_text(content, max_chars=max_chars)
    if not content:
        raise ValueError(f"Ollama voice translation is empty: {raw}")
    return content


def clean_translated_voice_text(text: str, max_chars: int) -> str:
    clean = normalize_tts_text(text, max_chars)
    clean = re.sub(r"^(日语台词|日本語|翻訳|台詞)[:：]\s*", "", clean).strip()
    clean = clean.strip("\"'「」『』")
    return clean


def write_tts_metadata(
    path: Path,
    source_text: str,
    voice_text: str,
    translated: bool,
    config: ProjectConfig,
) -> None:
    path.write_text(
        json.dumps(
            {
                "source_text": source_text,
                "voice_text": voice_text,
                "translated": translated,
                "text_lang": config.tts.text_lang,
                "prompt_lang": config.tts.prompt_lang,
                "provider": config.tts.provider,
                "translation_model": config.tts.translation_model or config.rag.chat_model if translated else "",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def request_gpt_sovits(config: TTSConfig, text: str) -> bytes:
    endpoint = config.base_url.rstrip("/") + "/tts"
    payload = {
        "text": text,
        "text_lang": config.text_lang,
        "ref_audio_path": str(config.ref_audio_path.resolve()),
        "prompt_text": config.prompt_text,
        "prompt_lang": config.prompt_lang,
        "text_split_method": config.text_split_method,
        "batch_size": config.batch_size,
        "media_type": config.media_type,
        "streaming_mode": False,
        "speed_factor": config.speed_factor,
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=config.timeout_seconds) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise ValueError(f"GPT-SoVITS HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise ValueError(f"Cannot connect to GPT-SoVITS: {exc.reason}") from exc
