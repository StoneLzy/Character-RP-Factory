from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

from .cleaning import normalize_text


SYSTEM_PROMPT = (
    "你正在扮演《学园偶像大师》的花海咲季。"
    "保持自信、明快、好胜、率直的语气，称呼用户为制作人。"
    "学习语气和互动节奏，不要大段复读原作台词。"
)

ZH_SYSTEM_PROMPT = (
    "你正在扮演《学园偶像大师》的花海咲季。"
    "请全程使用简体中文回复，保持自信、明快、好胜、率直的语气，称呼用户为制作人。"
    "学习语气和互动节奏，不要大段复读原作台词。"
)

JA_SYSTEM_PROMPT = (
    "あなたは『学園アイドルマスター』の花海咲季として振る舞います。"
    "自信家で明るく、負けず嫌いで率直な口調を保ち、ユーザーをプロデューサーとして扱ってください。"
    "原作台詞の長い再現ではなく、話し方と掛け合いのリズムを学習対象にします。"
)

KANA_RE = re.compile(r"[ぁ-ゟ゠-ヿ]")
RUBY_RE = re.compile(r"<r\\?=[^>]+>(.*?)</r>")
TAG_RE = re.compile(r"</?[^>]+>")


def export_jsonl(
    rows: Iterable[dict[str, str]],
    output_path: Path,
    export_format: str,
    max_context_chars: int,
    language: str = "ja",
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            if normalize_text(row.get("keep", "yes")).lower() in {"no", "false", "0"}:
                continue
            context, response = pick_language_fields(row, language)
            if language == "zh" and (contains_kana(context) or contains_kana(response)):
                continue
            if not response:
                continue
            record = build_record(
                row,
                export_format=export_format,
                max_context_chars=max_context_chars,
                language=language,
                context=context,
                response=response,
            )
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def pick_language_fields(row: dict[str, str], language: str) -> tuple[str, str]:
    if language == "zh":
        return (
            sanitize_chinese_text(row.get("context_translation", "")),
            sanitize_chinese_text(row.get("response_translation", "")),
        )
    return normalize_text(row.get("context", "")), normalize_text(row.get("response", ""))


def build_record(
    row: dict[str, str],
    export_format: str,
    max_context_chars: int,
    language: str,
    context: str | None = None,
    response: str | None = None,
) -> dict[str, object]:
    context = truncate_context(normalize_text(context if context is not None else row.get("context", "")), max_context_chars)
    response = normalize_text(response if response is not None else row.get("response", ""))
    metadata = {
        "sample_id": row.get("sample_id", ""),
        "source_file": row.get("source_file", ""),
        "source_row": row.get("source_row", ""),
        "quality_score": row.get("quality_score", ""),
        "language": language,
    }

    if export_format == "instruction":
        return {
            "instruction": instruction_text(language),
            "input": context,
            "output": response,
            "metadata": metadata,
        }

    return {
        "messages": [
            {"role": "system", "content": system_prompt(language)},
            {"role": "user", "content": build_user_content(context, language)},
            {"role": "assistant", "content": response},
        ],
        "metadata": metadata,
    }


def system_prompt(language: str) -> str:
    if language == "zh":
        return ZH_SYSTEM_PROMPT
    if language == "ja":
        return JA_SYSTEM_PROMPT
    return SYSTEM_PROMPT


def instruction_text(language: str) -> str:
    if language == "zh":
        return "根据中文上下文，以花海咲季的身份用简体中文回复制作人或当前对话对象。"
    return "次の文脈に続けて、花海咲季として自然に返答してください。"


def build_user_content(context: str, language: str) -> str:
    if not context:
        return "请以花海咲季的身份用简体中文自然回应。" if language == "zh" else "花海咲季として自然に返答してください。"
    if language == "zh":
        return f"下面是当前剧情对话的中文上下文，请以花海咲季的身份用简体中文自然接下一句。\n\n{context}"
    return f"次の会話文脈に続けて、花海咲季として自然に次の一言を返してください。\n\n{context}"


def truncate_context(context: str, max_context_chars: int) -> str:
    if max_context_chars <= 0 or len(context) <= max_context_chars:
        return context
    return context[-max_context_chars:].lstrip()


def sanitize_chinese_text(value: object) -> str:
    text = normalize_text(value)
    text = RUBY_RE.sub(r"\1", text)
    text = TAG_RE.sub("", text)
    return normalize_text(text)


def contains_kana(value: object) -> bool:
    return bool(KANA_RE.search(normalize_text(value)))
