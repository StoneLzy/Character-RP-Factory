from __future__ import annotations

from collections import defaultdict, deque
from typing import Iterable

from .cleaning import is_target_speaker, normalize_text


CONTEXT_SAMPLE_FIELDS = [
    "sample_id",
    "source_file",
    "source_row",
    "source_kind",
    "chapter",
    "scene",
    "context",
    "context_translation",
    "context_line_count",
    "response",
    "response_translation",
    "response_len",
    "speaker",
]


def build_context_samples(
    rows: Iterable[dict[str, str]],
    target_names: Iterable[str],
    previous_lines: int,
) -> list[dict[str, object]]:
    """Build target-character samples using full cleaned dialogue as context.

    The cleaned dialogue is the source of truth. ``character_lines.csv`` is only
    an index of candidate responses, because using it for context would drop
    producer lines, other characters, and choice rows.
    """
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row.get("source_file", "")].append(row)

    samples: list[dict[str, object]] = []
    for source_file in sorted(grouped):
        source_rows = sorted(grouped[source_file], key=_row_number)
        history: deque[dict[str, str]] = deque(maxlen=max(previous_lines, 0))
        for row in source_rows:
            if is_target_speaker(row.get("speaker", ""), target_names):
                sample_id = f"{len(samples) + 1:08d}"
                context_rows = list(history)
                response = normalize_text(row.get("text", ""))
                response_translation = normalize_text(row.get("translation", ""))
                samples.append(
                    {
                        "sample_id": sample_id,
                        "source_file": row.get("source_file", ""),
                        "source_row": row.get("source_row", ""),
                        "source_kind": row.get("source_kind", ""),
                        "chapter": row.get("chapter", ""),
                        "scene": row.get("scene", ""),
                        "context": format_context(context_rows, language="ja"),
                        "context_translation": format_context(context_rows, language="zh"),
                        "context_line_count": len(context_rows),
                        "response": response,
                        "response_translation": response_translation,
                        "response_len": len(response or response_translation),
                        "speaker": row.get("speaker", ""),
                    }
                )
            history.append(row)
    return samples


def format_context(rows: Iterable[dict[str, str]], language: str) -> str:
    lines: list[str] = []
    for row in rows:
        text_key = "translation" if language == "zh" else "text"
        text = normalize_text(row.get(text_key, ""))
        if not text:
            continue
        speaker = display_speaker(row, language=language)
        lines.append(f"{speaker}: {text}")
    return "\n".join(lines)


SPEAKER_TRANSLATIONS = {
    "{user}": "制作人",
    "プロデューサー": "制作人",
    "咲季": "咲季",
    "花海咲季": "花海咲季",
    "ことね": "琴音",
    "藤田ことね": "藤田琴音",
    "手毬": "手毬",
    "月村手毬": "月村手毬",
    "星南": "星南",
    "十王星南": "十王星南",
    "佑芽": "佑芽",
    "花海佑芽": "花海佑芽",
    "千奈": "千奈",
    "倉本千奈": "仓本千奈",
    "莉波": "莉波",
    "姫崎莉波": "姬崎莉波",
    "清夏": "清夏",
    "紫雲清夏": "紫云清夏",
    "広": "广",
    "篠澤広": "篠泽广",
    "麻央": "麻央",
    "有村麻央": "有村麻央",
    "リーリヤ": "莉莉娅",
    "葛城リーリヤ": "葛城莉莉娅",
    "美鈴": "美铃",
    "秦谷美鈴": "秦谷美铃",
    "燕": "燕",
    "あさり先生": "浅梨老师",
    "ダンストレーナー": "舞蹈训练师",
    "ビジュアルトレーナー": "形象训练师",
    "ボーカルトレーナー": "声乐训练师",
    "学園長": "学园长",
    "一同": "众人",
    "？？？": "？？？",
    "__narration__": "旁白",
    "女の子": "女孩子",
    "男の子": "男孩子",
    "子供": "孩子",
    "ファン": "粉丝",
    "スタッフ": "工作人员",
}


def display_speaker(row: dict[str, str], language: str) -> str:
    speaker = normalize_text(row.get("speaker", ""))
    if language == "zh" and speaker:
        return SPEAKER_TRANSLATIONS.get(speaker, speaker)
    if speaker:
        return speaker
    if normalize_text(row.get("line_id", "")) == "select":
        return "选项"
    return "旁白"


def _row_number(row: dict[str, str]) -> tuple[int, str]:
    value = normalize_text(row.get("source_row", ""))
    try:
        return int(value), value
    except ValueError:
        return 0, value
