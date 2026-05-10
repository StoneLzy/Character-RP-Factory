from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Iterable

from .cleaning import normalize_text
from .context_builder import CONTEXT_SAMPLE_FIELDS


REVIEW_FIELDS = [
    *CONTEXT_SAMPLE_FIELDS,
    "quality_score",
    "quality_reason",
    "keep",
    "notes",
    "emotion",
    "intent",
    "tone",
    "relationship",
]


SYSTEM_PATTERNS = (
    "点击继续",
    "任务完成",
    "获得道具",
    "ログインボーナス",
    "ロード中",
)


def score_samples(
    samples: Iterable[dict[str, str]],
    min_response_chars: int,
    max_response_chars: int,
    min_quality_score: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows = [dict(sample) for sample in samples]
    response_counts = Counter(normalize_for_duplicate(row.get("response", "")) for row in rows)
    scored: list[dict[str, object]] = []
    good: list[dict[str, object]] = []

    for row in rows:
        score, reasons = score_one_sample(
            row,
            response_counts=response_counts,
            min_response_chars=min_response_chars,
            max_response_chars=max_response_chars,
        )
        output = {
            **row,
            "quality_score": score,
            "quality_reason": "; ".join(reasons) if reasons else "ok",
            "keep": "yes" if score >= min_quality_score else "review",
            "notes": "",
            "emotion": "",
            "intent": "",
            "tone": "",
            "relationship": "",
        }
        scored.append(output)
        if score >= min_quality_score:
            good.append(output)
    return good, scored


def score_one_sample(
    row: dict[str, str],
    response_counts: Counter[str],
    min_response_chars: int,
    max_response_chars: int,
) -> tuple[int, list[str]]:
    response = normalize_text(row.get("response", ""))
    context = normalize_text(row.get("context", ""))
    score = 100
    reasons: list[str] = []

    response_len = len(response)
    if response_len < min_response_chars:
        score -= 35
        reasons.append("too_short")
    if response_len > max_response_chars:
        score -= 35
        reasons.append("too_long")
    if not context:
        score -= 40
        reasons.append("no_context")
    elif _to_int(row.get("context_line_count", 0)) < 2:
        score -= 8
        reasons.append("thin_context")
    if is_punctuation_only(response):
        score -= 35
        reasons.append("punctuation_only")
    if looks_like_system_text(response):
        score -= 45
        reasons.append("system_text")
    if response and response in context:
        score -= 15
        reasons.append("response_already_in_context")

    duplicate_key = normalize_for_duplicate(response)
    if duplicate_key and response_counts[duplicate_key] > 5:
        score -= 15
        reasons.append("frequent_duplicate_response")

    return max(score, 0), reasons


def normalize_for_duplicate(value: object) -> str:
    return re.sub(r"\s+", "", normalize_text(value)).lower()


def is_punctuation_only(value: object) -> bool:
    text = normalize_text(value)
    if not text:
        return True
    for char in text:
        category = unicodedata.category(char)
        if category[0] in {"L", "N"}:
            return False
        if "\u3040" <= char <= "\u30ff" or "\u4e00" <= char <= "\u9fff":
            return False
    return True


def looks_like_system_text(value: object) -> bool:
    text = normalize_text(value)
    return any(pattern in text for pattern in SYSTEM_PATTERNS)


def _to_int(value: object) -> int:
    try:
        return int(str(value))
    except ValueError:
        return 0
