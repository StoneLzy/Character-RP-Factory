from __future__ import annotations

import re
from typing import Iterable


STANDARD_FIELDS = [
    "line_id",
    "speaker",
    "text",
    "translation",
    "chapter",
    "scene",
    "order",
    "source_file",
    "source_row",
    "source_kind",
]


def normalize_text(value: object) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\ufeff", "")
    text = text.replace("\\n", "\n")
    text = re.sub(r"[ \t　]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def standardize_row(row: dict[str, object], aliases: dict[str, list[str]]) -> dict[str, str]:
    lowered = {str(key).strip().lower(): key for key in row}
    output: dict[str, str] = {}
    for field, names in aliases.items():
        output[field] = ""
        for name in names:
            original_key = lowered.get(str(name).strip().lower())
            if original_key is not None:
                output[field] = normalize_text(row.get(original_key, ""))
                break

    for metadata in ("source_file", "source_row", "source_kind"):
        output[metadata] = normalize_text(row.get(metadata, ""))

    if not output.get("chapter"):
        output["chapter"] = infer_chapter(output.get("source_file", ""))
    if not output.get("scene"):
        output["scene"] = infer_scene(output.get("source_file", ""))
    if not output.get("order"):
        output["order"] = output.get("source_row", "")
    return output


def infer_chapter(source_file: str) -> str:
    parts = source_file.split("/")
    if len(parts) >= 3:
        return "/".join(parts[-3:-1])
    return ""


def infer_scene(source_file: str) -> str:
    name = source_file.rsplit("/", 1)[-1]
    return name.removesuffix(".csv")


def is_system_or_noise(row: dict[str, str], excluded_speakers: Iterable[str], bad_patterns: Iterable[str]) -> bool:
    speaker = normalize_text(row.get("speaker", ""))
    text = normalize_text(row.get("text", ""))
    translation = normalize_text(row.get("translation", ""))

    if speaker in set(excluded_speakers):
        return True
    if not speaker and not text and not translation:
        return True
    if not text and not translation:
        return True

    haystack = "\n".join([speaker, text, translation])
    return any(re.search(pattern, haystack) for pattern in bad_patterns)


def clean_rows(
    rows: Iterable[dict[str, object]],
    aliases: dict[str, list[str]],
    excluded_speakers: Iterable[str],
    bad_patterns: Iterable[str],
) -> list[dict[str, str]]:
    cleaned: list[dict[str, str]] = []
    seen: set[tuple[str, str, str, str]] = set()
    excluded = tuple(excluded_speakers)
    patterns = tuple(pattern for pattern in bad_patterns if pattern)

    for row in rows:
        standardized = standardize_row(row, aliases)
        if is_system_or_noise(standardized, excluded, patterns):
            continue
        dedupe_key = (
            standardized.get("source_file", ""),
            standardized.get("speaker", ""),
            standardized.get("text", ""),
            standardized.get("translation", ""),
        )
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        cleaned.append(standardized)
    return cleaned


def is_target_speaker(speaker: str, target_names: Iterable[str]) -> bool:
    normalized = normalize_text(speaker).lower()
    return any(name and normalize_text(name).lower() in normalized for name in target_names)


def filter_character_lines(
    rows: Iterable[dict[str, str]],
    target_names: Iterable[str],
    min_chars: int,
    max_chars: int,
) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    for row in rows:
        response = row.get("text") or row.get("translation") or ""
        length = len(normalize_text(response))
        if not is_target_speaker(row.get("speaker", ""), target_names):
            continue
        if length < min_chars or length > max_chars:
            continue
        selected.append(row)
    return selected
