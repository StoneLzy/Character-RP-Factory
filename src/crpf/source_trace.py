from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any


SCENE_ID_RE = re.compile(r"scene-\d{5}")


def build_source_trace(scene_id: str, rag_docs_dir: Path, output_dir: Path) -> dict[str, Any]:
    normalized_scene_id = normalize_scene_id(scene_id)
    if not normalized_scene_id:
        raise ValueError("scene_id is required")

    scene_card_path = find_scene_card_path(normalized_scene_id, rag_docs_dir)
    summary = find_scene_jsonl_record(output_dir / "scene_summaries.jsonl", normalized_scene_id)
    chunk = find_scene_jsonl_record(output_dir / "scene_chunks.jsonl", normalized_scene_id)

    if scene_card_path is None and summary is None and chunk is None:
        raise ValueError(f"trace source not found: {normalized_scene_id}")

    card_text = scene_card_path.read_text(encoding="utf-8") if scene_card_path else ""
    dialogue_lines = extract_dialogue_lines(chunk or {})
    return {
        "scene_id": normalized_scene_id,
        "scene_card_path": scene_card_path.as_posix() if scene_card_path else "",
        "scene_card": card_text,
        "summary": compact_summary(summary or {}),
        "source_refs": (summary or chunk or {}).get("source_refs", []),
        "dialogue_lines": dialogue_lines,
        "dialogue_count": len(dialogue_lines),
    }


def normalize_scene_id(value: str) -> str:
    match = SCENE_ID_RE.search(str(value or ""))
    return match.group(0) if match else ""


def find_scene_card_path(scene_id: str, rag_docs_dir: Path) -> Path | None:
    scene_dir = rag_docs_dir / "scenes"
    if not scene_dir.exists():
        return None
    matches = sorted(scene_dir.glob(f"{scene_id}*.md"))
    return matches[0] if matches else None


def find_scene_jsonl_record(path: Path, scene_id: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    records = cached_jsonl_records(str(path))
    return records.get(scene_id)


@lru_cache(maxsize=4)
def cached_jsonl_records(path_text: str) -> dict[str, dict[str, Any]]:
    path = Path(path_text)
    records: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            scene_id = str(row.get("scene_id") or row.get("summary_id") or "")
            if scene_id:
                records[scene_id] = row
    return records


def compact_summary(row: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "scene_id",
        "title",
        "source_file",
        "chapter",
        "scene",
        "summary",
        "saki_role",
        "saki_presence",
        "speakers",
        "topics",
        "key_evidence",
        "story_facts",
        "relationship_facts",
        "worldbuilding_facts",
        "character_arc",
        "dialogue_style",
    ]
    return {key: row.get(key) for key in keys if key in row}


def extract_dialogue_lines(chunk: dict[str, Any]) -> list[dict[str, str]]:
    lines = chunk.get("dialogue_zh") or []
    if not isinstance(lines, list):
        return []
    output: list[dict[str, str]] = []
    for item in lines:
        if not isinstance(item, dict):
            continue
        output.append(
            {
                "source_row": str(item.get("source_row", "")),
                "speaker": str(item.get("speaker", "")),
                "text_zh": str(item.get("text_zh", "")),
                "text_ja": str(item.get("text_ja", "")),
            }
        )
    return output
