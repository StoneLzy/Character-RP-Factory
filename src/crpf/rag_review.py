from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Iterable


REVIEW_FIELDS = [
    "scene_id",
    "source_file",
    "source_ref",
    "source_kind",
    "chapter",
    "scene",
    "saki_presence",
    "summary_status",
    "title",
    "summary",
    "topics",
    "speakers",
    "line_count",
    "story_facts",
    "relationship_facts",
    "worldbuilding_facts",
    "character_arc",
    "dialogue_style",
    "key_evidence",
    "risk_flags",
    "keep",
    "notes",
]


def build_rag_review_outputs(summary_output_dir: Path) -> dict[str, Path]:
    summaries_path = summary_output_dir / "scene_summaries.jsonl"
    summaries = load_jsonl(summaries_path)
    if not summaries:
        raise FileNotFoundError(f"No scene summaries found: {summaries_path}")

    report_path = summary_output_dir / "rag_validation_report.md"
    review_path = summary_output_dir / "rag_review.csv"
    report_path.write_text(render_validation_report(summaries, summaries_path), encoding="utf-8")
    write_review_csv(review_path, summaries)
    return {
        "validation_report": report_path,
        "review_csv": review_path,
    }


def load_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
    return rows


def write_review_csv(path: Path, summaries: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        for row in summaries:
            writer.writerow(review_row(row))


def review_row(row: dict[str, object]) -> dict[str, object]:
    return {
        "scene_id": row.get("scene_id", ""),
        "source_file": row.get("source_file", ""),
        "source_ref": first(row.get("source_refs")),
        "source_kind": row.get("source_kind", ""),
        "chapter": row.get("chapter", ""),
        "scene": row.get("scene", ""),
        "saki_presence": row.get("saki_presence", ""),
        "summary_status": row.get("summary_status", ""),
        "title": row.get("title", ""),
        "summary": row.get("summary", ""),
        "topics": join_list(row.get("topics")),
        "speakers": join_list(row.get("speakers")),
        "line_count": row.get("line_count", ""),
        "story_facts": join_list(row.get("story_facts")),
        "relationship_facts": join_list(row.get("relationship_facts")),
        "worldbuilding_facts": join_list(row.get("worldbuilding_facts")),
        "character_arc": join_list(row.get("character_arc")),
        "dialogue_style": join_list(row.get("dialogue_style")),
        "key_evidence": format_evidence(row),
        "risk_flags": join_list(risk_flags(row)),
        "keep": "",
        "notes": "",
    }


def render_validation_report(summaries: list[dict[str, object]], summaries_path: Path) -> str:
    total = len(summaries)
    status = Counter(str(row.get("summary_status") or "unknown") for row in summaries)
    presence = Counter(str(row.get("saki_presence") or "unknown") for row in summaries)
    source_kind = Counter(str(row.get("source_kind") or "unknown") for row in summaries)
    chapters = Counter(str(row.get("chapter") or "unknown") for row in summaries)
    topics = Counter(topic for row in summaries for topic in as_list(row.get("topics")))
    speakers = Counter(speaker for row in summaries for speaker in as_list(row.get("speakers")))
    risks = Counter(flag for row in summaries for flag in risk_flags(row))
    failed = [row for row in summaries if row.get("summary_status") == "llm_failed_fallback_heuristic" or row.get("llm_error")]
    unknown_presence = [row for row in summaries if not row.get("saki_presence")]
    empty_summary = [row for row in summaries if not row.get("summary")]
    empty_evidence = [row for row in summaries if not row.get("key_evidence")]

    lines = [
        "# RAG 场景摘要验证报告",
        "",
        "## 结论",
        "",
        f"- 输入文件：`{summaries_path}`",
        f"- 场景摘要总数：{total}",
        f"- LLM 成功摘要：{status.get('llm_fast_scene_card', 0) + status.get('llm_scene_card', 0)}",
        f"- fallback / 错误摘要：{len(failed)}",
        f"- 空摘要：{len(empty_summary)}",
        f"- 空证据：{len(empty_evidence)}",
        f"- 缺失咲季出场状态：{len(unknown_presence)}",
        "",
        "## 摘要状态分布",
        "",
        *counter_lines(status),
        "",
        "## 咲季出场状态分布",
        "",
        *counter_lines(presence),
        "",
        "## 来源分布",
        "",
        *counter_lines(source_kind, limit=20),
        "",
        "## 高频章节",
        "",
        *counter_lines(chapters, limit=30),
        "",
        "## 主题分布",
        "",
        *counter_lines(topics, limit=20),
        "",
        "## 高频参与者",
        "",
        *counter_lines(speakers, limit=30),
        "",
        "## 风险标记",
        "",
        *counter_lines(risks, empty="- 未发现自动风险标记。"),
        "",
        "## 抽样预览",
        "",
        *sample_lines("direct", summaries),
        "",
        *sample_lines("mentioned", summaries),
        "",
        *sample_lines("background", summaries),
        "",
        "## 人工审核建议",
        "",
        "- 优先审核 `risk_flags` 非空的行。",
        "- `direct` 场景优先进入角色剧情与口吻 RAG。",
        "- `mentioned` 场景更适合进入人物关系、外部评价和团队剧情 RAG。",
        "- `background` 场景建议人工判断是否保留，避免稀释角色知识库。",
        "- 审核 CSV 的 `keep` 可填写 `yes/no/rewrite`，`notes` 写修订意见。",
    ]
    if failed:
        lines.extend(["", "## 失败场景", ""])
        for row in failed:
            lines.append(f"- `{row.get('scene_id')}` `{row.get('source_file')}`：{row.get('llm_error')}")
    return "\n".join(lines).rstrip() + "\n"


def risk_flags(row: dict[str, object]) -> list[str]:
    flags: list[str] = []
    status = str(row.get("summary_status") or "")
    presence = str(row.get("saki_presence") or "")
    speakers = {str(speaker) for speaker in as_list(row.get("speakers"))}
    if status not in {"llm_fast_scene_card", "llm_scene_card"}:
        flags.append("bad_status")
    if not presence:
        flags.append("missing_presence")
    if presence == "direct" and "咲季" not in speakers:
        flags.append("direct_without_saki_speaker")
    if presence == "background":
        flags.append("background_review")
    if not row.get("story_facts"):
        flags.append("missing_story_facts")
    if not row.get("key_evidence"):
        flags.append("missing_evidence")
    summary = str(row.get("summary") or "")
    if len(summary) < 40:
        flags.append("short_summary")
    if row.get("llm_error"):
        flags.append("llm_error")
    return flags


def sample_lines(presence: str, summaries: list[dict[str, object]], limit: int = 3) -> list[str]:
    rows = [row for row in summaries if row.get("saki_presence") == presence][:limit]
    if not rows:
        return [f"### {presence}", "", "- 无"]
    lines = [f"### {presence}", ""]
    for row in rows:
        lines.append(f"- `{row.get('scene_id')}` {row.get('title')}：{trim(str(row.get('summary') or ''), 120)}")
    return lines


def counter_lines(counter: Counter, limit: int | None = None, empty: str = "- 无") -> list[str]:
    if not counter:
        return [empty]
    items = counter.most_common(limit)
    return [f"- {key}: {value}" for key, value in items]


def format_evidence(row: dict[str, object]) -> str:
    evidence_rows = []
    source_file = str(row.get("source_file") or "")
    for evidence in as_list(row.get("key_evidence")):
        if not isinstance(evidence, dict):
            continue
        source_row = evidence.get("source_row", "")
        speaker = evidence.get("speaker", "")
        text = evidence.get("text_zh", "")
        evidence_rows.append(f"{source_file}:{source_row} {speaker}: {text}")
    return " | ".join(evidence_rows)


def first(value: object) -> object:
    values = as_list(value)
    return values[0] if values else ""


def as_list(value: object) -> list[object]:
    return value if isinstance(value, list) else []


def join_list(value: object) -> str:
    return "；".join(str(item) for item in as_list(value))


def trim(text: str, limit: int) -> str:
    text = text.replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"
