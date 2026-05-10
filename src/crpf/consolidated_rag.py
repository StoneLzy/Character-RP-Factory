from __future__ import annotations

import json
import re
import shutil
from collections import Counter
from pathlib import Path


CONSOLIDATED_DOCS = {
    "character_profile.md": {
        "title": "角色画像",
        "doc_type": "profile",
        "purpose": "咲季稳定人物画像、动机、性格、成长弧线、弱点和行为模式。",
        "retrieval": "回答咲季是什么样的人、为什么行动、怎样成长时优先检索。",
    },
    "plot_summary.md": {
        "title": "剧情概要",
        "doc_type": "plot",
        "purpose": "咲季相关主线、支线、章节事件和关键舞台整理。",
        "retrieval": "回答某段剧情发生了什么、某个事件如何影响咲季时优先检索。",
    },
    "relationships.md": {
        "title": "人物关系",
        "doc_type": "relationships",
        "purpose": "咲季与制作人、佑芽、琴音、手毬、星南等人物的稳定关系。",
        "retrieval": "回答角色关系、称呼、互动边界、外部评价时优先检索。",
    },
    "worldbuilding.md": {
        "title": "世界观设定",
        "doc_type": "worldbuilding",
        "purpose": "初星学园、宿舍、训练、选拔、H.I.F/N.I.A、偶像活动等设定。",
        "retrieval": "回答制度、地点、赛事、课程、工作设定时优先检索。",
    },
    "team_story.md": {
        "title": "团队剧情",
        "doc_type": "team_story",
        "purpose": "Re;IRIS、宿舍、班级、团队协作和竞争型友情。",
        "retrieval": "回答团队互动、组合冲突、共同生活、队友协作时优先检索。",
    },
    "dialogue_patterns.md": {
        "title": "对话风格",
        "doc_type": "dialogue",
        "purpose": "咲季中文口吻、称呼、情绪节奏、关系互动和 RP 写作边界。",
        "retrieval": "生成咲季回复、约束语气、避免 OOC 时优先检索。",
    },
}

TERM_NOTE = """## 入库注意事项

- `scene_id` 是证据索引，可回查 `outputs/scene_summaries.jsonl` 和 `data/rag_docs/scenes/`。
- `H.I.F`、`N.I.A`、`FINALE`、`一等星`、`Re;IRIS`、`Begrazia` 等术语保留原文；官方译名如需发布请二次核对。
- `求婚`、`禁忌之恋`、`疑似交往` 等内容仅作为特定剧情张力或误会，默认不要固化为长期恋爱设定。
- `姐姐/妹妹` 称呼仅在花海咲季与花海佑芽之间作为亲属关系处理；其他角色相关表达需按语境视为照顾、玩笑或称呼梗。
- AU/活动限定设定可用于性格投射，不应覆盖主线世界观。
"""


def sync_consolidated_rag(
    source_dir: Path,
    rag_docs_dir: Path,
    output_dir: Path,
) -> dict[str, Path]:
    source_docs = load_source_docs(source_dir)
    summaries = load_jsonl(output_dir / "scene_summaries.jsonl")
    known_scene_ids = {str(row.get("scene_id")) for row in summaries}

    rag_docs_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for filename, raw_text in source_docs.items():
        cleaned = clean_markdown(raw_text)
        cleaned = ensure_note_section(cleaned)
        target_path = rag_docs_dir / filename
        target_path.write_text(cleaned, encoding="utf-8")
        written[filename] = target_path

    index_path = rag_docs_dir / "index.md"
    index_path.write_text(render_index(rag_docs_dir, source_docs, summaries), encoding="utf-8")
    written["index"] = index_path

    validation_path = output_dir / "consolidated_validation_report.md"
    validation_path.write_text(
        render_validation_report(source_docs, summaries, known_scene_ids),
        encoding="utf-8",
    )
    written["validation_report"] = validation_path

    chunking_path = output_dir / "rag_chunking_plan.md"
    chunking_path.write_text(render_chunking_plan(), encoding="utf-8")
    written["chunking_plan"] = chunking_path
    return written


def load_source_docs(source_dir: Path) -> dict[str, str]:
    missing = [name for name in CONSOLIDATED_DOCS if not (source_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing consolidated output docs: {', '.join(missing)}")
    return {
        filename: (source_dir / filename).read_text(encoding="utf-8")
        for filename in CONSOLIDATED_DOCS
    }


def load_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
    return rows


def clean_markdown(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in normalized.split("\n")]
    lines = split_overlong_lines(lines)
    return "\n".join(lines).strip() + "\n"


def split_overlong_lines(lines: list[str], limit: int = 280) -> list[str]:
    cleaned: list[str] = []
    for line in lines:
        if len(line) <= limit or line.startswith("|"):
            cleaned.append(line)
            continue
        prefix = ""
        content = line
        for marker in ("- ", "> "):
            if line.startswith(marker):
                prefix = marker
                content = line[len(marker) :]
                break
        parts = split_sentence_like(content, limit - len(prefix))
        if not parts:
            cleaned.append(line)
            continue
        cleaned.append(prefix + parts[0])
        for part in parts[1:]:
            cleaned.append(("  " if prefix == "- " else prefix) + part)
    return cleaned


def split_sentence_like(text: str, limit: int) -> list[str]:
    if len(text) <= limit:
        return [text]
    pieces = re.split(r"(?<=[。；！？])", text)
    lines: list[str] = []
    current = ""
    for piece in pieces:
        if not piece:
            continue
        if current and len(current) + len(piece) > limit:
            lines.append(current)
            current = piece
        else:
            current += piece
    if current:
        lines.append(current)
    return lines if lines else [text]


def ensure_note_section(text: str) -> str:
    if "## 入库注意事项" in text:
        return text
    lines = text.split("\n")
    insert_at = 1
    while insert_at < len(lines) and (lines[insert_at].startswith(">") or not lines[insert_at].strip()):
        insert_at += 1
    return "\n".join(lines[:insert_at] + ["", TERM_NOTE.strip(), ""] + lines[insert_at:]).strip() + "\n"


def render_index(
    rag_docs_dir: Path,
    source_docs: dict[str, str],
    summaries: list[dict[str, object]],
) -> str:
    presence = Counter(str(row.get("saki_presence") or "unknown") for row in summaries)
    lines = [
        "# 花海咲季 RAG 文档索引",
        "",
        "## 文档层级",
        "",
        "- `data/rag_docs/*.md`：高层知识文档，适合召回角色画像、关系、剧情线、世界观和口吻规则。",
        "- `data/rag_docs/scenes/*.md`：500 张场景卡，适合追溯具体剧情和证据。",
        "- `outputs/scene_summaries.jsonl`：结构化场景摘要，可作为程序化 metadata 来源。",
        "",
        "## 数据状态",
        "",
        f"- 场景摘要数：{len(summaries)}",
        f"- 咲季出场状态：{format_counter(presence)}",
        f"- 场景卡目录：`{rag_docs_dir / 'scenes'}`",
        "",
        "## 文档说明",
        "",
        "| 文档 | 类型 | 用途 | 推荐检索场景 | scene_id 覆盖 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for filename, meta in CONSOLIDATED_DOCS.items():
        refs = extract_scene_ids(source_docs[filename])
        lines.append(
            f"| `{filename}` | {meta['doc_type']} | {meta['purpose']} | {meta['retrieval']} | {len(refs)} |"
        )
    lines.extend(
        [
            "",
            "## 检索建议",
            "",
            "- 角色扮演生成：优先检索 `character_profile.md` + `dialogue_patterns.md`，再按问题补充 `relationships.md` 或场景卡。",
            "- 剧情问答：优先检索 `plot_summary.md`，再用 `scene_id` 召回 `scenes/*.md` 追证据。",
            "- 人物关系问答：优先检索 `relationships.md`，关系不够细时召回对应场景卡。",
            "- 设定问答：优先检索 `worldbuilding.md`，避免把 AU/活动限定内容当主线设定。",
            "- 团队与 Re;IRIS 问答：优先检索 `team_story.md`。",
            "",
            "## 证据回查",
            "",
            "- 文档里的 `scene-xxxxx` 可在 `outputs/scene_summaries.jsonl` 中找到结构化摘要。",
            "- 对应 Markdown 场景卡位于 `data/rag_docs/scenes/scene-xxxxx_*.md`。",
            "- 如需原始 CSV 上下文，可用 `source_file` 和 `source_row` 回查 `CSV/`。",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def render_validation_report(
    source_docs: dict[str, str],
    summaries: list[dict[str, object]],
    known_scene_ids: set[str],
) -> str:
    lines = [
        "# Consolidated RAG 验证报告",
        "",
        "## 总览",
        "",
        f"- 正式摘要总数：{len(summaries)}",
        f"- 已检查文档数：{len(source_docs)}",
        "",
        "## 文档检查",
        "",
        "| 文档 | 行数 | 字符数 | 标题数 | scene_id 覆盖 | 坏引用 | 超长行 | 风险词 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    all_missing: list[tuple[str, str]] = []
    for filename, text in source_docs.items():
        refs = extract_scene_ids(text)
        missing = sorted(refs - known_scene_ids)
        all_missing.extend((filename, scene_id) for scene_id in missing)
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{filename}`",
                    str(text.count("\n") + 1),
                    str(len(text)),
                    str(len(re.findall(r"^#{1,6} ", text, flags=re.MULTILINE))),
                    str(len(refs)),
                    str(len(missing)),
                    str(count_long_lines(text)),
                    str(count_risk_words(text)),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## 结论",
            "",
            "- 所有文档已同步到 `data/rag_docs/`。",
            "- `scene_id` 坏引用为 0 时，可进入 RAG chunking 阶段。",
            "- `求婚`、`禁忌之恋`、`姐姐称呼梗`、AU 设定已作为入库注意事项处理，仍建议后续人工抽查。",
        ]
    )
    if all_missing:
        lines.extend(["", "## 坏引用", ""])
        for filename, scene_id in all_missing:
            lines.append(f"- `{filename}` references missing `{scene_id}`")
    return "\n".join(lines).rstrip() + "\n"


def render_chunking_plan() -> str:
    return """# RAG Chunking Plan

## 目标

把高层 RAG 文档与 500 张场景卡一起入库，形成“高层背景召回 + 精确剧情召回”的双层知识库。

## 文档切分规则

| 文档 | 推荐切分 | metadata.doc_type |
| --- | --- | --- |
| `character_profile.md` | 按二级/三级标题切；每个 chunk 保留当前标题链 | `profile` |
| `relationships.md` | 按人物关系标题切；每个关系单独 chunk | `relationships` |
| `plot_summary.md` | 按来源类型/章节/事件标题切 | `plot` |
| `worldbuilding.md` | 按设定条目和索引小节切 | `worldbuilding` |
| `team_story.md` | 按 Re;IRIS、宿舍、班级、关系条目切 | `team_story` |
| `dialogue_patterns.md` | 按口吻规则、称呼规则、互动对象切 | `dialogue` |
| `scenes/*.md` | 每张场景卡 1 个 chunk；长卡可按“场景理解/结构化线索/证据”拆 | `scene_card` |

## Metadata 字段

- `doc_type`: 上表类型。
- `source_doc`: Markdown 文件名。
- `heading_path`: 当前标题链。
- `scene_ids`: chunk 内出现的 `scene-xxxxx` 列表。
- `source_files`: 对场景卡从 `source_file` 抽取；对高层文档可留空。
- `saki_presence`: 场景卡使用 `direct/mentioned/background`；高层文档可留空或 mixed。
- `topics`: 可从 `outputs/scene_summaries.jsonl` 回填。

## 检索策略

- RP 回复：检索 `profile + dialogue + relationships`，再补 1-3 张 scene_card。
- 剧情问答：检索 `plot + scene_card`。
- 关系问答：检索 `relationships + scene_card`。
- 世界观问答：检索 `worldbuilding`，必要时补 `plot`。
- 团队剧情：检索 `team_story + relationships`。

## 注意

- 不要把长段原始台词入库为生成目标；保留短证据和 scene_id 即可。
- 遇到 `待核对` 或 AU/活动限定内容，生成时应降低置信度。
- 高层文档和 scene_card 都要保留，二者互补，不互相替代。
"""


def extract_scene_ids(text: str) -> set[str]:
    return set(re.findall(r"scene-\d{5}", text))


def count_long_lines(text: str, limit: int = 280) -> int:
    return sum(1 for line in text.splitlines() if len(line) > limit)


def count_risk_words(text: str) -> int:
    return len(re.findall(r"待核对|不确定|推断|求婚|禁忌之恋|疑似交往|AU|活动限定", text))


def format_counter(counter: Counter) -> str:
    if not counter:
        return "无"
    return "；".join(f"{key}: {value}" for key, value in counter.most_common())


def copytree_if_exists(src: Path, dst: Path) -> None:
    if src.exists() and not dst.exists():
        shutil.copytree(src, dst)
