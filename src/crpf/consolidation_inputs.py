from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


TOPIC_LABELS = {
    "plot": "剧情",
    "worldbuilding": "世界观",
    "relationships": "人物关系",
    "character_arc": "角色画像",
    "team_story": "团队剧情",
    "dialogue_style": "对话风格",
}

PERSON_ALIASES = {
    "制作人": ("制作人", "プロデューサー"),
    "佑芽": ("佑芽", "妹妹"),
    "琴音": ("琴音", "藤田琴音"),
    "手毬": ("手毬", "月村手毬"),
    "星南": ("星南", "十王星南"),
    "麻央": ("麻央", "有村麻央"),
    "莉莉娅": ("莉莉娅", "葛城莉莉娅"),
    "清夏": ("清夏", "紫云清夏"),
    "广": ("广", "广"),
    "燐羽": ("燐羽", "燐羽"),
    "燕": ("燕", "秦谷美铃", "美铃"),
    "千奈": ("千奈", "仓本千奈"),
    "莉波": ("莉波", "姫崎莉波"),
}

OUTPUT_FILES = {
    "readme": "README.md",
    "character_profile": "character_profile_input.md",
    "relationships": "relationships_input.md",
    "plot_summary": "plot_summary_input.md",
    "worldbuilding": "worldbuilding_input.md",
    "team_story": "team_story_input.md",
    "dialogue_patterns": "dialogue_patterns_input.md",
}


def prepare_consolidation_inputs(summary_output_dir: Path) -> dict[str, Path]:
    summaries_path = summary_output_dir / "scene_summaries.jsonl"
    rows = load_jsonl(summaries_path)
    if not rows:
        raise FileNotFoundError(f"No scene summaries found: {summaries_path}")

    output_dir = summary_output_dir / "consolidation_inputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    written = {
        "readme": write_text(output_dir / OUTPUT_FILES["readme"], render_readme(rows)),
        "character_profile": write_text(
            output_dir / OUTPUT_FILES["character_profile"],
            render_character_profile_input(rows),
        ),
        "relationships": write_text(
            output_dir / OUTPUT_FILES["relationships"],
            render_relationships_input(rows),
        ),
        "plot_summary": write_text(
            output_dir / OUTPUT_FILES["plot_summary"],
            render_plot_summary_input(rows),
        ),
        "worldbuilding": write_text(
            output_dir / OUTPUT_FILES["worldbuilding"],
            render_topic_input(
                rows,
                title="世界观设定整理输入",
                target_doc="data/rag_docs/worldbuilding.md",
                topic="worldbuilding",
                focus_fields=("worldbuilding_facts", "story_facts"),
                prompt=WEB_PROMPTS["worldbuilding"],
            ),
        ),
        "team_story": write_text(
            output_dir / OUTPUT_FILES["team_story"],
            render_topic_input(
                rows,
                title="团队剧情整理输入",
                target_doc="data/rag_docs/team_story.md",
                topic="team_story",
                focus_fields=("story_facts", "relationship_facts"),
                prompt=WEB_PROMPTS["team_story"],
            ),
        ),
        "dialogue_patterns": write_text(
            output_dir / OUTPUT_FILES["dialogue_patterns"],
            render_topic_input(
                rows,
                title="对话风格整理输入",
                target_doc="data/rag_docs/dialogue_patterns.md",
                topic="dialogue_style",
                focus_fields=("dialogue_style", "story_facts", "relationship_facts"),
                prompt=WEB_PROMPTS["dialogue_patterns"],
                direct_only=True,
            ),
        ),
    }
    return written


def render_readme(rows: list[dict[str, object]]) -> str:
    status = Counter(str(row.get("summary_status") or "unknown") for row in rows)
    presence = Counter(str(row.get("saki_presence") or "unknown") for row in rows)
    topics = Counter(topic for row in rows for topic in as_list(row.get("topics")))
    lines = [
        "# RAG 大总结输入包",
        "",
        "这些 Markdown 是给网页版 GPT 做二次总结用的输入材料，不会调用本地 LLM。",
        "",
        "## 使用方式",
        "",
        "1. 按目标文档逐个上传或粘贴对应 `*_input.md`。",
        "2. 使用文件开头的提示词，让网页版 GPT 输出最终 RAG Markdown。",
        "3. 把输出保存/替换到 `data/rag_docs/` 对应文件。",
        "4. 保留 `scene_id` 和 `source_file` 作为证据索引，不要复制长段原始台词。",
        "",
        "## 文件对应关系",
        "",
        "- `character_profile_input.md` -> `data/rag_docs/character_profile.md`",
        "- `relationships_input.md` -> `data/rag_docs/relationships.md`",
        "- `plot_summary_input.md` -> `data/rag_docs/plot_summary.md`",
        "- `worldbuilding_input.md` -> `data/rag_docs/worldbuilding.md`",
        "- `team_story_input.md` -> `data/rag_docs/team_story.md`",
        "- `dialogue_patterns_input.md` -> `data/rag_docs/dialogue_patterns.md`",
        "",
        "## 数据概况",
        "",
        f"- 场景摘要数：{len(rows)}",
        f"- 摘要状态：{format_counter(status)}",
        f"- 咲季出场状态：{format_counter(presence)}",
        f"- 主题分布：{format_counter(topics)}",
        "",
        "## 通用总结要求",
        "",
        "- 用中文总结。",
        "- 输出面向 RAG 检索的稳定事实，不写成剧情赏析文章。",
        "- 区分确定事实、推断、单场景事件。",
        "- 每个重要结论后保留若干 `scene_id` 作为证据。",
        "- 不要长篇复述原台词。",
        "- 遇到冲突信息时写 `待核对`，不要强行合并。",
    ]
    return join_lines(lines)


def render_character_profile_input(rows: list[dict[str, object]]) -> str:
    selected = [
        row
        for row in rows
        if row.get("saki_presence") == "direct"
        or "character_arc" in as_list(row.get("topics"))
        or row.get("saki_actions")
    ]
    lines = doc_header(
        "角色画像整理输入",
        "data/rag_docs/character_profile.md",
        WEB_PROMPTS["character_profile"],
        selected,
    )
    lines.extend(render_fact_frequency(selected, ("story_facts", "relationship_facts"), "候选角色事实"))
    lines.extend(["", "## 按章节/场景整理", ""])
    for chapter, chapter_rows in group_by(selected, "chapter").items():
        lines.extend([f"### {chapter}", ""])
        lines.extend(render_scene_bullets(chapter_rows, fields=("story_facts", "relationship_facts"), limit=None))
        lines.append("")
    return join_lines(lines)


def render_relationships_input(rows: list[dict[str, object]]) -> str:
    selected = [row for row in rows if row.get("relationship_facts") or "relationships" in as_list(row.get("topics"))]
    lines = doc_header(
        "人物关系整理输入",
        "data/rag_docs/relationships.md",
        WEB_PROMPTS["relationships"],
        selected,
    )
    lines.extend(render_fact_frequency(selected, ("relationship_facts",), "高频关系候选事实"))
    lines.extend(["", "## 按人物关系索引", ""])
    for person, aliases in PERSON_ALIASES.items():
        person_rows = [row for row in selected if contains_any(row_text(row), aliases)]
        if not person_rows:
            continue
        lines.extend([f"### 咲季 与 {person}", ""])
        lines.extend(render_scene_bullets(person_rows, fields=("relationship_facts", "story_facts"), limit=None))
        lines.append("")
    other_rows = [
        row
        for row in selected
        if not any(contains_any(row_text(row), aliases) for aliases in PERSON_ALIASES.values())
    ]
    if other_rows:
        lines.extend(["### 其他关系线", ""])
        lines.extend(render_scene_bullets(other_rows, fields=("relationship_facts", "story_facts"), limit=None))
    return join_lines(lines)


def render_plot_summary_input(rows: list[dict[str, object]]) -> str:
    selected = [row for row in rows if row.get("story_facts") or "plot" in as_list(row.get("topics"))]
    lines = doc_header(
        "剧情与章节事件整理输入",
        "data/rag_docs/plot_summary.md",
        WEB_PROMPTS["plot_summary"],
        selected,
    )
    lines.extend(["", "## 按来源类型/章节整理", ""])
    by_kind: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in selected:
        by_kind[str(row.get("source_kind") or "unknown")].append(row)
    for kind, kind_rows in sorted(by_kind.items()):
        lines.extend([f"### {kind}", ""])
        for chapter, chapter_rows in group_by(kind_rows, "chapter").items():
            lines.extend([f"#### {chapter}", ""])
            lines.extend(render_scene_bullets(chapter_rows, fields=("story_facts", "relationship_facts"), limit=None))
            lines.append("")
    return join_lines(lines)


def render_topic_input(
    rows: list[dict[str, object]],
    title: str,
    target_doc: str,
    topic: str,
    focus_fields: tuple[str, ...],
    prompt: str,
    direct_only: bool = False,
) -> str:
    selected = [row for row in rows if topic in as_list(row.get("topics"))]
    if direct_only:
        selected = [row for row in selected if row.get("saki_presence") == "direct"]
    lines = doc_header(title, target_doc, prompt, selected)
    lines.extend(render_fact_frequency(selected, focus_fields, "候选事实"))
    lines.extend(["", "## 场景材料", ""])
    for chapter, chapter_rows in group_by(selected, "chapter").items():
        lines.extend([f"### {chapter}", ""])
        lines.extend(render_scene_bullets(chapter_rows, fields=focus_fields, limit=None))
        lines.append("")
    return join_lines(lines)


def doc_header(title: str, target_doc: str, prompt: str, rows: list[dict[str, object]]) -> list[str]:
    presence = Counter(str(row.get("saki_presence") or "unknown") for row in rows)
    chapters = Counter(str(row.get("chapter") or "unknown") for row in rows)
    return [
        f"# {title}",
        "",
        "## 给网页版 GPT 的提示词",
        "",
        "```text",
        prompt.strip(),
        "```",
        "",
        "## 目标输出",
        "",
        f"- 目标文件：`{target_doc}`",
        "- 输出中文 Markdown。",
        "- 只写稳定设定、剧情事实和可检索条目。",
        "- 重要结论后保留证据 scene_id。",
        "- 不要复制长段原始台词。",
        "",
        "## 输入统计",
        "",
        f"- 输入场景数：{len(rows)}",
        f"- 咲季出场状态：{format_counter(presence)}",
        f"- 高频章节：{format_counter(chapters, limit=12)}",
    ]


def render_fact_frequency(
    rows: list[dict[str, object]],
    fields: tuple[str, ...],
    title: str,
    limit: int = 60,
) -> list[str]:
    counter: Counter[str] = Counter()
    for row in rows:
        for field in fields:
            counter.update(str(value) for value in as_list(row.get(field)) if str(value).strip())
    lines = ["", f"## {title}", ""]
    if not counter:
        return lines + ["- 无"]
    for fact, count in counter.most_common(limit):
        suffix = f"（{count} 次）" if count > 1 else ""
        lines.append(f"- {fact}{suffix}")
    return lines


def render_scene_bullets(
    rows: list[dict[str, object]],
    fields: tuple[str, ...],
    limit: int | None,
) -> list[str]:
    lines: list[str] = []
    use_rows = rows[:limit] if limit is not None else rows
    for row in use_rows:
        scene_id = row.get("scene_id", "")
        presence = row.get("saki_presence", "")
        title = row.get("title", "")
        source = first(row.get("source_refs")) or row.get("source_file", "")
        topics = "、".join(TOPIC_LABELS.get(str(topic), str(topic)) for topic in as_list(row.get("topics")))
        lines.extend(
            [
                f"- `{scene_id}` [{presence}] {title}",
                f"  - 来源：`{source}`",
                f"  - 主题：{topics}",
                f"  - 摘要：{row.get('summary', '')}",
            ]
        )
        facts = collect_fields(row, fields)
        if facts:
            lines.append(f"  - 结构化事实：{'；'.join(facts[:8])}")
        evidence = format_evidence(row, limit=3)
        if evidence:
            lines.append(f"  - 证据：{evidence}")
    if limit is not None and len(rows) > limit:
        lines.append(f"- 另有 {len(rows) - limit} 条场景未在本节展开。")
    return lines


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


def write_text(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def group_by(rows: Iterable[dict[str, object]], key: str) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key) or "unknown")].append(row)
    return dict(sorted(grouped.items(), key=lambda item: item[0]))


def collect_fields(row: dict[str, object], fields: tuple[str, ...]) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for field in fields:
        for value in as_list(row.get(field)):
            text = str(value).strip()
            if text and text not in seen:
                seen.add(text)
                values.append(text)
    return values


def row_text(row: dict[str, object]) -> str:
    pieces = [
        str(row.get("title") or ""),
        str(row.get("summary") or ""),
        " ".join(str(value) for value in as_list(row.get("speakers"))),
        " ".join(str(value) for value in as_list(row.get("story_facts"))),
        " ".join(str(value) for value in as_list(row.get("relationship_facts"))),
        " ".join(str(value) for value in as_list(row.get("worldbuilding_facts"))),
    ]
    return "\n".join(pieces)


def contains_any(text: str, aliases: tuple[str, ...]) -> bool:
    return any(alias in text for alias in aliases)


def format_evidence(row: dict[str, object], limit: int = 3) -> str:
    source_file = str(row.get("source_file") or "")
    snippets: list[str] = []
    for evidence in as_list(row.get("key_evidence"))[:limit]:
        if not isinstance(evidence, dict):
            continue
        snippets.append(
            f"`{source_file}:{evidence.get('source_row', '')}` {evidence.get('speaker', '')}: {evidence.get('text_zh', '')}"
        )
    return "；".join(snippets)


def format_counter(counter: Counter, limit: int | None = None) -> str:
    if not counter:
        return "无"
    return "；".join(f"{key}: {value}" for key, value in counter.most_common(limit))


def as_list(value: object) -> list[object]:
    return value if isinstance(value, list) else []


def first(value: object) -> object:
    values = as_list(value)
    return values[0] if values else ""


def join_lines(lines: list[str]) -> str:
    return "\n".join(str(line).rstrip() for line in lines).rstrip() + "\n"


WEB_PROMPTS = {
    "character_profile": """
你是角色 RAG 知识库编辑。请基于输入的场景摘要，为《学园偶像大师》花海咲季生成 `character_profile.md`。
要求：总结稳定角色画像、核心动机、性格特征、成长弧线、弱点与矛盾、与偶像活动相关的行为模式；按条目组织；每个重要结论附 2-5 个 scene_id 作为证据；不要复述长台词；不确定处标注“待核对”。
""",
    "relationships": """
你是角色关系知识库编辑。请基于输入材料生成 `relationships.md`。
要求：按人物分别总结咲季与制作人、佑芽、琴音、手毬、星南等人的关系；区分亲情、竞争、搭档、团队互动和外部评价；保留 scene_id 证据；不要把单场景玩笑误写成长期关系设定。
""",
    "plot_summary": """
你是剧情知识库编辑。请基于输入材料生成 `plot_summary.md`。
要求：按章节/来源类型整理咲季相关事件线；突出会影响角色理解的关键事件、选拔/训练/演出/团队节点；保留 scene_id 证据；不要逐条流水账，合并重复事件。
""",
    "worldbuilding": """
你是世界观知识库编辑。请基于输入材料生成 `worldbuilding.md`。
要求：总结初星学园、宿舍、训练、演出、选拔、课程、偶像活动、H.I.F/庆典等设定中与咲季相关的部分；区分通用设定和咲季个人经历；保留 scene_id 证据。
""",
    "team_story": """
你是团队剧情知识库编辑。请基于输入材料生成 `team_story.md`。
要求：总结咲季在 Re;IRIS/宿舍/同学互动/团队活动中的位置、团队冲突与协作、共同演出和日常事件；保留 scene_id 证据；避免重复列举相似日常。
""",
    "dialogue_patterns": """
你是角色口吻知识库编辑。请基于输入材料生成 `dialogue_patterns.md`。
要求：总结咲季的中文口吻、称呼、语气节奏、情绪反应、被调侃时的反弹、胜负欲表达、对制作人/佑芽/队友的互动方式；只总结风格规则，不复刻长台词；保留 scene_id 证据。
""",
}
