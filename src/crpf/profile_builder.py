from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable

from .cleaning import normalize_text


RAG_DOC_NAMES = [
    "character_profile.md",
    "plot_summary.md",
    "relationships.md",
    "worldbuilding.md",
    "dialogue_patterns.md",
]


def build_rag_docs(samples: Iterable[dict[str, str]], output_dir: Path, character_name: str = "花海咲季") -> list[Path]:
    rows = [row for row in samples if normalize_text(row.get("response_translation", ""))]
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = build_stats(rows)
    example_sets = select_example_sets(rows)

    docs = {
        "character_profile.md": render_character_profile(character_name, stats, example_sets),
        "plot_summary.md": render_plot_summary(character_name, stats, example_sets),
        "relationships.md": render_relationships(character_name, stats, example_sets),
        "worldbuilding.md": render_worldbuilding(character_name, stats, example_sets),
        "dialogue_patterns.md": render_dialogue_patterns(character_name, stats, example_sets),
    }

    written: list[Path] = []
    for name, content in docs.items():
        path = output_dir / name
        path.write_text(content, encoding="utf-8")
        written.append(path)
    return written


def build_stats(rows: list[dict[str, str]]) -> dict[str, object]:
    source_counts = Counter(row.get("source_kind", "") or "unknown" for row in rows)
    chapter_counts = Counter(row.get("chapter", "") or "unknown" for row in rows)
    response_lens = [len(normalize_text(row.get("response_translation", ""))) for row in rows]
    context_counts = Counter(row.get("context_line_count", "0") for row in rows)
    return {
        "total_samples": len(rows),
        "source_counts": source_counts,
        "top_chapters": chapter_counts.most_common(8),
        "avg_response_len": round(sum(response_lens) / len(response_lens), 1) if response_lens else 0,
        "max_response_len": max(response_lens) if response_lens else 0,
        "context_counts": context_counts,
    }


def select_example_sets(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    return {
        "general": pick_examples(rows, limit=8),
        "story": pick_examples(rows, kinds=("pstory", "dear", "cidol"), limit=8),
        "relationships": pick_examples(rows, keywords=("制作人", "佑芽", "琴音", "手毬", "朋友", "对手", "妹妹", "姐姐"), limit=10),
        "world": pick_examples(rows, keywords=("偶像", "学园", "训练", "试镜", "演唱会", "制作人", "课程", "工作"), limit=8),
        "dialogue": pick_examples(rows, keywords=("当然", "才不是", "真是的", "顶级偶像", "制作人", "哼", "呵呵", "嘿嘿"), limit=10),
    }


def pick_examples(
    rows: list[dict[str, str]],
    kinds: tuple[str, ...] | None = None,
    keywords: tuple[str, ...] | None = None,
    limit: int = 8,
) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    seen_sources: set[str] = set()
    for row in rows:
        if kinds and row.get("source_kind") not in kinds:
            continue
        combined = "\n".join([row.get("context_translation", ""), row.get("response_translation", "")])
        if keywords and not any(keyword in combined for keyword in keywords):
            continue
        source_key = f"{row.get('source_file')}:{row.get('source_row')}"
        if source_key in seen_sources:
            continue
        selected.append(row)
        seen_sources.add(source_key)
        if len(selected) >= limit:
            break

    if len(selected) < limit and not keywords:
        return selected
    if len(selected) < limit:
        for row in rows:
            source_key = f"{row.get('source_file')}:{row.get('source_row')}"
            if source_key in seen_sources:
                continue
            if kinds and row.get("source_kind") not in kinds:
                continue
            selected.append(row)
            seen_sources.add(source_key)
            if len(selected) >= limit:
                break
    return selected


def render_character_profile(character_name: str, stats: dict[str, object], examples: dict[str, list[dict[str, str]]]) -> str:
    return f"""# {character_name} 角色风格说明书

## 说明

本文档用于 RAG 检索时提供角色人格、口吻、行为边界和互动原则。它不是训练集，也不应粘贴完整原始剧情；请只保留人工总结后的稳定设定和少量可审阅例句。

## TODO

- [ ] 补充官方基础资料：年级、生日、身高、兴趣、特技等。
- [ ] 明确中文 RP 时的称呼规则：对用户、同学、妹妹、老师分别怎么称呼。
- [ ] 标记禁止事项：不复读大段原作台词，不编造未确认剧情事实。
- [ ] 人工总结咲季的核心矛盾、成长弧线和情绪触发点。

## 基础统计

{format_stats(stats)}

## 人工总结区

### 核心定位

- TODO: 用 3-5 条总结咲季作为偶像、学生、姐姐、搭档的身份定位。

### 性格关键词

- TODO: 例如自信、好胜、率直、反应大、容易害羞但会反击。

### 与制作人的互动原则

- TODO: 说明信赖、竞争目标、调侃尺度、感谢表达和边界。

## 候选台词样例

{format_examples(examples["general"])}
"""


def render_plot_summary(character_name: str, stats: dict[str, object], examples: dict[str, list[dict[str, str]]]) -> str:
    return f"""# {character_name} 剧情摘要

## 说明

本文档用于沉淀剧情事实和成长线，供 RAG 回答“发生过什么”和“为什么会这样”时检索。请人工提炼剧情，不要把 CSV 全量复制进来。

## TODO

- [ ] 按主线、亲爱度、卡牌剧情、活动剧情分段总结。
- [ ] 标注哪些是可靠事实，哪些只是推测或语气观察。
- [ ] 记录关键转折：试镜、演出、失败、胜利、与制作人的承诺。

## 基础统计

{format_stats(stats)}

## 人工总结区

### 主线成长

- TODO: 总结咲季从起点到目标推进的主要变化。

### 关键事件

- TODO: 用短条目记录事件名称、出处、影响。

### 待核对剧情点

- TODO: 列出需要回看 CSV 或官方资料确认的点。

## 候选台词样例

{format_examples(examples["story"])}
"""


def render_relationships(character_name: str, stats: dict[str, object], examples: dict[str, list[dict[str, str]]]) -> str:
    return f"""# {character_name} 人物关系

## 说明

本文档用于 RAG 检索人物关系、称呼、亲疏和互动模式。请把关系写成稳定事实和可执行对话规则，而不是堆台词。

## TODO

- [ ] 制作人：信赖、搭档、调侃、训练目标。
- [ ] 花海佑芽：妹妹、竞争对手、骄傲与关心并存。
- [ ] 其他偶像：朋友、对手、同学、队友关系。
- [ ] 教师/学园相关人物：称呼和敬语规则。

## 基础统计

{format_stats(stats)}

## 人工总结区

### 制作人

- TODO: 总结咲季对制作人的期待、依赖、反击和感谢方式。

### 佑芽

- TODO: 总结姐姐身份、竞争意识、保护欲和自豪感。

### 同学与对手

- TODO: 按角色列出关系标签和互动注意事项。

## 候选台词样例

{format_examples(examples["relationships"])}
"""


def render_worldbuilding(character_name: str, stats: dict[str, object], examples: dict[str, list[dict[str, str]]]) -> str:
    return f"""# {character_name} 世界观与设定

## 说明

本文档用于保存初星学园、偶像训练、制作人科、试镜、演出、课程、工作等世界观事实。它应该像可检索设定卡，而不是剧情原文仓库。

## TODO

- [ ] 解释初星学园、偶像科、制作人科的基本关系。
- [ ] 整理训练、试镜、演唱会、课程、营业/工作等常见场景。
- [ ] 标注哪些设定来自 CSV，哪些来自人工补充资料。

## 基础统计

{format_stats(stats)}

## 人工总结区

### 学园与训练系统

- TODO: 总结角色日常所在环境和训练流程。

### 偶像活动

- TODO: 记录演唱会、持歌、试镜、活动工作的常见设定。

### RAG 使用边界

- TODO: 世界观回答可以补事实，但不要替用户编造未确认官方设定。

## 候选台词样例

{format_examples(examples["world"])}
"""


def render_dialogue_patterns(character_name: str, stats: dict[str, object], examples: dict[str, list[dict[str, str]]]) -> str:
    return f"""# {character_name} 对话模式

## 说明

本文档用于约束中文 RP 输出风格，包括口头禅、句式、情绪节奏和拒绝/害羞/感谢/鼓劲时的表达。请人工归纳模式，样例只作为证据。

## TODO

- [ ] 总结第一人称、第二人称、常见语尾和高频感叹。
- [ ] 总结被夸、被调侃、失败、胜利、关心他人时的反应模板。
- [ ] 写出中文化原则：保留咲季能量感，但避免翻译腔过重。
- [ ] 标记训练/聊天时不应输出的内容：长篇原台词、露骨越界、未确认事实。

## 基础统计

{format_stats(stats)}

## 人工总结区

### 高频语气

- TODO: 例如“当然”“才不是”“真是的”“让你见识一下”等。

### 情绪节奏

- TODO: 咲季常从自信推进到被调侃后的慌张/害羞，再迅速反击或重整气势。

### 中文输出规则

- TODO: 说明用中文 RP 时哪些表达自然，哪些日文语气要转写。

## 候选台词样例

{format_examples(examples["dialogue"])}
"""


def format_stats(stats: dict[str, object]) -> str:
    source_counts: Counter[str] = stats["source_counts"]  # type: ignore[assignment]
    top_chapters: list[tuple[str, int]] = stats["top_chapters"]  # type: ignore[assignment]
    context_counts: Counter[str] = stats["context_counts"]  # type: ignore[assignment]
    source_lines = "\n".join(f"- {kind}: {count}" for kind, count in source_counts.most_common())
    chapter_lines = "\n".join(f"- {chapter}: {count}" for chapter, count in top_chapters)
    context_lines = "\n".join(f"- {count} 上下文行: {total}" for count, total in sorted(context_counts.items()))
    return f"""- 样本数：{stats["total_samples"]}
- 平均中文回复长度：{stats["avg_response_len"]} 字
- 最长中文回复长度：{stats["max_response_len"]} 字

### 来源分布

{source_lines}

### 主要章节

{chapter_lines}

### 上下文窗口分布

{context_lines}
"""


def format_examples(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "- TODO: 暂无候选样例。"
    blocks: list[str] = []
    for row in rows:
        context = trim_block(row.get("context_translation", ""), max_chars=220)
        response = trim_block(row.get("response_translation", ""), max_chars=160)
        original = trim_block(row.get("response", ""), max_chars=120)
        blocks.append(
            "\n".join(
                [
                    f"### {row.get('sample_id', '')} | {row.get('source_kind', '')} | {row.get('scene', '')}",
                    "",
                    f"- 来源：`{row.get('source_file', '')}:{row.get('source_row', '')}`",
                    f"- 质量分：{row.get('quality_score', '')}",
                    "- 中文上下文：",
                    "",
                    quote_block(context or "（无）"),
                    "",
                    "- 中文回复候选：",
                    "",
                    quote_block(response),
                    "",
                    "- 日文原句参考：",
                    "",
                    quote_block(original),
                ]
            )
        )
    return "\n\n".join(blocks)


def trim_block(value: object, max_chars: int) -> str:
    text = normalize_text(value)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def quote_block(value: str) -> str:
    return "\n".join(f"> {line}" if line else ">" for line in value.splitlines())
