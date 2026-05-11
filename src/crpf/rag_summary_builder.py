from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from .cleaning import normalize_text
from .context_builder import SPEAKER_TRANSLATIONS
from .exporters import sanitize_chinese_text
from .io_csv import find_csv_files, read_csv_any_encoding
from .providers import ModelProviderConfig, complete_chat


TARGET_NAMES = ("咲季", "花海咲季", "hski", "花海咲季", "小咲季")
TARGET_MENTION_WORDS = ("咲季", "花海咲季", "小咲季", "hski")
MAX_LINES_PER_BLOCK = 80

TOPIC_DEFS = {
    "plot": ("剧情推进", ("试镜", "演唱会", "合格", "失败", "胜利", "目标", "顶级偶像", "约定", "合同", "成长")),
    "worldbuilding": ("世界观设定", ("学园", "偶像", "制作人科", "课程", "训练", "试镜", "演唱会", "工作", "营业", "持歌")),
    "relationships": ("人物关系", ("制作人", "佑芽", "妹妹", "姐姐", "琴音", "手毬", "朋友", "对手", "伙伴", "家人", "父母")),
    "character_arc": ("角色成长", ("努力", "不甘心", "胜利", "失败", "顶级偶像", "训练", "成长", "目标", "自信", "悔")),
    "team_story": ("团队剧情", ("大家", "我们", "队伍", "伙伴", "同伴", "朋友", "琴音", "手毬", "清夏", "麻央", "广", "莉波")),
    "dialogue_style": ("对话风格", ("当然", "才不是", "真是的", "哼", "呵呵", "嘿嘿", "制作人", "让你见识", "不是吗", "啦")),
}

OUTPUT_TOPIC_FILES = {
    "plot": "plot_summary.md",
    "worldbuilding": "worldbuilding_summary.md",
    "relationships": "relationships_summary.md",
    "character_arc": "character_arc_summary.md",
    "team_story": "team_story_summary.md",
    "dialogue_style": "dialogue_style_summary.md",
}

RAG_DOC_FILES = {
    "plot": "plot_summary.md",
    "worldbuilding": "worldbuilding.md",
    "relationships": "relationships.md",
    "character_arc": "character_profile.md",
    "team_story": "team_story.md",
    "dialogue_style": "dialogue_patterns.md",
}


def build_raw_rag_summaries(raw_csv_dir: Path, output_dir: Path, rag_docs_dir: Path) -> dict[str, Path]:
    scenes = build_scene_chunks(raw_csv_dir)
    summaries = [summarize_scene(scene) for scene in scenes]
    return write_rag_summary_outputs(scenes, summaries, output_dir, rag_docs_dir)


def build_llm_rag_summaries(
    raw_csv_dir: Path,
    output_dir: Path,
    rag_docs_dir: Path,
    model: str,
    ollama_base_url: str,
    provider: str = "ollama",
    base_url: str | None = None,
    api_key_env: str = "",
    limit: int | None = None,
    resume: bool = True,
    summary_mode: str = "fast",
) -> dict[str, Path]:
    scenes = build_scene_chunks(raw_csv_dir)
    if limit is not None:
        scenes = scenes[:limit]
    existing: dict[str, dict[str, object]] = {}
    if resume:
        existing.update(load_existing_summaries(output_dir / "scene_summaries.jsonl"))
        existing.update(load_existing_summaries(output_dir / "scene_summaries.jsonl.partial"))

    summaries: list[dict[str, object]] = []
    for index, scene in enumerate(scenes, start=1):
        scene_id = str(scene["scene_id"])
        if scene_id in existing and can_reuse_summary(existing[scene_id], summary_mode):
            summaries.append(existing[scene_id])
            continue
        summary = summarize_scene_with_llm(
            scene,
            model=model,
            ollama_base_url=ollama_base_url,
            provider=provider,
            base_url=base_url,
            api_key_env=api_key_env,
            summary_mode=summary_mode,
        )
        summaries.append(summary)
        write_jsonl(output_dir / "scene_summaries.jsonl.partial", summaries)
        print(f"[{index}/{len(scenes)}] summarized {scene_id} {scene['source_file']}", flush=True)

    return write_rag_summary_outputs(scenes, summaries, output_dir, rag_docs_dir)


def can_reuse_summary(summary: dict[str, object], summary_mode: str) -> bool:
    status = str(summary.get("summary_status", ""))
    if summary_mode == "fast":
        return status in {"llm_fast_scene_card", "llm_scene_card"}
    return status == "llm_scene_card"


def write_rag_summary_outputs(
    scenes: list[dict[str, object]],
    summaries: list[dict[str, object]],
    output_dir: Path,
    rag_docs_dir: Path,
) -> dict[str, Path]:
    topic_docs = render_topic_documents(summaries)
    output_dir.mkdir(parents=True, exist_ok=True)
    topic_dir = output_dir / "topic_summaries"
    topic_dir.mkdir(parents=True, exist_ok=True)
    rag_docs_dir.mkdir(parents=True, exist_ok=True)
    scene_docs_dir = rag_docs_dir / "scenes"
    scene_docs_dir.mkdir(parents=True, exist_ok=True)

    scene_chunks_path = output_dir / "scene_chunks.jsonl"
    scene_summaries_path = output_dir / "scene_summaries.jsonl"
    write_jsonl(scene_chunks_path, scenes)
    write_jsonl(scene_summaries_path, summaries)

    written = {
        "scene_chunks": scene_chunks_path,
        "scene_summaries": scene_summaries_path,
    }
    for summary in summaries:
        scene_doc_path = scene_docs_dir / f"{summary['scene_id']}_{safe_filename(str(summary['scene']))}.md"
        scene_doc_path.write_text(render_scene_card(summary), encoding="utf-8")
    written["rag_scene_cards_dir"] = scene_docs_dir

    for topic, content in topic_docs.items():
        topic_path = topic_dir / OUTPUT_TOPIC_FILES[topic]
        topic_path.write_text(content, encoding="utf-8")
        written[f"topic_{topic}"] = topic_path

        rag_path = rag_docs_dir / RAG_DOC_FILES[topic]
        rag_path.write_text(content, encoding="utf-8")
        written[f"rag_{topic}"] = rag_path
    return written


def load_existing_summaries(path: Path) -> dict[str, dict[str, object]]:
    if not path.exists():
        return {}
    summaries: dict[str, dict[str, object]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            summaries[str(row.get("scene_id") or row.get("summary_id"))] = row
    return summaries


def build_scene_chunks(raw_csv_dir: Path) -> list[dict[str, object]]:
    scenes: list[dict[str, object]] = []
    for path in find_csv_files(raw_csv_dir):
        raw_rows = read_csv_any_encoding(path)
        lines = light_clean_rows(raw_rows)
        if not lines:
            continue
        source_file = path.as_posix()
        relevant_indexes = find_relevant_line_indexes(lines, source_file)
        if not relevant_indexes:
            continue
        scene_id = f"scene-{len(scenes) + 1:05d}"
        speakers = sorted({line["speaker"] for line in lines if line["speaker"]})
        start_row = lines[0]["source_row"]
        end_row = lines[-1]["source_row"]
        scenes.append(
            {
                "scene_id": scene_id,
                "chunk_id": scene_id,
                "source_file": source_file,
                "source_kind": source_kind(raw_csv_dir, path),
                "chapter": infer_chapter(raw_csv_dir, path),
                "scene": path.stem,
                "start_row": start_row,
                "end_row": end_row,
                "line_count": len(lines),
                "speakers": speakers,
                "related_reason": related_reason(source_file, lines),
                "dialogue_zh": lines,
                "dialogue_blocks": split_dialogue_blocks(lines),
                "source_refs": [f"{source_file}:{start_row}-{end_row}"],
                "summary_status": "heuristic_scene_card",
            }
        )
    return scenes


def light_clean_rows(rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    cleaned: list[dict[str, str]] = []
    for index, row in enumerate(rows, start=1):
        line_id = normalize_text(row.get("id", ""))
        raw_speaker = normalize_text(row.get("name", ""))
        raw_text = normalize_text(row.get("text", ""))
        raw_translation = normalize_text(row.get("trans", ""))
        if raw_speaker in {"info", "译者"}:
            continue
        if not raw_text and not raw_translation:
            continue
        speaker = display_speaker(raw_speaker, line_id)
        text_zh = sanitize_chinese_text(raw_translation)
        text_ja = normalize_text(raw_text)
        if not text_zh:
            text_zh = text_ja
        cleaned.append(
            {
                "source_row": str(index),
                "line_id": line_id,
                "speaker": speaker,
                "text_zh": text_zh,
                "text_ja": text_ja,
            }
        )
    return cleaned


def display_speaker(raw_speaker: str, line_id: str) -> str:
    if raw_speaker:
        return SPEAKER_TRANSLATIONS.get(raw_speaker, raw_speaker)
    if line_id == "select":
        return "选项"
    return "旁白"


def find_relevant_line_indexes(lines: list[dict[str, str]], source_file: str) -> set[int]:
    indexes: set[int] = set()
    path_lower = source_file.lower()
    source_is_target = "/hski/" in path_lower or "_hski" in path_lower or "-hski" in path_lower
    for index, line in enumerate(lines):
        combined = "\n".join([line["speaker"], line["text_zh"], line["text_ja"]])
        if source_is_target or any(word in combined for word in TARGET_MENTION_WORDS):
            indexes.add(index)
    return indexes


def split_dialogue_blocks(lines: list[dict[str, str]]) -> list[dict[str, object]]:
    blocks: list[dict[str, object]] = []
    for block_index, start in enumerate(range(0, len(lines), MAX_LINES_PER_BLOCK), start=1):
        block_lines = lines[start : start + MAX_LINES_PER_BLOCK]
        blocks.append(
            {
                "block_index": block_index,
                "start_row": block_lines[0]["source_row"],
                "end_row": block_lines[-1]["source_row"],
                "line_count": len(block_lines),
                "lines": block_lines,
            }
        )
    return blocks


def summarize_scene(scene: dict[str, object]) -> dict[str, object]:
    lines: list[dict[str, str]] = scene["dialogue_zh"]  # type: ignore[assignment]
    text = "\n".join(f"{line['speaker']}: {line['text_zh']}" for line in lines)
    topics = classify_topics(text)
    saki_lines = [line for line in lines if "咲季" in line["speaker"] or "咲季" in line["text_zh"]]
    evidence = pick_evidence_lines(lines)
    summary = {
        "summary_id": scene["scene_id"],
        "scene_id": scene["scene_id"],
        "source_file": scene["source_file"],
        "source_kind": scene["source_kind"],
        "chapter": scene["chapter"],
        "scene": scene["scene"],
        "dialogue_blocks": [
            {
                "block_index": block["block_index"],
                "start_row": block["start_row"],
                "end_row": block["end_row"],
                "line_count": block["line_count"],
            }
            for block in scene["dialogue_blocks"]  # type: ignore[index]
        ],
        "source_refs": scene["source_refs"],
        "line_count": scene["line_count"],
        "speakers": scene["speakers"],
        "topics": topics,
        "title": make_title(scene, topics, lines),
        "summary": make_summary_text(lines, topics, saki_lines),
        "saki_role": infer_saki_role(lines),
        "relationship_signals": extract_relationship_signals(text),
        "worldbuilding_signals": extract_keyword_hits(text, TOPIC_DEFS["worldbuilding"][1], limit=8),
        "dialogue_style_signals": extract_keyword_hits(text, TOPIC_DEFS["dialogue_style"][1], limit=8),
        "key_evidence": evidence,
        "summary_status": scene.get("summary_status", "heuristic_scene_card"),
    }
    return summary


def summarize_scene_with_llm(
    scene: dict[str, object],
    model: str,
    ollama_base_url: str,
    provider: str = "ollama",
    base_url: str | None = None,
    api_key_env: str = "",
    summary_mode: str = "fast",
) -> dict[str, object]:
    heuristic = summarize_scene(scene)
    dialogue = format_scene_dialogue_for_prompt(scene, max_line_chars=120 if summary_mode == "fast" else 180)
    prompt = build_scene_summary_prompt(scene, dialogue, summary_mode=summary_mode)
    try:
        data = call_summary_json(
            model=model,
            base_url=base_url or ollama_base_url,
            provider=provider,
            api_key_env=api_key_env,
            prompt=prompt,
            summary_mode=summary_mode,
        )
        return merge_llm_summary(scene, heuristic, data, summary_mode=summary_mode)
    except Exception as exc:
        if summary_mode == "fast":
            try:
                strict_prompt = prompt + "\n\n重要：必须输出合法 JSON；数组元素之间必须用英文逗号分隔；不要输出 Markdown。"
                data = call_summary_json(
                    model=model,
                    base_url=base_url or ollama_base_url,
                    provider=provider,
                    api_key_env=api_key_env,
                    prompt=strict_prompt,
                    summary_mode="fast_strict",
                )
                return merge_llm_summary(scene, heuristic, data, summary_mode="fast")
            except Exception as retry_exc:
                fallback = dict(heuristic)
                fallback["summary_status"] = "llm_failed_fallback_heuristic"
                fallback["llm_error"] = f"{exc}; strict retry failed: {retry_exc}"
                return fallback
        fallback = dict(heuristic)
        fallback["summary_status"] = "llm_failed_fallback_heuristic"
        fallback["llm_error"] = str(exc)
        return fallback


def build_scene_summary_prompt(scene: dict[str, object], dialogue: str, summary_mode: str = "fast") -> str:
    if summary_mode == "fast":
        return build_fast_scene_summary_prompt(scene, dialogue)
    return f"""/no_think
请阅读下面一整个 CSV 场景，生成用于角色 RAG 数据库的中文结构化摘要。

目标角色：花海咲季 / 咲季。

要求：
- 必须理解整段场景，不要只抓关键词。
- 区分“咲季直接出场发言”和“别人提到/评价咲季”。
- 只总结剧情事实、人物关系、世界观信息、角色弧线和对话风格。
- 不要复述长段原台词，不要编造场景外信息。
- 保留可追溯证据行号。
- 只输出 JSON，不要 Markdown，不要思考过程。

来源：{scene['source_file']}:{scene['start_row']}-{scene['end_row']}
章节：{scene['chapter']}
场景：{scene['scene']}
参与者：{'、'.join(scene['speakers'])}

对话：
{dialogue}

JSON 字段：
{{
  "title": "一句话场景标题",
  "summary": "150-300字中文剧情摘要",
  "saki_presence": "direct|mentioned|background",
  "saki_role": "咲季在本场景中的作用",
  "saki_actions": ["咲季做了什么或表达了什么"],
  "story_facts": ["可进入RAG的剧情事实"],
  "relationship_facts": ["人物关系事实"],
  "worldbuilding_facts": ["世界观/学园/偶像活动设定"],
  "character_arc": ["咲季性格、成长、动机、弱点、目标相关观察"],
  "dialogue_style": ["咲季口吻或互动模式观察"],
  "topics": ["plot|worldbuilding|relationships|character_arc|team_story|dialogue_style"],
  "key_evidence": [
    {{"source_row": "行号", "speaker": "说话人", "text_zh": "短证据摘录"}}
  ]
}}"""


def build_fast_scene_summary_prompt(scene: dict[str, object], dialogue: str) -> str:
    return f"""/no_think
你是剧情 RAG 资料员。阅读完整 CSV 场景，输出紧凑中文 JSON。

目标角色：花海咲季 / 咲季。
规则：理解整段场景；区分咲季 direct/mentioned/background；不要编造；证据只给短摘录。

来源：{scene['source_file']}:{scene['start_row']}-{scene['end_row']}
场景：{scene['scene']}
参与者：{'、'.join(scene['speakers'])}

对话：
{dialogue}

只输出这个 JSON 对象：
{{
  "title": "12字内标题",
  "summary": "80-140字中文摘要",
  "saki_presence": "direct|mentioned|background",
  "facts": ["剧情事实，最多4条"],
  "relationships": ["人物关系事实，最多3条"],
  "worldbuilding": ["设定事实，最多3条"],
  "evidence": [
    {{"source_row": "行号", "speaker": "说话人", "text_zh": "20字内摘录"}}
  ]
}}"""


def format_scene_dialogue_for_prompt(scene: dict[str, object], max_line_chars: int = 180) -> str:
    lines: list[dict[str, str]] = scene["dialogue_zh"]  # type: ignore[assignment]
    rendered = []
    for line in lines:
        text = trim_text(line["text_zh"], max_line_chars)
        rendered.append(f"[{line['source_row']}] {line['speaker']}: {text}")
    return "\n".join(rendered)


def call_summary_json(
    model: str,
    base_url: str,
    provider: str,
    api_key_env: str,
    prompt: str,
    summary_mode: str = "fast",
) -> dict[str, object]:
    content = complete_chat(
        ModelProviderConfig(provider=provider, model=model, base_url=base_url, api_key_env=api_key_env),
        prompt=prompt,
        system="你是严谨的剧情资料整理员，只输出合法 JSON，不输出思考过程。",
        temperature=0.1,
        num_ctx=4096 if summary_mode in {"fast", "fast_strict"} else 8192,
        num_predict=768 if summary_mode == "fast_strict" else (512 if summary_mode == "fast" else 1024),
        timeout=240,
        json_mode=True,
    )
    return parse_llm_json(content)


def call_ollama_json(model: str, ollama_base_url: str, prompt: str, summary_mode: str = "fast") -> dict[str, object]:
    return call_summary_json(
        model=model,
        base_url=ollama_base_url,
        provider="ollama",
        api_key_env="",
        prompt=prompt,
        summary_mode=summary_mode,
    )


def parse_llm_json(content: str) -> dict[str, object]:
    cleaned = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise
        data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("LLM output is not a JSON object")
    return data


def merge_llm_summary(
    scene: dict[str, object],
    heuristic: dict[str, object],
    data: dict[str, object],
    summary_mode: str = "fast",
) -> dict[str, object]:
    topics = normalize_topics(data.get("topics")) or heuristic["topics"]
    evidence = (
        normalize_evidence(data.get("key_evidence"), scene)
        or normalize_evidence(data.get("evidence"), scene)
        or heuristic["key_evidence"]
    )
    story_facts = normalize_list(data.get("story_facts")) or normalize_list(data.get("facts"))
    relationship_facts = normalize_list(data.get("relationship_facts")) or normalize_list(data.get("relationships"))
    worldbuilding_facts = normalize_list(data.get("worldbuilding_facts")) or normalize_list(data.get("worldbuilding"))
    summary = dict(heuristic)
    summary.update(
        {
            "topics": topics,
            "title": normalize_text(data.get("title", "")) or heuristic["title"],
            "summary": normalize_text(data.get("summary", "")) or heuristic["summary"],
            "saki_presence": normalize_text(data.get("saki_presence", "")) or infer_presence(scene),
            "saki_role": normalize_text(data.get("saki_role", "")) or heuristic["saki_role"],
            "saki_actions": normalize_list(data.get("saki_actions")),
            "story_facts": story_facts,
            "relationship_facts": relationship_facts,
            "worldbuilding_facts": worldbuilding_facts,
            "character_arc": normalize_list(data.get("character_arc")),
            "dialogue_style": normalize_list(data.get("dialogue_style")),
            "key_evidence": evidence,
            "summary_status": "llm_fast_scene_card" if summary_mode == "fast" else "llm_scene_card",
        }
    )
    return summary


def normalize_topics(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    allowed = set(TOPIC_DEFS)
    topics = [str(item) for item in value if str(item) in allowed]
    return topics[:6]


def normalize_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [normalize_text(item) for item in value if normalize_text(item)][:12]


def normalize_evidence(value: object, scene: dict[str, object]) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    lines_by_row = {line["source_row"]: line for line in scene["dialogue_zh"]}  # type: ignore[index]
    evidence: list[dict[str, str]] = []
    for item in value[:8]:
        if not isinstance(item, dict):
            continue
        row = normalize_text(item.get("source_row", ""))
        source_line = lines_by_row.get(row, {})
        evidence.append(
            {
                "source_row": row,
                "speaker": normalize_text(item.get("speaker", "")) or source_line.get("speaker", ""),
                "text_zh": trim_text(normalize_text(item.get("text_zh", "")) or source_line.get("text_zh", ""), 140),
            }
        )
    return [item for item in evidence if item["source_row"] and item["text_zh"]]


def infer_presence(scene: dict[str, object]) -> str:
    lines: list[dict[str, str]] = scene["dialogue_zh"]  # type: ignore[assignment]
    if any("咲季" in line["speaker"] for line in lines):
        return "direct"
    if any("咲季" in line["text_zh"] or "咲季" in line["text_ja"] for line in lines):
        return "mentioned"
    return "background"


def classify_topics(text: str) -> list[str]:
    scores: list[tuple[str, int]] = []
    for topic, (_, keywords) in TOPIC_DEFS.items():
        score = sum(text.count(keyword) for keyword in keywords)
        if score:
            scores.append((topic, score))
    if not scores:
        return ["plot"]
    return [topic for topic, _ in sorted(scores, key=lambda item: (-item[1], item[0]))[:4]]


def make_title(chunk: dict[str, object], topics: list[str], lines: list[dict[str, str]]) -> str:
    topic_name = TOPIC_DEFS[topics[0]][0] if topics else "剧情"
    scene = chunk["scene"]
    first_saki = next((line["text_zh"] for line in lines if "咲季" in line["speaker"]), "")
    if first_saki:
        return f"{scene}：{topic_name} - {trim_text(first_saki, 26)}"
    first_mention = next((line["text_zh"] for line in lines if "咲季" in line["text_zh"]), "")
    if first_mention:
        return f"{scene}：{topic_name} - 提及咲季"
    return f"{scene}：{topic_name}"


def make_summary_text(lines: list[dict[str, str]], topics: list[str], saki_lines: list[dict[str, str]]) -> str:
    speakers = [line["speaker"] for line in lines if line["speaker"]]
    speaker_text = "、".join(name for name, _ in Counter(speakers).most_common(5))
    topic_text = "、".join(TOPIC_DEFS[topic][0] for topic in topics)
    direct_saki_lines = [line for line in lines if "咲季" in line["speaker"]]
    mention_lines = [line for line in lines if "咲季" in line["text_zh"] and "咲季" not in line["speaker"]]
    if direct_saki_lines:
        first = trim_text(direct_saki_lines[0]["text_zh"], 45)
        last = trim_text(direct_saki_lines[-1]["text_zh"], 45)
        return f"本段涉及{speaker_text}等人，主题集中在{topic_text}。咲季以“{first}”切入，并在后续以“{last}”延续情绪或立场。"
    if mention_lines:
        speaker = mention_lines[0]["speaker"]
        mention = trim_text(mention_lines[0]["text_zh"], 60)
        return f"本段涉及{speaker_text}等人，主题集中在{topic_text}。咲季没有直接发言，但由{speaker}提及或评价，关键线索为“{mention}”。"
    evidence = trim_text(lines[0]["text_zh"], 60) if lines else ""
    return f"本段涉及{speaker_text}等人，主题集中在{topic_text}。对话中提及咲季相关信息，关键线索为“{evidence}”。"


def infer_saki_role(lines: list[dict[str, str]]) -> str:
    text = "\n".join(line["text_zh"] for line in lines)
    if "佑芽" in text or "妹妹" in text or "姐姐" in text:
        return "姐姐/竞争者：咲季与佑芽相关内容中常同时体现亲情、自豪和竞争意识。"
    if "制作人" in text:
        return "担当偶像/搭档：咲季与制作人围绕训练、胜利、信任和调侃展开互动。"
    if "对手" in text or "朋友" in text or "伙伴" in text:
        return "同学/对手：咲季在群像关系中以好胜、直率和带头推进对话的姿态出现。"
    if any("咲季" in line["speaker"] for line in lines):
        return "场景主导者：咲季直接参与对话，表达情绪、目标或判断。"
    return "被提及对象：咲季未必直接发言，但场景信息与她的关系或评价有关。"


def extract_relationship_signals(text: str) -> list[str]:
    signals: list[str] = []
    for name in ("制作人", "佑芽", "琴音", "手毬", "清夏", "麻央", "广", "莉波", "星南", "美铃", "莉莉娅"):
        if name in text:
            signals.append(name)
    for word in ("妹妹", "姐姐", "朋友", "对手", "伙伴", "家人", "父母"):
        if word in text:
            signals.append(word)
    return signals[:10]


def extract_keyword_hits(text: str, keywords: tuple[str, ...], limit: int) -> list[str]:
    return [keyword for keyword in keywords if keyword in text][:limit]


def pick_evidence_lines(lines: list[dict[str, str]], limit: int = 5) -> list[dict[str, str]]:
    scored: list[tuple[int, dict[str, str]]] = []
    for line in lines:
        combined = f"{line['speaker']}: {line['text_zh']}"
        score = 0
        if "咲季" in combined:
            score += 5
        score += sum(combined.count(keyword) for _, keywords in TOPIC_DEFS.values() for keyword in keywords)
        if score:
            scored.append((score, line))
    if not scored:
        scored = [(1, line) for line in lines[:limit]]
    picked = [line for _, line in sorted(scored, key=lambda item: -item[0])[:limit]]
    return [
        {
            "source_row": line["source_row"],
            "speaker": line["speaker"],
            "text_zh": trim_text(line["text_zh"], 120),
        }
        for line in picked
    ]


def render_topic_documents(summaries: list[dict[str, object]]) -> dict[str, str]:
    topic_map: dict[str, list[dict[str, object]]] = defaultdict(list)
    for summary in summaries:
        for topic in summary["topics"]:  # type: ignore[index]
            topic_map[topic].append(summary)

    docs: dict[str, str] = {}
    total = len(summaries)
    for topic, (title, _) in TOPIC_DEFS.items():
        rows = topic_map.get(topic, [])
        docs[topic] = render_topic_document(topic, title, rows, total)
    return docs


def render_topic_document(topic: str, title: str, rows: list[dict[str, object]], total_summaries: int) -> str:
    source_counts = Counter(str(row["source_kind"]) for row in rows)
    speaker_counts = Counter(speaker for row in rows for speaker in row["speakers"])  # type: ignore[index]
    lines = [
        f"# 花海咲季 {title}",
        "",
        "## 说明",
        "",
        "本文档由原始 CSV 全量对话轻清洗后，按章节/场景/对话块自动摘要生成。内容均为中文摘要，候选证据只保留少量行，并保留来源追溯信息。",
        "",
        "## TODO",
        "",
        "- [ ] 人工核对摘要是否误判剧情因果。",
        "- [ ] 将重复场景合并成更稳定的设定条目。",
        "- [ ] 标注哪些内容可进入 RAG，哪些只适合作训练风格参考。",
        "",
        "## 基础统计",
        "",
        f"- 全部相关场景摘要数：{total_summaries}",
        f"- 当前主题命中场景数：{len(rows)}",
        "",
        "### 来源分布",
        "",
        *[f"- {kind}: {count}" for kind, count in source_counts.most_common(12)],
        "",
        "### 高频参与者",
        "",
        *[f"- {speaker}: {count}" for speaker, count in speaker_counts.most_common(12)],
        "",
        "## 主题合并摘要",
        "",
        *render_merged_bullets(topic, rows),
        "",
        "## 场景摘要与来源",
        "",
    ]
    for row in rows:
        lines.extend(render_scene_summary(row))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_merged_bullets(topic: str, rows: list[dict[str, object]]) -> list[str]:
    if not rows:
        return ["- 暂无该主题场景。"]
    signals = Counter(signal for row in rows for signal in row.get("relationship_signals", []))  # type: ignore[arg-type]
    worlds = Counter(signal for row in rows for signal in row.get("worldbuilding_signals", []))  # type: ignore[arg-type]
    styles = Counter(signal for row in rows for signal in row.get("dialogue_style_signals", []))  # type: ignore[arg-type]
    bullets = [
        f"- 该主题共覆盖 {len(rows)} 个咲季相关场景，主要来源为 {format_counter_compact(Counter(str(row['source_kind']) for row in rows), 5)}。",
    ]
    if signals:
        bullets.append(f"- 高频关系线索：{format_counter_compact(signals, 8)}。")
    if worlds:
        bullets.append(f"- 高频世界观线索：{format_counter_compact(worlds, 8)}。")
    if styles:
        bullets.append(f"- 高频口吻线索：{format_counter_compact(styles, 8)}。")
    if topic == "character_arc":
        bullets.append("- 自动观察：咲季相关场景反复围绕目标感、训练/胜负、被夸后的反应、与制作人共同推进目标展开。")
    elif topic == "relationships":
        bullets.append("- 自动观察：制作人、佑芽以及同学/对手关系是最需要人工整理成稳定 RAG 事实的部分。")
    elif topic == "dialogue_style":
        bullets.append("- 自动观察：咲季的中文口吻应保留高能量、自信、直率、被调侃后反弹的节奏，但避免直接复刻长台词。")
    return bullets


def render_scene_summary(row: dict[str, object]) -> list[str]:
    lines = [
        f"### {row['summary_id']} | {row['title']}",
        "",
        f"- 来源：`{row['source_refs'][0]}`",  # type: ignore[index]
        f"- 章节/场景：`{row['chapter']}` / `{row['scene']}`",
        f"- 对话块：{format_dialogue_blocks(row)}",
        f"- 参与者：{'、'.join(row['speakers'])}",  # type: ignore[arg-type]
        f"- 主题：{'、'.join(TOPIC_DEFS[topic][0] for topic in row['topics'])}",  # type: ignore[index]
        f"- 摘要：{row['summary']}",
        f"- 咲季定位：{row['saki_role']}",
        f"- 摘要状态：{row.get('summary_status', 'unknown')}",
    ]
    if row.get("story_facts") or row.get("relationship_facts") or row.get("character_arc"):
        lines.append("- 结构化理解：")
        for label, key in [
            ("剧情事实", "story_facts"),
            ("关系事实", "relationship_facts"),
            ("角色弧线", "character_arc"),
            ("口吻模式", "dialogue_style"),
        ]:
            values = row.get(key, [])
            if isinstance(values, list) and values:
                lines.append(f"  - {label}：{'；'.join(str(value) for value in values[:4])}")
    lines.append("- 证据摘录：")
    for evidence in row["key_evidence"]:  # type: ignore[index]
        lines.append(f"  - `{row['source_file']}:{evidence['source_row']}` {evidence['speaker']}: {evidence['text_zh']}")
    return lines


def render_scene_card(row: dict[str, object]) -> str:
    lines = [
        f"# {row['scene_id']} {row['scene']}",
        "",
        "## 说明",
        "",
        "这是一张从原始 CSV 生成的花海咲季相关场景卡，用于 RAG 精确检索。摘要为中文，证据保留行号来源；完整对话在 `outputs/scene_chunks.jsonl` 中追溯。",
        "",
        "## 元数据",
        "",
        f"- 来源：`{row['source_refs'][0]}`",  # type: ignore[index]
        f"- 章节：`{row['chapter']}`",
        f"- 场景：`{row['scene']}`",
        f"- 行数：{row['line_count']}",
        f"- 对话块：{format_dialogue_blocks(row)}",
        f"- 参与者：{'、'.join(row['speakers'])}",  # type: ignore[arg-type]
        f"- 主题：{'、'.join(TOPIC_DEFS[topic][0] for topic in row['topics'])}",  # type: ignore[index]
        f"- 摘要状态：{row.get('summary_status', 'unknown')}",
        "",
        "## 场景理解摘要",
        "",
        f"- 摘要：{row['summary']}",
        f"- 咲季定位：{row['saki_role']}",
        f"- 咲季出场状态：{row.get('saki_presence', 'unknown')}",
        "",
        "## 结构化线索",
        "",
        f"- 关系线索：{'、'.join(row.get('relationship_signals', [])) or '无'}",  # type: ignore[arg-type]
        f"- 世界观线索：{'、'.join(row.get('worldbuilding_signals', [])) or '无'}",  # type: ignore[arg-type]
        f"- 口吻线索：{'、'.join(row.get('dialogue_style_signals', [])) or '无'}",  # type: ignore[arg-type]
        "",
        "## LLM 结构化理解",
        "",
        *format_structured_list("咲季行动/表达", row.get("saki_actions", [])),
        *format_structured_list("剧情事实", row.get("story_facts", [])),
        *format_structured_list("关系事实", row.get("relationship_facts", [])),
        *format_structured_list("世界观事实", row.get("worldbuilding_facts", [])),
        *format_structured_list("角色弧线", row.get("character_arc", [])),
        *format_structured_list("对话风格", row.get("dialogue_style", [])),
        "",
        "## 证据摘录",
        "",
    ]
    for evidence in row["key_evidence"]:  # type: ignore[index]
        lines.append(f"- `{row['source_file']}:{evidence['source_row']}` {evidence['speaker']}: {evidence['text_zh']}")
    lines.extend(
        [
            "",
            "## 人工修订区",
            "",
            "- TODO: 人工核对本场景真实剧情因果。",
            "- TODO: 补充可进入 RAG 的稳定事实。",
            "- TODO: 标记不应让模型复读的原作长台词。",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def format_structured_list(title: str, values: object) -> list[str]:
    if not isinstance(values, list) or not values:
        return [f"### {title}", "", "- 无", ""]
    return [f"### {title}", "", *[f"- {value}" for value in values], ""]


def format_dialogue_blocks(row: dict[str, object]) -> str:
    blocks = row.get("dialogue_blocks", [])
    if not blocks:
        return "1"
    return "；".join(
        f"block {block['block_index']} 行 {block['start_row']}-{block['end_row']}"
        for block in blocks  # type: ignore[union-attr]
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def source_kind(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).parts[0]
    except (ValueError, IndexError):
        return ""


def infer_chapter(root: Path, path: Path) -> str:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return ""
    if len(relative.parts) >= 2:
        return "/".join(relative.parts[:-1])
    return ""


def related_reason(source_file: str, lines: list[dict[str, str]]) -> str:
    lowered = source_file.lower()
    reasons: list[str] = []
    if "/hski/" in lowered or "_hski" in lowered or "-hski" in lowered:
        reasons.append("source_path_hski")
    if any("咲季" in line["speaker"] for line in lines):
        reasons.append("saki_speaker")
    if any("咲季" in line["text_zh"] or "咲季" in line["text_ja"] for line in lines):
        reasons.append("saki_mentioned")
    return ",".join(reasons) or "related"


def trim_text(value: object, limit: int) -> str:
    text = normalize_text(value)
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def format_counter_compact(counter: Counter[str], limit: int) -> str:
    return "、".join(f"{key}({value})" for key, value in counter.most_common(limit))


def safe_filename(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return safe[:80] or "scene"
