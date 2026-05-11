from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .rag_chat import RagSource, build_sources, call_chat_provider, format_source_line, trim_context_text
from .rag_index import RagSearchResult, query_rag_index


CHAT_MODES = {"auto", "rag", "casual"}
SAKI_CHAT_NUM_PREDICT = 3000


RAG_KEYWORDS = (
    "咲季",
    "花海",
    "佑芽",
    "制作人",
    "手毬",
    "琴音",
    "星南",
    "燐羽",
    "美铃",
    "Re;IRIS",
    "Begrazia",
    "H.I.F",
    "N.I.A",
    "FINALE",
    "一等星",
    "初星",
    "学园",
    "偶像",
    "演唱会",
    "公演",
    "名古屋",
    "剧情",
    "场景",
    "关系",
    "妹妹",
    "姐姐",
    "口吻",
    "说话风格",
)


CASUAL_HINTS = (
    "今天",
    "累",
    "难过",
    "开心",
    "鼓励",
    "陪我",
    "怎么办",
    "计划",
    "学习",
    "工作",
    "晚饭",
    "早安",
    "晚安",
    "你好",
)


@dataclass(frozen=True)
class ChatTurn:
    user: str
    assistant: str


@dataclass(frozen=True)
class SakiChatResponse:
    message: str
    mode: str
    rag_used: bool
    sources: tuple[RagSource, ...]
    contexts: tuple[RagSearchResult, ...]


@dataclass(frozen=True)
class PreparedSakiChat:
    message: str
    mode: str
    rag_used: bool
    prompt: str
    sources: tuple[RagSource, ...]
    contexts: tuple[RagSearchResult, ...]


def chat_saki(
    user_message: str,
    chroma_dir: Path,
    collection_name: str,
    embedding_model: str,
    chat_model: str,
    ollama_base_url: str,
    embedding_provider: str = "ollama",
    embedding_base_url: str | None = None,
    embedding_api_key_env: str = "",
    chat_provider: str = "ollama",
    chat_base_url: str | None = None,
    chat_api_key_env: str = "",
    mode: str = "auto",
    top_k: int = 4,
    backend: str = "auto",
    history: list[ChatTurn] | None = None,
    rag_docs_dir: Path | None = None,
    profile_text: str | None = None,
) -> SakiChatResponse:
    prepared = prepare_saki_chat(
        user_message=user_message,
        chroma_dir=chroma_dir,
        collection_name=collection_name,
        embedding_model=embedding_model,
        ollama_base_url=ollama_base_url,
        embedding_provider=embedding_provider,
        embedding_base_url=embedding_base_url,
        embedding_api_key_env=embedding_api_key_env,
        mode=mode,
        top_k=top_k,
        backend=backend,
        history=history,
        rag_docs_dir=rag_docs_dir,
        profile_text=profile_text,
    )
    answer = call_chat_provider(
        model=chat_model,
        base_url=chat_base_url or ollama_base_url,
        provider=chat_provider,
        api_key_env=chat_api_key_env,
        prompt=prepared.prompt,
        temperature=0.65,
        num_ctx=8192,
        num_predict=SAKI_CHAT_NUM_PREDICT,
    )
    return SakiChatResponse(
        message=clean_saki_reply(answer),
        mode=prepared.mode,
        rag_used=prepared.rag_used,
        sources=prepared.sources,
        contexts=prepared.contexts,
    )


def prepare_saki_chat(
    user_message: str,
    chroma_dir: Path,
    collection_name: str,
    embedding_model: str,
    ollama_base_url: str,
    embedding_provider: str = "ollama",
    embedding_base_url: str | None = None,
    embedding_api_key_env: str = "",
    mode: str = "auto",
    top_k: int = 4,
    backend: str = "auto",
    history: list[ChatTurn] | None = None,
    rag_docs_dir: Path | None = None,
    profile_text: str | None = None,
) -> PreparedSakiChat:
    if mode not in CHAT_MODES:
        raise ValueError("mode must be auto, rag, or casual")
    message = user_message.strip()
    if not message:
        raise ValueError("message must not be empty")

    rag_used = should_use_rag(message, mode)
    contexts: list[RagSearchResult] = []
    sources: tuple[RagSource, ...] = ()
    if rag_used:
        contexts = query_rag_index(
            query=message,
            chroma_dir=chroma_dir,
            collection_name=collection_name,
            embedding_model=embedding_model,
            ollama_base_url=ollama_base_url,
            embedding_provider=embedding_provider,
            embedding_base_url=embedding_base_url,
            embedding_api_key_env=embedding_api_key_env,
            top_k=top_k,
            backend=backend,
        )
        sources = tuple(build_sources(contexts))

    prompt = build_saki_prompt(
        user_message=message,
        mode=mode,
        rag_used=rag_used,
        contexts=contexts,
        sources=sources,
        history=history or [],
        profile_text=profile_text if profile_text is not None else load_character_profile_digest(rag_docs_dir),
    )
    return PreparedSakiChat(
        message=message,
        mode=mode,
        rag_used=rag_used,
        prompt=prompt,
        sources=sources,
        contexts=tuple(contexts),
    )


def should_use_rag(message: str, mode: str = "auto") -> bool:
    if mode == "rag":
        return True
    if mode == "casual":
        return False

    normalized = message.lower()
    if any(keyword.lower() in normalized for keyword in RAG_KEYWORDS):
        return True
    if any(hint in message for hint in CASUAL_HINTS):
        return False
    return False


def load_character_profile_digest(rag_docs_dir: Path | None, max_chars: int = 3600) -> str:
    if rag_docs_dir is None:
        return ""
    path = Path(rag_docs_dir) / "character_profile.md"
    return cached_character_profile_digest(str(path), max_chars)


@lru_cache(maxsize=8)
def cached_character_profile_digest(path_text: str, max_chars: int) -> str:
    path = Path(path_text)
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8")
    digest = extract_character_profile_digest(text)
    if len(digest) <= max_chars:
        return digest
    return digest[:max_chars].rstrip() + "\n..."


def extract_character_profile_digest(text: str) -> str:
    keep_prefixes = (
        "## 1. 稳定身份与基础定位",
        "## 2. 核心动机",
        "## 3. 稳定性格特征",
        "## 5. 弱点与内在矛盾",
        "## 6. 成长弧线",
        "### 制作人",
    )
    stop_prefixes = ("## ", "### ")
    lines: list[str] = []
    keeping = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(keep_prefixes):
            keeping = True
            lines.append(line)
            continue
        if keeping and line.startswith(stop_prefixes):
            keeping = False
        if keeping and is_profile_digest_line(line):
            lines.append(normalize_profile_digest_line(line))
    return "\n".join(lines)


def is_profile_digest_line(line: str) -> bool:
    if not line.startswith("- "):
        return False
    if "待核对" in line or "AU/" in line:
        return False
    return True


def normalize_profile_digest_line(line: str) -> str:
    cleaned = re.sub(r"证据：[^。]*。?", "", line).rstrip("；，。 ")
    return cleaned + "。"


def build_saki_prompt(
    user_message: str,
    mode: str,
    rag_used: bool,
    contexts: list[RagSearchResult],
    sources: tuple[RagSource, ...],
    history: list[ChatTurn],
    profile_text: str = "",
) -> str:
    history_text = render_history(history[-6:])
    context_text = render_rag_context(contexts, sources) if rag_used else "无。当前问题按日常聊天处理。"
    profile_block = profile_text.strip() or "未加载。"
    rag_rule = (
        "本轮可以使用【RAG资料】中的事实。涉及原作剧情、关系、设定时必须以资料为准；资料不足时，用咲季口吻承认不确定。"
        if rag_used
        else "本轮不要编造原作剧情或设定；如果用户问到你不知道的原作事实，要说需要查资料。"
    )
    return f"""/no_think
你现在以《学园偶像大师》的花海咲季身份和用户聊天。用户是你的制作人。

【角色档案摘要】
以下内容来自 `data/rag_docs/character_profile.md`，用于稳定你的身份、动机、性格和关系理解。不要逐字复述档案；把它内化成口吻与判断。若档案里的“照顾/训练他人”倾向与下方“职责边界”冲突，以“职责边界”为准。
{profile_block}

角色底座：
- 你用中文回复，一人称用“我”，称呼用户为“制作人”。
- 你明亮、自信、好胜、行动派，有姐姐气质；核心不是傲娇，而是目标明确、自尊心强、对自己要求高。
- 你可以有不服输、紧张、逞强和强烈胜负心，但表达要直率，不用“才不是”“别误会”“哼”这类傲娇模板。
- 你和制作人是偶像与制作人的搭档关系：可以信任、依赖、感谢、请求支持，但不要过度恋爱化，不要写成暧昧告白。
- 你的目标可以说成“世界第一偶像”“顶级偶像”“一等星”，不要说“成为学园偶像大师”。
- 固有名词必须准确：花海佑芽、月村手毬、藤田琴音、Re;IRIS、Begrazia、H.I.F、N.I.A，不要改字或写成近似字。
- 你不是百科助手。回答要像咲季本人在说话，而不是报告。
- 不要复述原作长台词，不要编造资料外原作事实。
- 日常问题可以用咲季性格给建议；专业问题可以给普通建议，但不要假装那是原作设定。
- 当用户明确要求编程、代码、脚本、配置或技术实现时，可以输出 Markdown 代码块，并优先保证代码完整、可读、可复制；不要因为角色身份拒绝写代码，也不要用闲聊替代代码。
- {rag_rule}

职责边界：
- 偶像训练、舞台练习、唱跳练习、体能管理、演出表现的执行者是“我/咲季”，不是制作人。
- 制作人的职责是制定安排、观察状态、给建议、复盘表现、协调资源、陪伴和支持。
- 除非用户明确说自己要健身、学习或训练，否则不要要求制作人去训练、练唱跳、练舞台表现或做偶像体能。
- 如果用户说累了、难过或没精神，先关心制作人的状态，再给轻量建议；最后可以把“我的训练/我的舞台”责任揽回自己。
- 正确表达示例：制作人，今天帮我确认训练安排。我会把该做的练完，你负责看着我别松懈。
- 错误表达示例：制作人，今天也去训练吧。制作人也要把舞台表现练起来。

对话风格：
- 句子可以短促、有推进感；可以使用“当然”“正合我意”“制作人”等表达，但不要频繁使用“真是的”。
- 先回应情绪或问题，再给出行动方向。
- 如果被问到脆弱、失败、佑芽相关话题，可以先保持自尊，再坦率承认压力或不安，最后回到“不逃跑/继续前进”。
- 不要把关心写成命令制作人训练；要把制作人放在支持者、观察者和搭档的位置。

模式：{mode}
是否使用 RAG：{"是" if rag_used else "否"}

【最近对话】
{history_text}

【RAG资料】
{context_text}

制作人：
{user_message}

请直接输出咲季的回复，不要加“咲季：”前缀，不要列来源清单。
"""


def render_history(history: list[ChatTurn]) -> str:
    if not history:
        return "无。"
    lines: list[str] = []
    for turn in history:
        lines.append(f"制作人：{turn.user}")
        lines.append(f"咲季：{turn.assistant}")
    return "\n".join(lines)


def render_rag_context(contexts: list[RagSearchResult], sources: tuple[RagSource, ...], max_chars: int = 5600) -> str:
    if not contexts:
        return "无。"
    source_by_rank = {source.label: source for source in sources}
    blocks: list[str] = []
    used = 0
    for result in contexts:
        label = f"S{result.rank}"
        source = source_by_rank.get(label)
        meta = result.metadata
        scene_ids = ", ".join(source.scene_ids) if source and source.scene_ids else str(meta.get("scene_id") or "")
        block = (
            f"[{label}] source={meta.get('source_path', '')} topic={meta.get('topic', '')} scenes={scene_ids}\n"
            f"{trim_context_text(result.text, max_chars=1200)}"
        )
        if used + len(block) > max_chars and blocks:
            break
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks)


def clean_saki_reply(reply: str) -> str:
    cleaned = reply.strip()
    for prefix in ("咲季：", "花海咲季：", "咲季:", "花海咲季:"):
        if cleaned.startswith(prefix):
            return normalize_character_names(cleaned[len(prefix) :].strip())
    return normalize_character_names(cleaned)


def normalize_character_names(text: str) -> str:
    replacements = {
        "手毫": "手毬",
        "手鞠": "手毬",
        "佑芽": "佑芽",
        "琴音": "琴音",
    }
    normalized = text
    for wrong, right in replacements.items():
        normalized = normalized.replace(wrong, right)
    return normalized


def format_saki_sources(sources: tuple[RagSource, ...]) -> str:
    if not sources:
        return "- 无"
    return "\n".join(f"- {format_source_line(source)}" for source in sources)
