from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from .rag_index import RagSearchResult, query_rag_index


@dataclass(frozen=True)
class RagSource:
    label: str
    source_path: str
    topic: str
    scene_ids: tuple[str, ...]
    distance: float | None


@dataclass(frozen=True)
class RagAnswer:
    question: str
    answer: str
    sources: tuple[RagSource, ...]
    contexts: tuple[RagSearchResult, ...]


def ask_rag(
    question: str,
    chroma_dir: Path,
    collection_name: str,
    embedding_model: str,
    chat_model: str,
    ollama_base_url: str,
    top_k: int = 5,
    backend: str = "auto",
    max_context_chars: int = 6500,
) -> RagAnswer:
    """Answer a knowledge question using retrieved RAG chunks and a local Ollama chat model."""
    contexts = query_rag_index(
        query=question,
        chroma_dir=chroma_dir,
        collection_name=collection_name,
        embedding_model=embedding_model,
        ollama_base_url=ollama_base_url,
        top_k=top_k,
        backend=backend,
    )
    if not contexts:
        raise RuntimeError("RAG index returned no context; rebuild the index or try a broader question.")

    sources = tuple(build_sources(contexts))
    prompt = build_rag_prompt(question, contexts, sources, max_context_chars=max_context_chars)
    answer = call_ollama_chat(
        model=chat_model,
        ollama_base_url=ollama_base_url,
        prompt=prompt,
    )
    return RagAnswer(question=question, answer=answer, sources=sources, contexts=tuple(contexts))


def build_rag_prompt(
    question: str,
    contexts: list[RagSearchResult],
    sources: tuple[RagSource, ...],
    max_context_chars: int = 6500,
) -> str:
    blocks: list[str] = []
    used_chars = 0
    source_by_rank = {source.label: source for source in sources}
    for result in contexts:
        label = f"S{result.rank}"
        source = source_by_rank.get(label)
        meta = result.metadata
        scene_ids = ", ".join(source.scene_ids) if source and source.scene_ids else str(meta.get("scene_id") or "")
        header = (
            f"[{label}] source={meta.get('source_path', '')} "
            f"topic={meta.get('topic', '')} scenes={scene_ids}"
        )
        text = trim_context_text(result.text, max_chars=1400)
        block = f"{header}\n{text}"
        if used_chars + len(block) > max_context_chars and blocks:
            break
        blocks.append(block)
        used_chars += len(block)

    context_text = "\n\n".join(blocks)
    source_text = "\n".join(format_source_line(source) for source in sources)
    return f"""/no_think
你是《学园偶像大师》花海咲季 RAG 知识库的中文问答助手。

回答规则：
- 只能根据【RAG资料】回答，不要编造资料外设定。
- 如果资料不足，请明确说“资料不足”，并说明缺少什么。
- 用中文回答，优先给出稳定事实，再补充“根据资料推断”的内容。
- 不要大段复述原作台词，不要写成角色扮演。
- 涉及剧情、人物关系或口吻规则时，引用 [S1] 这类来源编号。
- 来源编号不是必须全部使用；只引用与问题直接相关的来源，弱相关资料可以忽略。
- 如果不同来源相关度不同，优先使用排名靠前、内容更直接的资料。
- 回答末尾不需要重新列来源，程序会单独输出来源列表。

问题：
{question.strip()}

【RAG资料】
{context_text}

【可用来源】
{source_text}
"""


def call_ollama_chat(
    model: str,
    ollama_base_url: str,
    prompt: str,
    temperature: float = 0.2,
    num_ctx: int = 8192,
    num_predict: int = 900,
    timeout: int = 300,
) -> str:
    url = ollama_base_url.rstrip("/") + "/api/chat"
    payload = {
        "model": model,
        "stream": False,
        "think": False,
        "options": {
            "temperature": temperature,
            "top_p": 0.9,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
        },
        "messages": [
            {
                "role": "system",
                "content": "你是严谨的中文 RAG 问答助手。只根据给定资料回答，不输出思考过程。",
            },
            {"role": "user", "content": prompt},
        ],
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=timeout) as response:
            raw = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Ollama chat request failed: {exc}") from exc

    content = str(raw.get("message", {}).get("content", "")).strip()
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    if not content:
        raise RuntimeError(f"Ollama chat response is empty: {raw}")
    return content


def stream_ollama_chat(
    model: str,
    ollama_base_url: str,
    prompt: str,
    temperature: float = 0.2,
    num_ctx: int = 8192,
    num_predict: int = 900,
    timeout: int = 300,
) -> Iterator[str]:
    url = ollama_base_url.rstrip("/") + "/api/chat"
    payload = {
        "model": model,
        "stream": True,
        "think": False,
        "options": {
            "temperature": temperature,
            "top_p": 0.9,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
        },
        "messages": [
            {
                "role": "system",
                "content": "你是严谨的中文 RAG 问答助手。只根据给定资料回答，不输出思考过程。",
            },
            {"role": "user", "content": prompt},
        ],
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=timeout) as response:
            for line in response:
                if not line.strip():
                    continue
                raw = json.loads(line.decode("utf-8"))
                if raw.get("error"):
                    raise RuntimeError(str(raw["error"]))
                content = str(raw.get("message", {}).get("content", ""))
                if content:
                    yield content
                if raw.get("done"):
                    break
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Ollama chat stream request failed: {exc}") from exc


def build_sources(contexts: list[RagSearchResult]) -> list[RagSource]:
    sources: list[RagSource] = []
    for result in contexts:
        meta = result.metadata
        sources.append(
            RagSource(
                label=f"S{result.rank}",
                source_path=str(meta.get("source_path", "")),
                topic=str(meta.get("topic", "")),
                scene_ids=parse_scene_ids(meta.get("scene_ids") or meta.get("scene_id")),
                distance=result.distance,
            )
        )
    return sources


def parse_scene_ids(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, list):
        return tuple(str(item) for item in value if str(item))
    raw = str(value).strip()
    if not raw:
        return ()
    try:
        loaded = json.loads(raw)
    except json.JSONDecodeError:
        loaded = None
    if isinstance(loaded, list):
        return tuple(str(item) for item in loaded if str(item))
    return tuple(re.findall(r"scene-\d{5}", raw)) or (raw,)


def format_source_line(source: RagSource) -> str:
    distance = f"{source.distance:.4f}" if source.distance is not None else "n/a"
    scene_ids = ", ".join(source.scene_ids) if source.scene_ids else "无"
    return f"[{source.label}] {source.source_path} | topic={source.topic} | scenes={scene_ids} | distance={distance}"


def format_sources(sources: tuple[RagSource, ...]) -> str:
    return "\n".join(f"- {format_source_line(source)}" for source in sources)


def trim_context_text(text: str, max_chars: int) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= max_chars:
        return compact
    return compact[:max_chars].rstrip() + "..."
