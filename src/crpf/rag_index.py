from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import requests


SCENE_ID_RE = re.compile(r"scene-\d{5}")
HEADING_RE = re.compile(r"^(#{1,4})\s+(.+?)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class RagChunk:
    id: str
    text: str
    metadata: dict[str, str | int | float | bool]


@dataclass(frozen=True)
class RagIndexStats:
    documents: int
    chunks: int
    collection_name: str
    chroma_dir: Path
    embedding_model: str
    backend: str


@dataclass(frozen=True)
class RagSearchResult:
    rank: int
    distance: float | None
    text: str
    metadata: dict[str, Any]


TOPIC_BY_FILENAME = {
    "character_profile.md": "character_profile",
    "plot_summary.md": "plot_summary",
    "relationships.md": "relationships",
    "worldbuilding.md": "worldbuilding",
    "team_story.md": "team_story",
    "dialogue_patterns.md": "dialogue_patterns",
}


def build_rag_index(
    rag_docs_dir: Path,
    chroma_dir: Path,
    collection_name: str,
    embedding_model: str,
    ollama_base_url: str,
    chunk_size: int = 500,
    chunk_overlap: int = 80,
    reset: bool = False,
    backend: str = "auto",
) -> RagIndexStats:
    """Chunk Markdown RAG docs, embed them with Ollama, and write a vector index."""
    chunks = build_rag_chunks(rag_docs_dir, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not chunks:
        raise ValueError(f"No RAG Markdown chunks found under {rag_docs_dir}")

    embeddings = ollama_embed_texts(
        [chunk.text for chunk in chunks],
        model=embedding_model,
        base_url=ollama_base_url,
    )
    resolved_backend = resolve_backend(backend)
    if resolved_backend == "chroma":
        collection = open_chroma_collection(chroma_dir, collection_name, reset=reset)
        collection.upsert(
            ids=[chunk.id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            metadatas=[chunk.metadata for chunk in chunks],
            embeddings=embeddings,
        )
    else:
        write_simple_vector_index(chroma_dir, collection_name, chunks, embeddings)

    doc_count = len({chunk.metadata["source_path"] for chunk in chunks})
    return RagIndexStats(
        documents=doc_count,
        chunks=len(chunks),
        collection_name=collection_name,
        chroma_dir=chroma_dir,
        embedding_model=embedding_model,
        backend=resolved_backend,
    )


def query_rag_index(
    query: str,
    chroma_dir: Path,
    collection_name: str,
    embedding_model: str,
    ollama_base_url: str,
    top_k: int = 5,
    backend: str = "auto",
) -> list[RagSearchResult]:
    """Search the local ChromaDB RAG index with an Ollama query embedding."""
    if not query.strip():
        raise ValueError("query must not be empty")

    embedding = ollama_embed_texts([query], model=embedding_model, base_url=ollama_base_url)[0]
    resolved_backend = resolve_backend(backend)
    if resolved_backend == "simple":
        return query_simple_vector_index(embedding, chroma_dir, collection_name, top_k=top_k)

    collection = open_chroma_collection(chroma_dir, collection_name, reset=False)
    raw = collection.query(
        query_embeddings=[embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    documents = raw.get("documents", [[]])[0]
    metadatas = raw.get("metadatas", [[]])[0]
    distances = raw.get("distances", [[]])[0]
    results: list[RagSearchResult] = []
    for index, document in enumerate(documents):
        distance = distances[index] if index < len(distances) else None
        metadata = metadatas[index] if index < len(metadatas) and metadatas[index] else {}
        results.append(
            RagSearchResult(
                rank=index + 1,
                distance=float(distance) if distance is not None else None,
                text=document,
                metadata=dict(metadata),
            )
        )
    return results


def resolve_backend(backend: str) -> str:
    if backend not in {"auto", "chroma", "simple"}:
        raise ValueError("backend must be 'auto', 'chroma', or 'simple'")
    if backend == "auto":
        return "chroma" if is_chromadb_available() else "simple"
    if backend == "chroma" and not is_chromadb_available():
        raise RuntimeError(
            "Missing optional dependency 'chromadb'. Install it after network is available with: "
            'python3 -m pip install -e ".[rag]"'
        )
    return backend


def is_chromadb_available() -> bool:
    try:
        import chromadb  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True


def open_chroma_collection(chroma_dir: Path, collection_name: str, reset: bool = False) -> Any:
    try:
        import chromadb
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError(
            "Missing optional dependency 'chromadb'. Install it after network is available with: "
            'python3 -m pip install -e ".[rag]"'
        ) from exc

    chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_dir))
    if reset:
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def write_simple_vector_index(
    chroma_dir: Path,
    collection_name: str,
    chunks: list[RagChunk],
    embeddings: list[list[float]],
) -> Path:
    if len(chunks) != len(embeddings):
        raise ValueError("chunks and embeddings length mismatch")
    chroma_dir.mkdir(parents=True, exist_ok=True)
    path = simple_vector_index_path(chroma_dir, collection_name)
    with path.open("w", encoding="utf-8") as handle:
        for chunk, embedding in zip(chunks, embeddings):
            record = {
                "id": chunk.id,
                "document": chunk.text,
                "metadata": chunk.metadata,
                "embedding": embedding,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def query_simple_vector_index(
    query_embedding: list[float],
    chroma_dir: Path,
    collection_name: str,
    top_k: int,
) -> list[RagSearchResult]:
    path = simple_vector_index_path(chroma_dir, collection_name)
    if not path.exists():
        raise RuntimeError(f"Missing simple vector index: {path}; run build-rag-index first.")

    scored: list[tuple[float, dict[str, Any]]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            score = cosine_similarity(query_embedding, record["embedding"])
            scored.append((score, record))
    scored.sort(key=lambda item: item[0], reverse=True)

    results: list[RagSearchResult] = []
    for rank, (score, record) in enumerate(scored[:top_k], start=1):
        results.append(
            RagSearchResult(
                rank=rank,
                distance=1.0 - score,
                text=str(record["document"]),
                metadata=dict(record.get("metadata") or {}),
            )
        )
    return results


def simple_vector_index_path(chroma_dir: Path, collection_name: str) -> Path:
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", collection_name).strip("_") or "rag"
    return chroma_dir / f"{safe_name}.simple.jsonl"


def cosine_similarity(left: list[float], right: list[float]) -> float:
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for left_value, right_value in zip(left, right):
        dot += left_value * right_value
        left_norm += left_value * left_value
        right_norm += right_value * right_value
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / ((left_norm ** 0.5) * (right_norm ** 0.5))


def build_rag_chunks(rag_docs_dir: Path, chunk_size: int = 500, chunk_overlap: int = 80) -> list[RagChunk]:
    paths = discover_rag_markdown_files(rag_docs_dir)
    chunks: list[RagChunk] = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        if is_scene_card(path, rag_docs_dir):
            chunks.append(build_scene_chunk(path, text, rag_docs_dir))
        else:
            chunks.extend(build_document_chunks(path, text, rag_docs_dir, chunk_size, chunk_overlap))
    return chunks


def discover_rag_markdown_files(rag_docs_dir: Path) -> list[Path]:
    if not rag_docs_dir.exists():
        return []
    top_level = [
        path
        for path in rag_docs_dir.glob("*.md")
        if path.name in TOPIC_BY_FILENAME
    ]
    scene_cards = sorted((rag_docs_dir / "scenes").glob("*.md")) if (rag_docs_dir / "scenes").exists() else []
    return sorted(top_level) + scene_cards


def is_scene_card(path: Path, rag_docs_dir: Path) -> bool:
    try:
        return path.relative_to(rag_docs_dir).parts[0] == "scenes"
    except ValueError:
        return False


def build_scene_chunk(path: Path, text: str, rag_docs_dir: Path) -> RagChunk:
    rel_path = path.relative_to(rag_docs_dir).as_posix()
    scene_id = extract_first_scene_id(text) or extract_first_scene_id(path.name) or path.stem
    metadata = normalize_metadata(
        {
            "doc_type": "scene_card",
            "topic": "scene",
            "source_doc": path.name,
            "source_path": rel_path,
            "scene_id": scene_id,
            "scene_ids": json.dumps([scene_id], ensure_ascii=False),
            "heading_path": first_heading(text) or scene_id,
            "source_csv": extract_bullet_value(text, "来源"),
            "chapter": extract_bullet_value(text, "章节"),
            "scene": extract_bullet_value(text, "场景"),
            "saki_presence": extract_bullet_value(text, "咲季出场状态"),
            "topics": extract_bullet_value(text, "主题"),
        }
    )
    return RagChunk(id=f"{scene_id}::card", text=compact_markdown(text), metadata=metadata)


def build_document_chunks(
    path: Path,
    text: str,
    rag_docs_dir: Path,
    chunk_size: int,
    chunk_overlap: int,
) -> list[RagChunk]:
    rel_path = path.relative_to(rag_docs_dir).as_posix()
    topic = TOPIC_BY_FILENAME.get(path.name, path.stem)
    sections = split_markdown_sections(text)
    chunks: list[RagChunk] = []
    chunk_index = 0
    for heading_path, section_text in sections:
        if is_low_information_chunk(section_text):
            continue
        for part in split_text_by_paragraphs(section_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap):
            if is_low_information_chunk(part):
                continue
            scene_ids = sorted(set(SCENE_ID_RE.findall(part)))
            chunk_id = stable_chunk_id(rel_path, chunk_index, part)
            metadata = normalize_metadata(
                {
                    "doc_type": "summary_doc",
                    "topic": topic,
                    "source_doc": path.name,
                    "source_path": rel_path,
                    "scene_id": scene_ids[0] if scene_ids else "",
                    "scene_ids": json.dumps(scene_ids, ensure_ascii=False),
                    "heading_path": " > ".join(heading_path),
                }
            )
            chunks.append(RagChunk(id=chunk_id, text=part, metadata=metadata))
            chunk_index += 1
    return chunks


def is_low_information_chunk(text: str) -> bool:
    content_lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    content = " ".join(content_lines).strip()
    return len(content) < 20


def split_markdown_sections(text: str) -> list[tuple[list[str], str]]:
    lines = text.splitlines()
    sections: list[tuple[list[str], list[str]]] = []
    current_heading: list[str] = []
    current_lines: list[str] = []

    for line in lines:
        match = re.match(r"^(#{1,4})\s+(.+?)\s*$", line)
        if match:
            if current_lines:
                sections.append((current_heading or ["文档开头"], current_lines))
                current_lines = []
            level = len(match.group(1))
            title = match.group(2).strip()
            current_heading = current_heading[: level - 1] + [title]
        current_lines.append(line)

    if current_lines:
        sections.append((current_heading or ["文档开头"], current_lines))

    return [(heading, compact_markdown("\n".join(section_lines))) for heading, section_lines in sections if compact_markdown("\n".join(section_lines))]


def split_text_by_paragraphs(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    clean = compact_markdown(text)
    if len(clean) <= chunk_size:
        return [clean]

    paragraphs = re.split(r"\n\s*\n", clean)
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = paragraph if not current else f"{current}\n\n{paragraph}"
        if len(candidate) <= chunk_size or not current:
            current = candidate
            continue
        chunks.append(current)
        overlap = current[-chunk_overlap:] if chunk_overlap > 0 else ""
        current = f"{overlap}\n\n{paragraph}".strip() if overlap else paragraph
    if current:
        chunks.append(current)
    return chunks


def ollama_embed_texts(
    texts: Iterable[str],
    model: str,
    base_url: str,
    batch_size: int = 16,
    timeout: int = 120,
) -> list[list[float]]:
    inputs = [text for text in texts]
    if not inputs:
        return []

    session = requests.Session()
    session.trust_env = False
    embeddings: list[list[float]] = []
    endpoint = f"{base_url.rstrip('/')}/api/embed"

    for offset in range(0, len(inputs), batch_size):
        batch = inputs[offset : offset + batch_size]
        response = session.post(
            endpoint,
            json={"model": model, "input": batch},
            timeout=timeout,
        )
        if response.status_code == 404:
            embeddings.extend(ollama_embed_texts_legacy(session, batch, model, base_url, timeout))
            continue
        response.raise_for_status()
        payload = response.json()
        batch_embeddings = payload.get("embeddings")
        if not isinstance(batch_embeddings, list):
            raise RuntimeError(f"Ollama embedding response missing 'embeddings': {payload}")
        embeddings.extend([[float(value) for value in item] for item in batch_embeddings])
    return embeddings


def ollama_embed_texts_legacy(
    session: requests.Session,
    texts: list[str],
    model: str,
    base_url: str,
    timeout: int,
) -> list[list[float]]:
    endpoint = f"{base_url.rstrip('/')}/api/embeddings"
    embeddings: list[list[float]] = []
    for text in texts:
        response = session.post(endpoint, json={"model": model, "prompt": text}, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
        embedding = payload.get("embedding")
        if not isinstance(embedding, list):
            raise RuntimeError(f"Ollama legacy embedding response missing 'embedding': {payload}")
        embeddings.append([float(value) for value in embedding])
    return embeddings


def stable_chunk_id(rel_path: str, index: int, text: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    return f"{rel_path}::{index:04d}::{digest}"


def extract_first_scene_id(text: str) -> str:
    match = SCENE_ID_RE.search(text)
    return match.group(0) if match else ""


def first_heading(text: str) -> str:
    match = HEADING_RE.search(text)
    return match.group(2).strip() if match else ""


def extract_bullet_value(text: str, label: str) -> str:
    pattern = re.compile(rf"^\s*-\s*{re.escape(label)}：\s*(.+?)\s*$", re.MULTILINE)
    match = pattern.search(text)
    if not match:
        return ""
    value = match.group(1).strip()
    return value.replace("`", "")


def compact_markdown(text: str) -> str:
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    compacted = "\n".join(lines).strip()
    return re.sub(r"\n{3,}", "\n\n", compacted)


def normalize_metadata(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
    normalized: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        if value is None:
            normalized[key] = ""
        elif isinstance(value, (str, int, float, bool)):
            normalized[key] = value
        else:
            normalized[key] = json.dumps(value, ensure_ascii=False)
    return normalized
