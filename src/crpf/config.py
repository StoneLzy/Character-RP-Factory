from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - exercised only in lean local envs
    yaml = None


DEFAULT_CONFIG_PATH = Path("config.yaml")


DEFAULT_COLUMN_ALIASES: dict[str, list[str]] = {
    "line_id": ["id", "dialogue_id", "line_id"],
    "speaker": ["name", "speaker", "character", "说话人"],
    "text": ["text", "dialogue", "line", "台词"],
    "translation": ["trans", "translation", "zh", "中文"],
    "chapter": ["chapter", "章节"],
    "scene": ["scene", "场景"],
    "order": ["order", "index", "顺序"],
}


@dataclass(frozen=True)
class RagConfig:
    collection_name: str = "hski_character_rag"
    embedding_model: str = "bge-m3"
    chat_model: str = "qwen3.5:9b"
    ollama_base_url: str = "http://localhost:11434"
    chunk_size: int = 500
    chunk_overlap: int = 80
    top_k: int = 5


@dataclass(frozen=True)
class ProjectConfig:
    raw_csv_dir: Path = Path("CSV")
    output_dir: Path = Path("outputs/hski")
    summary_output_dir: Path = Path("outputs")
    processed_dir: Path = Path("data/processed")
    rag_docs_dir: Path = Path("data/rag_docs")
    chroma_dir: Path = Path("data/chroma_db")
    target_names: tuple[str, ...] = ("咲季", "花海咲季", "hski")
    user_names: tuple[str, ...] = ("{user}", "プロデューサー", "制作人")
    column_aliases: dict[str, list[str]] = field(default_factory=lambda: dict(DEFAULT_COLUMN_ALIASES))
    rag: RagConfig = field(default_factory=RagConfig)
    previous_lines: int = 3
    min_response_chars: int = 2
    max_response_chars: int = 500
    min_quality_score: int = 70
    export_format: str = "chatml"
    max_context_chars: int = 1200
    excluded_speakers: tuple[str, ...] = ("info", "译者", "select")
    bad_patterns: tuple[str, ...] = ()

    @property
    def merged_csv(self) -> Path:
        return self.output_dir / "merged.csv"

    @property
    def cleaned_csv(self) -> Path:
        return self.output_dir / "cleaned.csv"

    @property
    def character_lines_csv(self) -> Path:
        return self.output_dir / "character_lines.csv"

    @property
    def samples_with_context_csv(self) -> Path:
        return self.output_dir / "samples_with_context.csv"

    @property
    def good_samples_csv(self) -> Path:
        return self.output_dir / "good_samples.csv"

    @property
    def review_samples_csv(self) -> Path:
        return self.output_dir / "review_samples.csv"

    @property
    def training_samples_jsonl(self) -> Path:
        return self.output_dir / "training_samples.jsonl"

    @property
    def training_samples_ja_jsonl(self) -> Path:
        return self.output_dir / "training_samples_ja.jsonl"

    @property
    def training_samples_zh_jsonl(self) -> Path:
        return self.output_dir / "training_samples_zh.jsonl"


def _as_path(value: Any, default: str) -> Path:
    if value is None:
        return Path(default)
    return Path(str(value))


def _as_tuple(value: Any, default: tuple[str, ...]) -> tuple[str, ...]:
    if value is None:
        return default
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


def load_config(path: str | Path | None = None) -> ProjectConfig:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    data: dict[str, Any] = {}
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as handle:
            if yaml is not None:
                loaded = yaml.safe_load(handle) or {}
            else:
                loaded = parse_simple_yaml(handle.read())
            if not isinstance(loaded, dict):
                raise ValueError(f"Config must be a mapping: {config_path}")
            data = loaded

    paths = data.get("paths", {})
    character = data.get("character", {})
    columns = data.get("columns", {})
    context = data.get("context", {})
    cleaning = data.get("cleaning", {})
    quality = data.get("quality", {})
    rag = data.get("rag", {})
    training = data.get("training", {})

    aliases = DEFAULT_COLUMN_ALIASES | (columns.get("aliases") or {})

    return ProjectConfig(
        raw_csv_dir=_as_path(paths.get("raw_csv_dir"), "CSV"),
        output_dir=_as_path(paths.get("output_dir"), "outputs/hski"),
        summary_output_dir=_as_path(paths.get("summary_output_dir"), "outputs"),
        processed_dir=_as_path(paths.get("processed_dir"), "data/processed"),
        rag_docs_dir=_as_path(paths.get("rag_docs_dir"), "data/rag_docs"),
        chroma_dir=_as_path(paths.get("chroma_dir"), "data/chroma_db"),
        target_names=_as_tuple(character.get("target_names"), ("咲季", "花海咲季", "hski")),
        user_names=_as_tuple(character.get("user_names"), ("{user}", "プロデューサー", "制作人")),
        column_aliases=aliases,
        rag=RagConfig(
            collection_name=str(rag.get("collection_name", "hski_character_rag")),
            embedding_model=str(rag.get("embedding_model", "bge-m3")),
            chat_model=str(rag.get("chat_model", "qwen3.5:9b")),
            ollama_base_url=str(rag.get("ollama_base_url", "http://localhost:11434")),
            chunk_size=int(rag.get("chunk_size", 500)),
            chunk_overlap=int(rag.get("chunk_overlap", 80)),
            top_k=int(rag.get("top_k", 5)),
        ),
        previous_lines=int(context.get("previous_lines", 3)),
        min_response_chars=int(cleaning.get("min_response_chars", 2)),
        max_response_chars=int(cleaning.get("max_response_chars", 500)),
        min_quality_score=int(quality.get("min_quality_score", 70)),
        export_format=str(training.get("format", "chatml")),
        max_context_chars=int(training.get("max_context_chars", 1200)),
        excluded_speakers=_as_tuple(cleaning.get("excluded_speakers"), ("info", "译者", "select")),
        bad_patterns=_as_tuple(cleaning.get("bad_patterns"), ()),
    )


def parse_simple_yaml(text: str) -> dict[str, Any]:
    """Parse the small YAML subset used by config.example.yaml.

    This keeps the CLI usable on lean Python environments where PyYAML is not
    installed. It intentionally supports only nested mappings and scalar lists.
    """
    lines = [
        line.rstrip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    parsed, _ = _parse_block(lines, 0, 0)
    if not isinstance(parsed, dict):
        raise ValueError("Config root must be a mapping")
    return parsed


def _parse_block(lines: list[str], index: int, indent: int) -> tuple[Any, int]:
    is_list = False
    cursor = index
    while cursor < len(lines):
        current = lines[cursor]
        current_indent = len(current) - len(current.lstrip(" "))
        if current_indent < indent:
            return {}, cursor
        if current_indent == indent:
            is_list = current.lstrip().startswith("- ")
            break
        cursor += 1

    if is_list:
        values: list[Any] = []
        while index < len(lines):
            line = lines[index]
            current_indent = len(line) - len(line.lstrip(" "))
            if current_indent < indent:
                break
            if current_indent > indent:
                index += 1
                continue
            stripped = line.strip()
            if not stripped.startswith("- "):
                break
            values.append(_parse_scalar(stripped[2:].strip()))
            index += 1
        return values, index

    values: dict[str, Any] = {}
    while index < len(lines):
        line = lines[index]
        current_indent = len(line) - len(line.lstrip(" "))
        if current_indent < indent:
            break
        if current_indent > indent:
            index += 1
            continue
        key, separator, raw_value = line.strip().partition(":")
        if not separator:
            index += 1
            continue
        raw_value = raw_value.strip()
        if raw_value:
            values[key] = _parse_scalar(raw_value)
            index += 1
        else:
            nested, index = _parse_block(lines, index + 1, indent + 2)
            values[key] = nested
    return values, index


def _parse_scalar(value: str) -> Any:
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        return value
