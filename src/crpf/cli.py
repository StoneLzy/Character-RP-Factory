from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from .cleaning import STANDARD_FIELDS, clean_rows, filter_character_lines
from .config import load_config
from .consolidated_rag import sync_consolidated_rag
from .consolidation_inputs import prepare_consolidation_inputs
from .context_builder import CONTEXT_SAMPLE_FIELDS, build_context_samples
from .exporters import export_jsonl
from .io_csv import merge_csv_tree, read_csv_any_encoding, write_csv
from .profile_builder import build_rag_docs
from .rag_chat import ask_rag, format_sources
from .rag_index import build_rag_index, query_rag_index
from .quality import REVIEW_FIELDS, score_samples
from .rag_review import build_rag_review_outputs
from .rag_summary_builder import build_llm_rag_summaries, build_raw_rag_summaries
from .saki_chat import CHAT_MODES, ChatTurn, chat_saki, format_saki_sources
from .webui import WebUISettings, run_webui


app = typer.Typer(help="Character-RP-Factory CSV cleaning and RP asset CLI.")
console = Console()


@app.command()
def init(config: Path = typer.Option(Path("config.example.yaml"), "--config", "-c")) -> None:
    """Create the standard workspace directories."""
    cfg = load_config(config)
    for path in [cfg.raw_csv_dir, cfg.output_dir, cfg.processed_dir, cfg.rag_docs_dir, cfg.chroma_dir]:
        path.mkdir(parents=True, exist_ok=True)
    console.print("[green]Workspace directories are ready.[/green]")


@app.command()
def merge(config: Path = typer.Option(Path("config.example.yaml"), "--config", "-c")) -> None:
    """Merge all CSV files under paths.raw_csv_dir."""
    cfg = load_config(config)
    rows = merge_csv_tree(cfg.raw_csv_dir)
    write_csv(cfg.merged_csv, rows)
    console.print(f"[green]Merged {len(rows)} rows -> {cfg.merged_csv}[/green]")


@app.command()
def clean(config: Path = typer.Option(Path("config.example.yaml"), "--config", "-c")) -> None:
    """Clean merged CSV rows and export target character lines."""
    cfg = load_config(config)
    if cfg.merged_csv.exists():
        raw_rows = read_csv_any_encoding(cfg.merged_csv)
    else:
        raw_rows = merge_csv_tree(cfg.raw_csv_dir)

    cleaned = clean_rows(
        raw_rows,
        aliases=cfg.column_aliases,
        excluded_speakers=cfg.excluded_speakers,
        bad_patterns=cfg.bad_patterns,
    )
    character_lines = filter_character_lines(
        cleaned,
        target_names=cfg.target_names,
        min_chars=cfg.min_response_chars,
        max_chars=cfg.max_response_chars,
    )

    write_csv(cfg.cleaned_csv, cleaned, fieldnames=STANDARD_FIELDS)
    write_csv(cfg.character_lines_csv, character_lines, fieldnames=STANDARD_FIELDS)
    console.print(f"[green]Cleaned {len(cleaned)} rows -> {cfg.cleaned_csv}[/green]")
    console.print(f"[green]Selected {len(character_lines)} target lines -> {cfg.character_lines_csv}[/green]")


@app.command("build-context")
def build_context(config: Path = typer.Option(Path("config.example.yaml"), "--config", "-c")) -> None:
    """Attach previous dialogue lines to each target character response."""
    cfg = load_config(config)
    if not cfg.cleaned_csv.exists():
        raise typer.BadParameter(f"Missing {cfg.cleaned_csv}; run clean first.")

    cleaned_rows = read_csv_any_encoding(cfg.cleaned_csv)
    samples = build_context_samples(
        cleaned_rows,
        target_names=cfg.target_names,
        previous_lines=cfg.previous_lines,
    )
    write_csv(cfg.samples_with_context_csv, samples, fieldnames=CONTEXT_SAMPLE_FIELDS)
    console.print(
        f"[green]Built {len(samples)} context samples "
        f"with previous_lines={cfg.previous_lines} -> {cfg.samples_with_context_csv}[/green]"
    )


@app.command()
def score(config: Path = typer.Option(Path("config.example.yaml"), "--config", "-c")) -> None:
    """Score context samples and create good/review CSV files."""
    cfg = load_config(config)
    if not cfg.samples_with_context_csv.exists():
        raise typer.BadParameter(f"Missing {cfg.samples_with_context_csv}; run build-context first.")

    samples = read_csv_any_encoding(cfg.samples_with_context_csv)
    good_samples, review_samples = score_samples(
        samples,
        min_response_chars=cfg.min_response_chars,
        max_response_chars=cfg.max_response_chars,
        min_quality_score=cfg.min_quality_score,
    )
    write_csv(cfg.good_samples_csv, good_samples, fieldnames=REVIEW_FIELDS)
    write_csv(cfg.review_samples_csv, review_samples, fieldnames=REVIEW_FIELDS)
    console.print(
        f"[green]Scored {len(review_samples)} samples; kept {len(good_samples)} "
        f"(threshold={cfg.min_quality_score})[/green]"
    )
    console.print(f"[green]Good samples -> {cfg.good_samples_csv}[/green]")
    console.print(f"[green]Review samples -> {cfg.review_samples_csv}[/green]")


@app.command("export-jsonl")
def export_jsonl_command(
    config: Path = typer.Option(Path("config.example.yaml"), "--config", "-c"),
    fmt: str | None = typer.Option(None, "--format", "-f", help="chatml or instruction"),
    language: str = typer.Option("both", "--language", "-l", help="ja, zh, or both"),
) -> None:
    """Export scored samples to LoRA/QLoRA JSONL."""
    cfg = load_config(config)
    source = cfg.good_samples_csv if cfg.good_samples_csv.exists() else cfg.samples_with_context_csv
    if not source.exists():
        raise typer.BadParameter(f"Missing {source}; run build-context and score first.")

    export_format = fmt or cfg.export_format
    if export_format not in {"chatml", "instruction"}:
        raise typer.BadParameter("format must be 'chatml' or 'instruction'")
    if language not in {"ja", "zh", "both"}:
        raise typer.BadParameter("language must be 'ja', 'zh', or 'both'")
    rows = read_csv_any_encoding(source)
    targets = []
    if language in {"ja", "both"}:
        targets.append(("ja", cfg.training_samples_ja_jsonl))
    if language in {"zh", "both"}:
        targets.append(("zh", cfg.training_samples_zh_jsonl))

    for lang, output_path in targets:
        count = export_jsonl(
            rows,
            output_path=output_path,
            export_format=export_format,
            max_context_chars=cfg.max_context_chars,
            language=lang,
        )
        console.print(f"[green]Exported {count} {export_format}/{lang} records -> {output_path}[/green]")


@app.command("build-rag-docs")
def build_rag_docs_command(config: Path = typer.Option(Path("config.example.yaml"), "--config", "-c")) -> None:
    """Generate editable RAG Markdown templates."""
    cfg = load_config(config)
    source = cfg.good_samples_csv if cfg.good_samples_csv.exists() else cfg.samples_with_context_csv
    if not source.exists():
        raise typer.BadParameter(f"Missing {source}; run build-context and score first.")

    rows = read_csv_any_encoding(source)
    paths = build_rag_docs(rows, cfg.rag_docs_dir)
    for path in paths:
        console.print(f"[green]Generated {path}[/green]")


@app.command("summarize-raw-rag")
def summarize_raw_rag(config: Path = typer.Option(Path("config.example.yaml"), "--config", "-c")) -> None:
    """Summarize Saki-related full scenes from raw CSV into RAG docs."""
    cfg = load_config(config)
    written = build_raw_rag_summaries(cfg.raw_csv_dir, cfg.summary_output_dir, cfg.rag_docs_dir)
    for label, path in written.items():
        console.print(f"[green]{label}: {path}[/green]")


@app.command("summarize-scenes-llm")
def summarize_scenes_llm(
    config: Path = typer.Option(Path("config.example.yaml"), "--config", "-c"),
    model: str = typer.Option("qwen3.5:9b", "--model", "-m"),
    base_url: str = typer.Option("http://localhost:11434", "--ollama-base-url"),
    limit: int | None = typer.Option(None, "--limit", help="Only summarize first N scenes for testing."),
    no_resume: bool = typer.Option(False, "--no-resume", help="Ignore existing LLM summaries."),
    summary_mode: str = typer.Option("fast", "--summary-mode", help="fast or full."),
) -> None:
    """Use a local Ollama LLM to write true scene summaries."""
    if summary_mode not in {"fast", "full"}:
        raise typer.BadParameter("summary-mode must be 'fast' or 'full'")
    cfg = load_config(config)
    written = build_llm_rag_summaries(
        cfg.raw_csv_dir,
        cfg.summary_output_dir,
        cfg.rag_docs_dir,
        model=model,
        ollama_base_url=base_url,
        limit=limit,
        resume=not no_resume,
        summary_mode=summary_mode,
    )
    for label, path in written.items():
        console.print(f"[green]{label}: {path}[/green]")


@app.command("review-rag")
def review_rag(config: Path = typer.Option(Path("config.example.yaml"), "--config", "-c")) -> None:
    """Write validation report and human review CSV for scene summaries."""
    cfg = load_config(config)
    written = build_rag_review_outputs(cfg.summary_output_dir)
    for label, path in written.items():
        console.print(f"[green]{label}: {path}[/green]")


@app.command("prepare-consolidation-inputs")
def prepare_consolidation_inputs_command(
    config: Path = typer.Option(Path("config.example.yaml"), "--config", "-c"),
) -> None:
    """Prepare Markdown input packs for web LLM RAG consolidation."""
    cfg = load_config(config)
    written = prepare_consolidation_inputs(cfg.summary_output_dir)
    for label, path in written.items():
        console.print(f"[green]{label}: {path}[/green]")


@app.command("sync-consolidated-rag")
def sync_consolidated_rag_command(
    config: Path = typer.Option(Path("config.example.yaml"), "--config", "-c"),
    source_dir: Path = typer.Option(Path("outputs/consolidation_outputs"), "--source-dir"),
) -> None:
    """Clean and sync web-consolidated RAG docs into data/rag_docs."""
    cfg = load_config(config)
    written = sync_consolidated_rag(source_dir, cfg.rag_docs_dir, cfg.summary_output_dir)
    for label, path in written.items():
        console.print(f"[green]{label}: {path}[/green]")


@app.command("build-rag-index")
def build_rag_index_command(
    config: Path = typer.Option(Path("config.example.yaml"), "--config", "-c"),
    embedding_model: str | None = typer.Option(None, "--embedding-model", "-m"),
    collection_name: str | None = typer.Option(None, "--collection-name"),
    backend: str = typer.Option("auto", "--backend", help="auto, chroma, or simple."),
    reset: bool = typer.Option(False, "--reset", help="Rebuild the existing local index before writing."),
) -> None:
    """Embed data/rag_docs Markdown files and build a local vector index."""
    cfg = load_config(config)
    try:
        stats = build_rag_index(
            rag_docs_dir=cfg.rag_docs_dir,
            chroma_dir=cfg.chroma_dir,
            collection_name=collection_name or cfg.rag.collection_name,
            embedding_model=embedding_model or cfg.rag.embedding_model,
            ollama_base_url=cfg.rag.ollama_base_url,
            chunk_size=cfg.rag.chunk_size,
            chunk_overlap=cfg.rag.chunk_overlap,
            reset=reset,
            backend=backend,
        )
    except Exception as exc:
        console.print(f"RAG index build failed: {exc}", style="red", markup=False)
        raise typer.Exit(1) from exc

    console.print(
        "[green]Built RAG index[/green] "
        f"collection={stats.collection_name} docs={stats.documents} chunks={stats.chunks} "
        f"model={stats.embedding_model} backend={stats.backend} path={stats.chroma_dir}"
    )


@app.command("rag-query")
def rag_query_command(
    query: str = typer.Argument(..., help="Question or search text."),
    config: Path = typer.Option(Path("config.example.yaml"), "--config", "-c"),
    embedding_model: str | None = typer.Option(None, "--embedding-model", "-m"),
    collection_name: str | None = typer.Option(None, "--collection-name"),
    backend: str = typer.Option("auto", "--backend", help="auto, chroma, or simple."),
    top_k: int | None = typer.Option(None, "--top-k", "-k"),
) -> None:
    """Search the local RAG vector index and print retrieved chunks."""
    cfg = load_config(config)
    try:
        results = query_rag_index(
            query=query,
            chroma_dir=cfg.chroma_dir,
            collection_name=collection_name or cfg.rag.collection_name,
            embedding_model=embedding_model or cfg.rag.embedding_model,
            ollama_base_url=cfg.rag.ollama_base_url,
            top_k=top_k or cfg.rag.top_k,
            backend=backend,
        )
    except Exception as exc:
        console.print(f"RAG query failed: {exc}", style="red", markup=False)
        raise typer.Exit(1) from exc

    for result in results:
        meta = result.metadata
        distance = f"{result.distance:.4f}" if result.distance is not None else "n/a"
        scene_ids = meta.get("scene_ids") or meta.get("scene_id") or ""
        console.print(
            f"\n[bold cyan]#{result.rank}[/bold cyan] distance={distance} "
            f"source={meta.get('source_path', '')} topic={meta.get('topic', '')} scenes={scene_ids}"
        )
        snippet = result.text.replace("\n", " ")
        if len(snippet) > 420:
            snippet = snippet[:420].rstrip() + "..."
        console.print(snippet, markup=False)


@app.command("rag-ask")
def rag_ask_command(
    question: str = typer.Argument(..., help="Question to answer from the local RAG index."),
    config: Path = typer.Option(Path("config.example.yaml"), "--config", "-c"),
    embedding_model: str | None = typer.Option(None, "--embedding-model"),
    chat_model: str | None = typer.Option(None, "--chat-model", "-m"),
    collection_name: str | None = typer.Option(None, "--collection-name"),
    backend: str = typer.Option("auto", "--backend", help="auto, chroma, or simple."),
    top_k: int | None = typer.Option(None, "--top-k", "-k"),
    show_context: bool = typer.Option(False, "--show-context", help="Print retrieved context snippets after the answer."),
) -> None:
    """Answer a question using retrieved RAG chunks and a local Ollama chat model."""
    cfg = load_config(config)
    try:
        answer = ask_rag(
            question=question,
            chroma_dir=cfg.chroma_dir,
            collection_name=collection_name or cfg.rag.collection_name,
            embedding_model=embedding_model or cfg.rag.embedding_model,
            chat_model=chat_model or cfg.rag.chat_model,
            ollama_base_url=cfg.rag.ollama_base_url,
            top_k=top_k or cfg.rag.top_k,
            backend=backend,
        )
    except Exception as exc:
        console.print(f"RAG ask failed: {exc}", style="red", markup=False)
        raise typer.Exit(1) from exc

    console.print("\n[bold green]回答[/bold green]")
    console.print(answer.answer, markup=False)
    console.print("\n[bold cyan]来源[/bold cyan]")
    console.print(format_sources(answer.sources), markup=False)

    if show_context:
        console.print("\n[bold cyan]召回上下文[/bold cyan]")
        for context in answer.contexts:
            snippet = context.text.replace("\n", " ")
            if len(snippet) > 700:
                snippet = snippet[:700].rstrip() + "..."
            console.print(f"\n[{context.rank}] {context.metadata.get('source_path', '')}", markup=False)
            console.print(snippet, markup=False)


@app.command("webui")
def webui_command(
    config: Path = typer.Option(Path("config.example.yaml"), "--config", "-c"),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8765, "--port", "-p"),
    embedding_model: str | None = typer.Option(None, "--embedding-model"),
    chat_model: str | None = typer.Option(None, "--chat-model", "-m"),
    collection_name: str | None = typer.Option(None, "--collection-name"),
    backend: str = typer.Option("auto", "--backend", help="auto, chroma, or simple."),
    open_browser: bool = typer.Option(False, "--open", help="Open the local WebUI in the default browser."),
) -> None:
    """Start a local browser UI for RAG query and RAG ask."""
    cfg = load_config(config)
    settings = WebUISettings(
        config=cfg,
        embedding_model=embedding_model or cfg.rag.embedding_model,
        chat_model=chat_model or cfg.rag.chat_model,
        collection_name=collection_name or cfg.rag.collection_name,
        backend=backend,
    )
    console.print(f"[green]Starting RAG WebUI on http://{host}:{port}[/green]")
    run_webui(settings=settings, host=host, port=port, open_browser=open_browser)


@app.command("chat-saki")
def chat_saki_command(
    message: str | None = typer.Argument(None, help="Single-turn message. Omit to start interactive chat."),
    config: Path = typer.Option(Path("config.example.yaml"), "--config", "-c"),
    mode: str = typer.Option("auto", "--mode", help="auto, rag, or casual."),
    top_k: int = typer.Option(4, "--top-k", "-k"),
    embedding_model: str | None = typer.Option(None, "--embedding-model"),
    chat_model: str | None = typer.Option(None, "--chat-model", "-m"),
    collection_name: str | None = typer.Option(None, "--collection-name"),
    backend: str = typer.Option("auto", "--backend", help="auto, chroma, or simple."),
    show_sources: bool = typer.Option(False, "--show-sources", help="Show RAG sources when retrieval is used."),
) -> None:
    """Chat as Saki with optional RAG grounding."""
    if mode not in CHAT_MODES:
        raise typer.BadParameter("mode must be auto, rag, or casual")
    cfg = load_config(config)
    common = {
        "chroma_dir": cfg.chroma_dir,
        "collection_name": collection_name or cfg.rag.collection_name,
        "embedding_model": embedding_model or cfg.rag.embedding_model,
        "chat_model": chat_model or cfg.rag.chat_model,
        "ollama_base_url": cfg.rag.ollama_base_url,
        "mode": mode,
        "top_k": top_k,
        "backend": backend,
        "rag_docs_dir": cfg.rag_docs_dir,
    }

    if message is not None:
        try:
            response = chat_saki(user_message=message, history=[], **common)
        except Exception as exc:
            console.print(f"chat-saki failed: {exc}", style="red", markup=False)
            raise typer.Exit(1) from exc
        console.print(response.message, markup=False)
        if show_sources:
            console.print("\n[bold cyan]RAG[/bold cyan] " + ("used" if response.rag_used else "skipped"))
            console.print(format_saki_sources(response.sources), markup=False)
        return

    history: list[ChatTurn] = []
    console.print("[green]chat-saki started. Type /exit to quit, /rag /casual /auto to switch mode.[/green]")
    current_mode = mode
    while True:
        user_input = console.input("[bold]制作人> [/bold]").strip()
        if not user_input:
            continue
        if user_input in {"/exit", "/quit"}:
            break
        if user_input in {"/rag", "/casual", "/auto"}:
            current_mode = user_input[1:]
            common["mode"] = current_mode
            console.print(f"[cyan]mode={current_mode}[/cyan]")
            continue
        try:
            response = chat_saki(user_message=user_input, history=history, **common)
        except Exception as exc:
            console.print(f"chat-saki failed: {exc}", style="red", markup=False)
            continue
        console.print(f"[bold magenta]咲季>[/bold magenta] {response.message}", markup=False)
        if show_sources and response.rag_used:
            console.print(format_saki_sources(response.sources), markup=False)
        history.append(ChatTurn(user=user_input, assistant=response.message))
        history = history[-6:]


if __name__ == "__main__":
    app()
