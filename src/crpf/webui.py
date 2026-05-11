from __future__ import annotations

import json
import threading
import webbrowser
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from .chat_store import ChatHistoryStore, Conversation, StoredMessage
from .config import ProjectConfig
from .rag_chat import ask_rag, stream_ollama_chat
from .rag_index import RagSearchResult, query_rag_index
from .saki_chat import CHAT_MODES, ChatTurn, chat_saki, clean_saki_reply, prepare_saki_chat
from .source_trace import build_source_trace, normalize_scene_id
from .tts import synthesize_saki_tts


ASSETS_DIR = Path(__file__).with_name("assets")
SAKI_AVATAR_PATH = ASSETS_DIR / "saki-avatar.jpg"


@dataclass(frozen=True)
class WebUISettings:
    config: ProjectConfig
    embedding_model: str
    chat_model: str
    collection_name: str
    backend: str


def run_webui(
    settings: WebUISettings,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = False,
) -> None:
    handler = make_handler(settings)
    server = ThreadingHTTPServer((host, port), handler)
    url = f"http://{host}:{server.server_port}"
    if open_browser:
        threading.Timer(0.2, lambda: webbrowser.open(url)).start()
    print(f"RAG WebUI running at {url}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def make_handler(settings: WebUISettings) -> type[BaseHTTPRequestHandler]:
    chat_store = ChatHistoryStore(chat_history_db_path(settings.config))
    chat_store.ensure_schema()

    class RagWebUIHandler(BaseHTTPRequestHandler):
        server_version = "CRPFWebUI/0.1"

        def do_GET(self) -> None:  # noqa: N802 - http.server naming
            parsed_url = urlparse(self.path)
            path = parsed_url.path
            if path == "/":
                self.write_html(render_index_html(settings))
                return
            if path == "/assets/saki-avatar.jpg":
                self.write_file(SAKI_AVATAR_PATH, "image/jpeg")
                return
            if path.startswith("/tts-audio/"):
                filename = Path(unquote(path.removeprefix("/tts-audio/"))).name
                audio_path = settings.config.tts.output_dir / filename
                self.write_file(audio_path, audio_content_type(audio_path))
                return
            if path == "/health":
                self.write_json(
                    {
                        "ok": True,
                        "collection_name": settings.collection_name,
                        "embedding_model": settings.embedding_model,
                        "chat_model": settings.chat_model,
                        "backend": settings.backend,
                    }
                )
                return
            if path == "/api/trace":
                try:
                    query = parse_qs(parsed_url.query)
                    scene_id = normalize_scene_id(
                        first_query_value(query, "scene_id") or first_query_value(query, "source_path") or ""
                    )
                    self.write_json(
                        build_source_trace(
                            scene_id=scene_id,
                            rag_docs_dir=settings.config.rag_docs_dir,
                            output_dir=settings.config.summary_output_dir,
                        )
                    )
                except ValueError as exc:
                    self.write_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            if path == "/api/conversations":
                self.write_json(
                    {
                        "conversations": [
                            serialize_conversation(conversation)
                            for conversation in chat_store.list_conversations()
                        ]
                    }
                )
                return
            if path.startswith("/api/conversations/"):
                conversation_id = path.removeprefix("/api/conversations/").strip("/")
                conversation = require_conversation(chat_store, conversation_id)
                self.write_json(
                    {
                        "conversation": serialize_conversation(conversation),
                        "messages": [
                            serialize_message(message)
                            for message in chat_store.get_messages(conversation.id)
                        ],
                    }
                )
                return
            self.write_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802 - http.server naming
            path = urlparse(self.path).path
            try:
                payload = self.read_json()
                if path == "/api/conversations":
                    title = str(payload.get("title") or "新聊天")
                    conversation = chat_store.create_conversation(title)
                    self.write_json(
                        {"conversation": serialize_conversation(conversation)},
                        status=HTTPStatus.CREATED,
                    )
                    return
                if path == "/api/query":
                    self.handle_query(payload)
                    return
                if path == "/api/ask":
                    self.handle_ask(payload)
                    return
                if path == "/api/chat-saki":
                    self.handle_chat_saki(payload)
                    return
                if path == "/api/chat-saki-stream":
                    self.handle_chat_saki_stream(payload)
                    return
                if path == "/api/tts/saki":
                    self.handle_tts_saki(payload)
                    return
                self.write_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)
            except ValueError as exc:
                self.write_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            except Exception as exc:
                self.write_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

        def do_DELETE(self) -> None:  # noqa: N802 - http.server naming
            path = urlparse(self.path).path
            try:
                if path.startswith("/api/conversations/"):
                    conversation_id = path.removeprefix("/api/conversations/").strip("/")
                    deleted = chat_store.delete_conversation(conversation_id)
                    if not deleted:
                        raise ValueError("conversation not found")
                    self.write_json({"ok": True})
                    return
                self.write_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)
            except ValueError as exc:
                self.write_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

        def handle_query(self, payload: dict[str, Any]) -> None:
            question = require_question(payload)
            top_k = parse_top_k(payload.get("top_k"), settings.config.rag.top_k)
            backend = parse_backend(payload.get("backend"), settings.backend)
            results = query_rag_index(
                query=question,
                chroma_dir=settings.config.chroma_dir,
                collection_name=settings.collection_name,
                embedding_model=settings.embedding_model,
                ollama_base_url=settings.config.rag.ollama_base_url,
                top_k=top_k,
                backend=backend,
            )
            self.write_json({"results": [serialize_search_result(result) for result in results]})

        def handle_ask(self, payload: dict[str, Any]) -> None:
            question = require_question(payload)
            top_k = parse_top_k(payload.get("top_k"), settings.config.rag.top_k)
            backend = parse_backend(payload.get("backend"), settings.backend)
            conversation_id = parse_optional_text(payload.get("conversation_id"))
            answer = ask_rag(
                question=question,
                chroma_dir=settings.config.chroma_dir,
                collection_name=settings.collection_name,
                embedding_model=settings.embedding_model,
                chat_model=settings.chat_model,
                ollama_base_url=settings.config.rag.ollama_base_url,
                top_k=top_k,
                backend=backend,
            )
            sources = [
                {
                    "label": source.label,
                    "source_path": source.source_path,
                    "topic": source.topic,
                    "scene_ids": list(source.scene_ids),
                    "distance": source.distance,
                }
                for source in answer.sources
            ]
            contexts = [serialize_search_result(result) for result in answer.contexts]
            conversation = get_or_create_conversation(chat_store, conversation_id, question)
            chat_store.add_message(
                conversation.id,
                "user",
                question,
                {"kind": "rag-ask", "top_k": top_k, "backend": backend},
            )
            chat_store.add_message(
                conversation.id,
                "assistant",
                answer.answer,
                {
                    "kind": "rag-ask",
                    "sources": sources,
                    "contexts": contexts,
                },
            )
            self.write_json(
                {
                    "question": answer.question,
                    "answer": answer.answer,
                    "conversation_id": conversation.id,
                    "conversation": serialize_conversation(chat_store.get_conversation(conversation.id) or conversation),
                    "sources": sources,
                    "contexts": contexts,
                }
            )

        def handle_chat_saki(self, payload: dict[str, Any]) -> None:
            question = require_question(payload)
            top_k = parse_top_k(payload.get("top_k"), settings.config.rag.top_k)
            backend = parse_backend(payload.get("backend"), settings.backend)
            mode = parse_chat_mode(payload.get("mode"), "auto")
            conversation_id = parse_optional_text(payload.get("conversation_id"))
            if conversation_id:
                conversation = require_conversation(chat_store, conversation_id)
                history = stored_messages_to_chat_turns(chat_store.get_messages(conversation.id))
            else:
                conversation = None
                history = parse_chat_history(payload.get("history"))
            response = chat_saki(
                user_message=question,
                chroma_dir=settings.config.chroma_dir,
                collection_name=settings.collection_name,
                embedding_model=settings.embedding_model,
                chat_model=settings.chat_model,
                ollama_base_url=settings.config.rag.ollama_base_url,
                mode=mode,
                top_k=top_k,
                backend=backend,
                history=history,
                rag_docs_dir=settings.config.rag_docs_dir,
            )
            sources = [
                {
                    "label": source.label,
                    "source_path": source.source_path,
                    "topic": source.topic,
                    "scene_ids": list(source.scene_ids),
                    "distance": source.distance,
                }
                for source in response.sources
            ]
            contexts = [serialize_search_result(result) for result in response.contexts]
            if conversation is None:
                conversation = chat_store.create_conversation(question)
            chat_store.add_message(
                conversation.id,
                "user",
                question,
                {"kind": "chat-saki", "mode": mode, "top_k": top_k, "backend": backend},
            )
            chat_store.add_message(
                conversation.id,
                "assistant",
                response.message,
                {
                    "kind": "chat-saki",
                    "mode": response.mode,
                    "rag_used": response.rag_used,
                    "sources": sources,
                    "contexts": contexts,
                },
            )
            self.write_json(
                {
                    "question": question,
                    "message": response.message,
                    "mode": response.mode,
                    "rag_used": response.rag_used,
                    "conversation_id": conversation.id,
                    "conversation": serialize_conversation(chat_store.get_conversation(conversation.id) or conversation),
                    "sources": sources,
                    "contexts": contexts,
                }
            )

        def handle_chat_saki_stream(self, payload: dict[str, Any]) -> None:
            question = require_question(payload)
            top_k = parse_top_k(payload.get("top_k"), settings.config.rag.top_k)
            backend = parse_backend(payload.get("backend"), settings.backend)
            mode = parse_chat_mode(payload.get("mode"), "auto")
            conversation_id = parse_optional_text(payload.get("conversation_id"))
            if conversation_id:
                conversation = require_conversation(chat_store, conversation_id)
                history = stored_messages_to_chat_turns(chat_store.get_messages(conversation.id))
            else:
                conversation = None
                history = parse_chat_history(payload.get("history"))
            prepared = prepare_saki_chat(
                user_message=question,
                chroma_dir=settings.config.chroma_dir,
                collection_name=settings.collection_name,
                embedding_model=settings.embedding_model,
                ollama_base_url=settings.config.rag.ollama_base_url,
                mode=mode,
                top_k=top_k,
                backend=backend,
                history=history,
                rag_docs_dir=settings.config.rag_docs_dir,
            )
            sources = [
                {
                    "label": source.label,
                    "source_path": source.source_path,
                    "topic": source.topic,
                    "scene_ids": list(source.scene_ids),
                    "distance": source.distance,
                }
                for source in prepared.sources
            ]
            contexts = [serialize_search_result(result) for result in prepared.contexts]
            if conversation is None:
                conversation = chat_store.create_conversation(question)
            chat_store.add_message(
                conversation.id,
                "user",
                question,
                {"kind": "chat-saki", "mode": mode, "top_k": top_k, "backend": backend},
            )

            self.write_ndjson_headers()
            self.write_ndjson(
                {
                    "type": "start",
                    "conversation_id": conversation.id,
                    "mode": prepared.mode,
                    "rag_used": prepared.rag_used,
                }
            )
            parts: list[str] = []
            try:
                for delta in stream_ollama_chat(
                    model=settings.chat_model,
                    ollama_base_url=settings.config.rag.ollama_base_url,
                    prompt=prepared.prompt,
                    temperature=0.65,
                    num_ctx=8192,
                    num_predict=700,
                ):
                    parts.append(delta)
                    self.write_ndjson({"type": "delta", "text": delta})
                message = clean_saki_reply("".join(parts))
                chat_store.add_message(
                    conversation.id,
                    "assistant",
                    message,
                    {
                        "kind": "chat-saki",
                        "mode": prepared.mode,
                        "rag_used": prepared.rag_used,
                        "sources": sources,
                        "contexts": contexts,
                    },
                )
                self.write_ndjson(
                    {
                        "type": "done",
                        "message": message,
                        "mode": prepared.mode,
                        "rag_used": prepared.rag_used,
                        "conversation_id": conversation.id,
                        "conversation": serialize_conversation(
                            chat_store.get_conversation(conversation.id) or conversation
                        ),
                        "sources": sources,
                        "contexts": contexts,
                    }
                )
            except Exception as exc:
                self.write_ndjson({"type": "error", "error": str(exc)})

        def handle_tts_saki(self, payload: dict[str, Any]) -> None:
            text = str(payload.get("text", "")).strip()
            if not text:
                raise ValueError("text is required")
            result = synthesize_saki_tts(settings.config, text)
            self.write_json(
                {
                    "audio_url": f"/tts-audio/{result.path.name}",
                    "cache_path": str(result.path),
                    "cached": result.cached,
                    "voice_text": result.voice_text,
                    "translated": result.translated,
                    "metadata_path": str(result.metadata_path),
                }
            )

        def read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length).decode("utf-8") if length else "{}"
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise ValueError("JSON body must be an object")
            return data

        def write_html(self, html: str) -> None:
            body = html.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def write_json(self, data: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def write_file(self, path: Path, content_type: str) -> None:
            if not path.exists():
                self.write_json({"error": "asset not found"}, status=HTTPStatus.NOT_FOUND)
                return
            body = path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "public, max-age=3600")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def write_ndjson_headers(self) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()

        def write_ndjson(self, data: dict[str, Any]) -> None:
            body = (json.dumps(data, ensure_ascii=False) + "\n").encode("utf-8")
            self.wfile.write(body)
            self.wfile.flush()

        def log_message(self, format: str, *args: Any) -> None:
            return

    return RagWebUIHandler


def require_question(payload: dict[str, Any]) -> str:
    question = str(payload.get("question", "")).strip()
    if not question:
        raise ValueError("question is required")
    return question


def parse_top_k(value: Any, default: int) -> int:
    if value in {None, ""}:
        return default
    top_k = int(value)
    if top_k < 1 or top_k > 12:
        raise ValueError("top_k must be between 1 and 12")
    return top_k


def parse_backend(value: Any, default: str) -> str:
    backend = str(value or default).strip()
    if backend not in {"auto", "chroma", "simple"}:
        raise ValueError("backend must be auto, chroma, or simple")
    return backend


def audio_content_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".wav":
        return "audio/wav"
    if suffix == ".mp3":
        return "audio/mpeg"
    if suffix in {".ogg", ".oga"}:
        return "audio/ogg"
    return "application/octet-stream"


def parse_chat_mode(value: Any, default: str) -> str:
    mode = str(value or default).strip()
    if mode not in CHAT_MODES:
        raise ValueError("mode must be auto, rag, or casual")
    return mode


def parse_chat_history(value: Any) -> list[ChatTurn]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("history must be a list")
    turns: list[ChatTurn] = []
    for item in value[-6:]:
        if not isinstance(item, dict):
            continue
        user = str(item.get("user", "")).strip()
        assistant = str(item.get("assistant", "")).strip()
        if user and assistant:
            turns.append(ChatTurn(user=user, assistant=assistant))
    return turns


def parse_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def first_query_value(query: dict[str, list[str]], key: str) -> str:
    values = query.get(key) or []
    return values[0] if values else ""


def chat_history_db_path(config: ProjectConfig) -> Path:
    return config.chroma_dir.parent / "chat_history.sqlite3"


def require_conversation(store: ChatHistoryStore, conversation_id: str | None) -> Conversation:
    if not conversation_id:
        raise ValueError("conversation_id is required")
    conversation = store.get_conversation(conversation_id)
    if conversation is None:
        raise ValueError("conversation not found")
    return conversation


def get_or_create_conversation(
    store: ChatHistoryStore,
    conversation_id: str | None,
    title_seed: str,
) -> Conversation:
    if conversation_id:
        return require_conversation(store, conversation_id)
    return store.create_conversation(title_seed)


def stored_messages_to_chat_turns(messages: list[StoredMessage]) -> list[ChatTurn]:
    turns: list[ChatTurn] = []
    pending_user: str | None = None
    for message in messages:
        if message.role == "user":
            pending_user = message.content
        elif message.role == "assistant" and pending_user:
            turns.append(ChatTurn(user=pending_user, assistant=message.content))
            pending_user = None
    return turns[-6:]


def serialize_conversation(conversation: Conversation) -> dict[str, Any]:
    return {
        "id": conversation.id,
        "title": conversation.title,
        "created_at": conversation.created_at,
        "updated_at": conversation.updated_at,
        "message_count": conversation.message_count,
    }


def serialize_message(message: StoredMessage) -> dict[str, Any]:
    return {
        "id": message.id,
        "conversation_id": message.conversation_id,
        "role": message.role,
        "content": message.content,
        "created_at": message.created_at,
        "metadata": message.metadata,
    }


def serialize_search_result(result: RagSearchResult) -> dict[str, Any]:
    return {
        "rank": result.rank,
        "distance": result.distance,
        "text": result.text,
        "metadata": result.metadata,
    }


def render_index_html(settings: WebUISettings) -> str:
    return render_chatgpt_style_html(settings)

    examples = [
        "咲季为什么害怕输给佑芽？",
        "咲季和制作人的关系是什么？",
        "咲季的说话风格有什么特点？",
        "Re;IRIS 对咲季的成长有什么影响？",
    ]
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CRPF RAG</title>
  <style>
    :root {{
      --bg: #f6f3ee;
      --panel: #fffdf9;
      --ink: #26231f;
      --muted: #6f6860;
      --line: #ded7cd;
      --accent: #0f766e;
      --accent-strong: #115e59;
      --rose: #be4b49;
      --gold: #a16207;
      --code: #f0ece4;
      --shadow: 0 16px 45px rgba(48, 42, 34, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.55;
    }}
    button, textarea, input, select {{ font: inherit; }}
    .app {{
      min-height: 100vh;
      display: grid;
      grid-template-columns: minmax(320px, 420px) minmax(0, 1fr);
    }}
    aside {{
      border-right: 1px solid var(--line);
      background: #fcfaf6;
      padding: 24px;
      display: flex;
      flex-direction: column;
      gap: 18px;
    }}
    main {{
      padding: 24px;
      display: grid;
      gap: 18px;
      align-content: start;
    }}
    h1 {{
      margin: 0;
      font-size: 24px;
      line-height: 1.2;
      letter-spacing: 0;
    }}
    h2 {{
      margin: 0 0 10px;
      font-size: 15px;
      letter-spacing: 0;
    }}
    .meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      color: var(--muted);
      font-size: 13px;
    }}
    .pill {{
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 999px;
      padding: 4px 9px;
      white-space: nowrap;
    }}
    label {{
      display: grid;
      gap: 7px;
      font-weight: 650;
      font-size: 13px;
    }}
    textarea {{
      min-height: 152px;
      resize: vertical;
      border: 1px solid var(--line);
      background: var(--panel);
      color: var(--ink);
      border-radius: 8px;
      padding: 13px 14px;
      outline: none;
      box-shadow: var(--shadow);
    }}
    textarea:focus, input:focus, select:focus {{
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(15, 118, 110, 0.16);
    }}
    .row {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }}
    input, select {{
      width: 100%;
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 8px;
      padding: 10px 12px;
      color: var(--ink);
      outline: none;
    }}
    .actions {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }}
    button {{
      border: 0;
      border-radius: 8px;
      padding: 11px 14px;
      color: white;
      background: var(--accent);
      cursor: pointer;
      font-weight: 700;
      min-height: 44px;
    }}
    button.secondary {{ background: #544b42; }}
    button.tertiary {{ background: #7f4f24; }}
    button:disabled {{ opacity: 0.55; cursor: wait; }}
    .examples {{
      display: grid;
      gap: 8px;
    }}
    .example {{
      text-align: left;
      color: var(--ink);
      background: var(--panel);
      border: 1px solid var(--line);
      font-weight: 600;
      min-height: auto;
      padding: 9px 11px;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      padding: 18px;
      overflow: hidden;
    }}
    .answer {{
      white-space: pre-wrap;
      font-size: 15px;
    }}
    .chat-log {{
      display: grid;
      gap: 12px;
    }}
    .bubble {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 11px 12px;
      background: #f8f6f0;
      white-space: pre-wrap;
    }}
    .bubble.user {{
      background: #eef7f5;
      border-color: rgba(15, 118, 110, 0.25);
    }}
    .bubble-name {{
      color: var(--muted);
      font-size: 12px;
      font-weight: 800;
      margin-bottom: 4px;
    }}
    .status {{
      color: var(--muted);
      font-size: 13px;
    }}
    .status.error {{ color: var(--rose); font-weight: 700; }}
    .sources, .contexts {{
      display: grid;
      gap: 10px;
    }}
    .source, .context {{
      border-top: 1px solid var(--line);
      padding-top: 10px;
    }}
    .source:first-child, .context:first-child {{
      border-top: 0;
      padding-top: 0;
    }}
    .source-head {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
      font-size: 13px;
      font-weight: 700;
    }}
    .badge {{
      background: var(--code);
      color: var(--accent-strong);
      border-radius: 999px;
      padding: 2px 8px;
    }}
    .distance {{ color: var(--gold); }}
    .snippet {{
      color: var(--muted);
      font-size: 13px;
      margin-top: 6px;
      white-space: pre-wrap;
    }}
    @media (max-width: 860px) {{
      .app {{ grid-template-columns: 1fr; }}
      aside {{ border-right: 0; border-bottom: 1px solid var(--line); }}
      .row, .actions {{ grid-template-columns: 1fr; }}
      main, aside {{ padding: 18px; }}
    }}
  </style>
</head>
<body>
  <div class="app">
    <aside>
      <div>
        <h1>CRPF RAG</h1>
        <div class="meta">
          <span class="pill">{settings.collection_name}</span>
          <span class="pill">{settings.embedding_model}</span>
          <span class="pill">{settings.chat_model}</span>
        </div>
      </div>
      <label>
        问题
        <textarea id="question">咲季为什么害怕输给佑芽？</textarea>
      </label>
      <div class="row">
        <label>
          Top K
          <input id="topK" type="number" min="1" max="12" value="4">
        </label>
        <label>
          后端
          <input id="backend" value="{settings.backend}">
        </label>
      </div>
      <label>
        咲季聊天模式
        <select id="chatMode">
          <option value="auto" selected>auto</option>
          <option value="rag">rag</option>
          <option value="casual">casual</option>
        </select>
      </label>
      <div class="actions">
        <button id="askBtn">问答</button>
        <button id="chatBtn" class="tertiary">咲季聊天</button>
        <button id="queryBtn" class="secondary">检索</button>
        <button id="clearBtn" class="secondary" type="button">清空聊天</button>
      </div>
      <section>
        <h2>样例</h2>
        <div class="examples">
          {"".join(f'<button class="example" data-question="{escape_html(item)}">{escape_html(item)}</button>' for item in examples)}
        </div>
      </section>
    </aside>
    <main>
      <section class="panel">
        <h2>回答 / 聊天</h2>
        <div id="status" class="status">Ready</div>
        <div id="answer" class="answer"></div>
      </section>
      <section class="panel">
        <h2>来源</h2>
        <div id="sources" class="sources"></div>
      </section>
      <section class="panel">
        <h2>召回上下文</h2>
        <div id="contexts" class="contexts"></div>
      </section>
    </main>
  </div>
  <script>
    const question = document.getElementById('question');
    const topK = document.getElementById('topK');
    const backend = document.getElementById('backend');
    const chatMode = document.getElementById('chatMode');
    const statusEl = document.getElementById('status');
    const answerEl = document.getElementById('answer');
    const sourcesEl = document.getElementById('sources');
    const contextsEl = document.getElementById('contexts');
    const askBtn = document.getElementById('askBtn');
    const chatBtn = document.getElementById('chatBtn');
    const queryBtn = document.getElementById('queryBtn');
    const clearBtn = document.getElementById('clearBtn');
    let chatHistory = [];

    document.querySelectorAll('.example').forEach((button) => {{
      button.addEventListener('click', () => {{
        question.value = button.dataset.question;
      }});
    }});

    askBtn.addEventListener('click', () => runAsk());
    chatBtn.addEventListener('click', () => runChatSaki());
    queryBtn.addEventListener('click', () => runQuery());
    clearBtn.addEventListener('click', () => {{
      chatHistory = [];
      answerEl.textContent = '';
      sourcesEl.innerHTML = '';
      contextsEl.innerHTML = '';
      setBusy(false, 'Chat cleared');
    }});

    function payload() {{
      return {{
        question: question.value,
        top_k: Number(topK.value || 4),
        backend: backend.value || 'auto'
      }};
    }}

    async function postJSON(path, body) {{
      const response = await fetch(path, {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify(body)
      }});
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || response.statusText);
      return data;
    }}

    async function runAsk() {{
      setBusy(true, 'Thinking');
      answerEl.textContent = '';
      sourcesEl.innerHTML = '';
      contextsEl.innerHTML = '';
      try {{
        const data = await postJSON('/api/ask', payload());
        answerEl.className = 'answer';
        answerEl.textContent = data.answer;
        renderSources(data.sources || []);
        renderContexts(data.contexts || []);
        setBusy(false, 'Done');
      }} catch (error) {{
        setError(error.message);
      }}
    }}

    async function runQuery() {{
      setBusy(true, 'Searching');
      answerEl.textContent = '';
      sourcesEl.innerHTML = '';
      contextsEl.innerHTML = '';
      try {{
        const data = await postJSON('/api/query', payload());
        renderContexts(data.results || []);
        setBusy(false, 'Done');
      }} catch (error) {{
        setError(error.message);
      }}
    }}

    async function runChatSaki() {{
      const current = question.value.trim();
      if (!current) {{
        setError('请输入问题或消息');
        return;
      }}
      setBusy(true, 'Saki is replying');
      sourcesEl.innerHTML = '';
      contextsEl.innerHTML = '';
      try {{
        const data = await postJSON('/api/chat-saki', {{
          ...payload(),
          mode: chatMode.value || 'auto',
          history: chatHistory
        }});
        chatHistory.push({{ user: current, assistant: data.message }});
        chatHistory = chatHistory.slice(-6);
        renderChatLog();
        renderSources(data.sources || []);
        renderContexts(data.contexts || []);
        setBusy(false, data.rag_used ? 'Done · RAG used' : 'Done · RAG skipped');
      }} catch (error) {{
        setError(error.message);
      }}
    }}

    function setBusy(disabled, text) {{
      askBtn.disabled = disabled;
      chatBtn.disabled = disabled;
      queryBtn.disabled = disabled;
      clearBtn.disabled = disabled;
      statusEl.className = 'status';
      statusEl.textContent = text;
    }}

    function setError(message) {{
      askBtn.disabled = false;
      chatBtn.disabled = false;
      queryBtn.disabled = false;
      clearBtn.disabled = false;
      statusEl.className = 'status error';
      statusEl.textContent = message;
    }}

    function renderChatLog() {{
      answerEl.className = 'answer chat-log';
      answerEl.innerHTML = chatHistory.map((turn) => `
        <div class="bubble user">
          <div class="bubble-name">制作人</div>
          <div>${{escapeText(turn.user)}}</div>
        </div>
        <div class="bubble">
          <div class="bubble-name">咲季</div>
          <div>${{escapeText(turn.assistant)}}</div>
        </div>
      `).join('');
    }}

    function renderSources(sources) {{
      sourcesEl.innerHTML = sources.map((source) => `
        <div class="source">
          <div class="source-head">
            <span class="badge">${{escapeText(source.label)}}</span>
            <span>${{escapeText(source.source_path)}}</span>
            <span>${{escapeText(source.topic)}}</span>
            <span class="distance">${{formatDistance(source.distance)}}</span>
          </div>
          <div class="snippet">${{escapeText((source.scene_ids || []).join(', ') || '无 scene_id')}}</div>
        </div>
      `).join('');
    }}

    function renderContexts(contexts) {{
      contextsEl.innerHTML = contexts.map((item) => {{
        const meta = item.metadata || {{}};
        return `
          <div class="context">
            <div class="source-head">
              <span class="badge">#${{item.rank}}</span>
              <span>${{escapeText(meta.source_path || '')}}</span>
              <span>${{escapeText(meta.topic || '')}}</span>
              <span class="distance">${{formatDistance(item.distance)}}</span>
            </div>
            <div class="snippet">${{escapeText(compact(item.text || ''))}}</div>
          </div>
        `;
      }}).join('');
    }}

    function compact(text) {{
      const clean = text.replace(/\\s+/g, ' ').trim();
      return clean.length > 900 ? clean.slice(0, 900).trim() + '...' : clean;
    }}

    function formatDistance(value) {{
      return typeof value === 'number' ? value.toFixed(4) : 'n/a';
    }}

    function escapeText(value) {{
      return String(value).replace(/[&<>"']/g, (char) => ({{
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
      }}[char]));
    }}
  </script>
</body>
</html>"""


def escape_html(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def render_chatgpt_style_html(settings: WebUISettings) -> str:
    examples = [
        "今天有点累，鼓励我一下",
        "讲讲名古屋公演",
        "你和佑芽是什么关系？",
        "咲季的说话风格有什么特点？",
        "Re;IRIS 对咲季的成长有什么影响？",
    ]
    suggestions_html = "\n".join(
        f'<button class="suggestion" type="button" data-question="{escape_html(item)}">{escape_html(item)}</button>'
        for item in examples[:4]
    )
    backend_options = "\n".join(
        f'<option value="{value}"{" selected" if value == settings.backend else ""}>{label}</option>'
        for value, label in [("auto", "Auto"), ("chroma", "Chroma"), ("simple", "Simple")]
    )
    chat_mode_options = "\n".join(
        f'<option value="{mode}">{"Auto" if mode == "auto" else "强制 RAG" if mode == "rag" else "日常聊天"}</option>'
        for mode in ("auto", "rag", "casual")
        if mode in CHAT_MODES
    )

    html = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CRPF Saki Chat</title>
  <style>
    :root {
      --bg: #fbf8f8;
      --sidebar: #fff0f4;
      --panel: #ffffff;
      --panel-soft: #fff7f9;
      --ink: #262025;
      --muted: #75656d;
      --line: #ead8df;
      --accent: #c84f72;
      --accent-strong: #a83d5d;
      --accent-soft: #fde4ec;
      --blue: #5877b8;
      --shadow: 0 18px 50px rgba(122, 63, 82, 0.12);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      height: 100vh;
      min-height: 100vh;
      overflow: hidden;
      background:
        radial-gradient(circle at 82% 10%, rgba(255, 214, 226, 0.75), transparent 32%),
        linear-gradient(180deg, #fffafb 0%, var(--bg) 48%, #f9f1f4 100%);
      color: var(--ink);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans SC", sans-serif;
      line-height: 1.55;
    }
    button, textarea, input, select { font: inherit; }
    button { border: 0; cursor: pointer; }
    .app {
      height: 100vh;
      min-height: 100vh;
      display: grid;
      grid-template-columns: 292px minmax(0, 1fr);
      overflow: hidden;
    }
    .sidebar {
      height: 100vh;
      min-height: 100vh;
      min-width: 0;
      overflow: hidden;
      background: color-mix(in srgb, var(--sidebar) 90%, white);
      border-right: 1px solid var(--line);
      padding: 18px 14px;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }
    .brand {
      display: grid;
      grid-template-columns: 48px 1fr;
      gap: 12px;
      align-items: center;
      padding: 8px 8px 12px;
    }
    .avatar, .bubble-avatar, .hero-avatar {
      object-fit: cover;
      background: #fff;
      border: 2px solid rgba(255, 255, 255, 0.85);
      box-shadow: 0 8px 20px rgba(151, 72, 100, 0.18);
    }
    .avatar { width: 48px; height: 48px; border-radius: 16px; }
    .brand-title { font-weight: 760; font-size: 17px; }
    .brand-subtitle { color: var(--muted); font-size: 12px; margin-top: 2px; }
    .new-chat {
      width: 100%;
      display: flex;
      align-items: center;
      gap: 9px;
      padding: 12px 13px;
      border-radius: 10px;
      color: #fff;
      background: linear-gradient(135deg, var(--accent), #e3829d);
      font-weight: 650;
      box-shadow: 0 12px 30px rgba(200, 79, 114, 0.22);
    }
    .nav-section {
      min-height: 0;
      display: grid;
      grid-template-rows: auto minmax(0, 1fr);
      gap: 8px;
      overflow: hidden;
      flex: 1 1 auto;
    }
    .nav-title {
      padding: 0 8px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0;
    }
    .conversation-list {
      display: grid;
      align-content: start;
      gap: 6px;
      overflow: auto;
      padding-right: 2px;
    }
    .conversation-row {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 30px;
      gap: 4px;
      align-items: center;
      border-radius: 9px;
    }
    .conversation-row.active {
      background: rgba(255, 255, 255, 0.86);
      box-shadow: inset 0 0 0 1px rgba(234, 216, 223, 0.75);
    }
    .history-item {
      width: 100%;
      min-height: 40px;
      padding: 10px 12px;
      border-radius: 8px;
      text-align: left;
      color: #44363e;
      background: transparent;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .history-item:hover { background: rgba(255, 255, 255, 0.72); }
    .conversation-delete {
      width: 30px;
      height: 30px;
      border-radius: 8px;
      background: transparent;
      color: var(--muted);
      font-size: 16px;
    }
    .conversation-delete:hover {
      color: var(--accent-strong);
      background: var(--accent-soft);
    }
    .settings {
      margin-top: auto;
      display: grid;
      gap: 10px;
      padding: 12px;
      border: 1px solid rgba(234, 216, 223, 0.9);
      background: rgba(255, 255, 255, 0.58);
      border-radius: 12px;
    }
    .setting-row {
      display: grid;
      grid-template-columns: 70px 1fr;
      gap: 8px;
      align-items: center;
      color: var(--muted);
      font-size: 13px;
    }
    input, select, textarea {
      border: 1px solid var(--line);
      background: #fff;
      color: var(--ink);
      border-radius: 9px;
      outline: none;
    }
    input:focus, select:focus, textarea:focus {
      border-color: rgba(200, 79, 114, 0.65);
      box-shadow: 0 0 0 3px rgba(200, 79, 114, 0.12);
    }
    input, select { width: 100%; min-height: 34px; padding: 0 9px; }
    .model-chip {
      padding: 9px 10px;
      border-radius: 10px;
      background: #fff;
      color: var(--muted);
      font-size: 12px;
      border: 1px solid var(--line);
    }
    .main {
      height: 100vh;
      min-width: 0;
      min-height: 0;
      display: grid;
      grid-template-rows: auto minmax(0, 1fr) auto;
      overflow: hidden;
    }
    .topbar {
      height: 64px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 28px;
      border-bottom: 1px solid rgba(234, 216, 223, 0.75);
      background: rgba(255, 255, 255, 0.58);
      backdrop-filter: blur(12px);
    }
    .top-title { display: flex; align-items: center; gap: 10px; font-weight: 740; }
    .top-dot {
      width: 10px;
      height: 10px;
      border-radius: 999px;
      background: #45b36b;
      box-shadow: 0 0 0 4px rgba(69, 179, 107, 0.14);
    }
    .status { color: var(--muted); font-size: 13px; }
    .chat-scroll {
      min-height: 0;
      overflow: auto;
      padding: 32px 20px 24px;
      overscroll-behavior: contain;
    }
    .chat-inner {
      width: min(860px, 100%);
      margin: 0 auto;
      display: grid;
      gap: 18px;
    }
    .welcome {
      min-height: min(55vh, 520px);
      display: grid;
      align-content: center;
      justify-items: center;
      text-align: center;
      gap: 18px;
      padding: 28px 0 10px;
    }
    .hero-avatar {
      width: 112px;
      height: 112px;
      border-radius: 28px;
    }
    h1 {
      margin: 0;
      font-size: clamp(28px, 4vw, 42px);
      line-height: 1.15;
      letter-spacing: 0;
    }
    .welcome p {
      margin: 0;
      color: var(--muted);
      max-width: 560px;
    }
    .suggestions {
      width: min(680px, 100%);
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      margin-top: 4px;
    }
    .suggestion {
      min-height: 48px;
      padding: 12px 14px;
      border: 1px solid var(--line);
      border-radius: 12px;
      background: rgba(255, 255, 255, 0.78);
      color: #453840;
      text-align: left;
      box-shadow: 0 10px 26px rgba(122, 63, 82, 0.06);
    }
    .suggestion:hover { border-color: rgba(200, 79, 114, 0.42); background: #fff; }
    .messages {
      display: grid;
      gap: 20px;
    }
    .message {
      display: grid;
      grid-template-columns: 38px minmax(0, 1fr);
      gap: 12px;
      align-items: start;
    }
    .message.user { grid-template-columns: minmax(0, 1fr) 38px; }
    .message.user .bubble { order: 1; justify-self: end; background: #fff; border-color: #e6dbe1; }
    .message.user .bubble-avatar { order: 2; background: linear-gradient(135deg, #6f8fd7, #91b4ee); }
    .bubble-avatar {
      width: 38px;
      height: 38px;
      border-radius: 12px;
      display: grid;
      place-items: center;
      color: #fff;
      font-weight: 750;
      overflow: hidden;
    }
    .bubble {
      max-width: 100%;
      border: 1px solid rgba(234, 216, 223, 0.92);
      background: rgba(255, 255, 255, 0.82);
      border-radius: 14px;
      padding: 13px 15px;
      box-shadow: 0 10px 30px rgba(122, 63, 82, 0.06);
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }
    .bubble-meta {
      margin-top: 10px;
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      color: var(--muted);
      font-size: 12px;
    }
    .tag {
      padding: 3px 8px;
      border-radius: 999px;
      background: var(--accent-soft);
      color: var(--accent-strong);
    }
    .bubble-actions {
      margin-top: 10px;
      display: flex;
      gap: 8px;
      align-items: center;
    }
    .voice-btn {
      min-height: 30px;
      padding: 0 10px;
      border-radius: 9px;
      background: #edf3ff;
      color: var(--blue);
      border: 1px solid #d8e5ff;
      font-size: 12px;
      font-weight: 720;
    }
    .voice-btn:disabled {
      opacity: 0.6;
      cursor: wait;
    }
    .composer-wrap {
      align-self: end;
      padding: 12px 20px 18px;
      background: linear-gradient(180deg, rgba(251, 248, 248, 0), rgba(251, 248, 248, 0.96) 32%);
    }
    .composer {
      width: min(860px, 100%);
      margin: 0 auto;
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 10px;
      align-items: end;
      padding: 10px;
      border: 1px solid var(--line);
      border-radius: 18px;
      background: rgba(255, 255, 255, 0.92);
      box-shadow: var(--shadow);
    }
    textarea {
      width: 100%;
      min-height: 56px;
      max-height: 180px;
      resize: vertical;
      border: 0;
      box-shadow: none;
      padding: 12px 12px;
      background: transparent;
    }
    textarea:focus { box-shadow: none; }
    .composer-actions {
      display: flex;
      align-items: center;
      gap: 8px;
      padding-bottom: 3px;
    }
    .mode-btn, .send-btn {
      min-height: 38px;
      border-radius: 11px;
      padding: 0 12px;
      background: var(--panel-soft);
      color: var(--accent-strong);
      font-weight: 650;
      border: 1px solid var(--line);
    }
    .mode-btn:hover { background: #fff; }
    .send-btn {
      min-width: 42px;
      padding: 0 14px;
      color: #fff;
      background: var(--accent);
      border-color: var(--accent);
    }
    .send-btn:disabled, .mode-btn:disabled { opacity: 0.55; cursor: wait; }
    .details {
      width: min(860px, 100%);
      margin: 12px auto 0;
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }
    details {
      border: 1px solid var(--line);
      border-radius: 12px;
      background: rgba(255, 255, 255, 0.8);
      overflow: hidden;
      max-height: 34vh;
    }
    summary {
      padding: 11px 13px;
      font-weight: 700;
      cursor: pointer;
      position: relative;
      list-style-position: inside;
    }
    details.has-new summary::after {
      content: "";
      position: absolute;
      top: 12px;
      right: 13px;
      width: 9px;
      height: 9px;
      border-radius: 999px;
      background: #3b82f6;
      box-shadow: 0 0 0 4px rgba(59, 130, 246, 0.16);
    }
    .detail-body {
      max-height: min(28vh, 320px);
      overflow: auto;
      padding: 0 13px 13px;
      display: grid;
      gap: 10px;
      overscroll-behavior: contain;
    }
    .source, .context {
      padding: 10px;
      border-radius: 10px;
      background: #fff;
      border: 1px solid #f0e2e7;
    }
    .source-head {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
      color: var(--muted);
      font-size: 12px;
    }
    .scene-list {
      overflow-wrap: anywhere;
      line-height: 1.5;
    }
    .trace-actions {
      width: 100%;
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-top: 2px;
    }
    .badge {
      padding: 2px 7px;
      border-radius: 999px;
      background: #edf3ff;
      color: var(--blue);
      font-weight: 760;
    }
    .snippet { margin-top: 8px; color: #473940; font-size: 13px; overflow-wrap: anywhere; }
    .trace-btn {
      min-height: 26px;
      padding: 0 9px;
      border-radius: 8px;
      background: #edf3ff;
      color: var(--blue);
      font-weight: 700;
      border: 1px solid #d8e5ff;
    }
    .trace-panel {
      width: min(860px, 100%);
      margin: 10px auto 0;
    }
    .trace-panel details {
      max-height: 40vh;
    }
    .trace-grid {
      display: grid;
      gap: 10px;
    }
    .trace-meta {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      color: var(--muted);
      font-size: 12px;
    }
    .trace-card {
      padding: 10px;
      border-radius: 10px;
      background: #fff;
      border: 1px solid #f0e2e7;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      font-size: 13px;
    }
    .dialogue-card {
      white-space: normal;
    }
    .dialogue-list {
      display: grid;
      gap: 6px;
      margin-top: 8px;
    }
    .dialogue-line {
      display: grid;
      gap: 2px;
      padding: 6px 8px;
      border-radius: 8px;
      background: #fff;
      border: 1px solid #f0e2e7;
      font-size: 13px;
      line-height: 1.45;
    }
    .dialogue-line strong {
      color: var(--accent-strong);
    }
    .dialogue-ja {
      color: var(--muted);
    }
    .empty { color: var(--muted); font-size: 13px; padding: 2px 0; }
    .footer-note {
      width: min(860px, 100%);
      margin: 8px auto 0;
      color: var(--muted);
      font-size: 12px;
      text-align: center;
    }
    @media (max-width: 900px) {
      body { overflow: auto; }
      .app { height: auto; min-height: 100vh; grid-template-columns: 1fr; overflow: visible; }
      .sidebar { height: auto; min-height: auto; max-height: 42vh; border-right: 0; border-bottom: 1px solid var(--line); }
      .settings { margin-top: 0; }
      .main { height: 100vh; min-height: 520px; }
      .details { grid-template-columns: 1fr; }
    }
    @media (max-width: 640px) {
      .topbar { padding: 0 16px; }
      .chat-scroll { padding: 24px 14px 18px; }
      .suggestions { grid-template-columns: 1fr; }
      .composer { grid-template-columns: 1fr; }
      .composer-actions { justify-content: flex-end; flex-wrap: wrap; }
    }
  </style>
</head>
<body>
  <div class="app">
    <aside class="sidebar">
      <div class="brand">
        <img class="avatar" src="/assets/saki-avatar.jpg" alt="花海咲季">
        <div>
          <div class="brand-title">CRPF Saki</div>
          <div class="brand-subtitle">RAG + Chat-Saki 本地助手</div>
        </div>
      </div>
      <button class="new-chat" type="button" id="newChat">＋ 新聊天</button>
      <div class="nav-section">
        <div class="nav-title">历史聊天</div>
        <div class="conversation-list" id="conversationList">
          <div class="empty">还没有历史聊天。</div>
        </div>
      </div>
      <div class="settings">
        <div class="setting-row">
          <label for="topK">Top K</label>
          <input id="topK" type="number" min="1" max="20" value="__TOP_K__">
        </div>
        <div class="setting-row">
          <label for="backend">检索库</label>
          <select id="backend">__BACKEND_OPTIONS__</select>
        </div>
        <div class="setting-row">
          <label for="chatMode">模式</label>
          <select id="chatMode">__CHAT_MODE_OPTIONS__</select>
        </div>
        <div class="model-chip">
          Collection: __COLLECTION__<br>
          Embedding: __EMBED_MODEL__<br>
          Chat: __CHAT_MODEL__
        </div>
      </div>
    </aside>
    <main class="main">
      <header class="topbar">
        <div class="top-title"><span class="top-dot"></span><span>花海咲季 Chat</span></div>
        <div class="status" id="status">待机中</div>
      </header>
      <section class="chat-scroll" id="chatScroll">
        <div class="chat-inner">
          <div class="welcome" id="welcome">
            <img class="hero-avatar" src="/assets/saki-avatar.jpg" alt="花海咲季">
            <h1>今天想和咲季聊什么？</h1>
            <p>可以直接日常聊天，也可以问剧情、人物关系、章节事件。需要查资料时会自动检索你的咲季 RAG 库。</p>
            <div class="suggestions">__SUGGESTIONS__</div>
          </div>
          <div class="messages" id="messages"></div>
        </div>
      </section>
      <section class="composer-wrap">
        <div class="composer">
          <textarea id="question" placeholder="有问题，尽管问。比如：讲讲名古屋公演"></textarea>
          <div class="composer-actions">
            <button class="mode-btn" type="button" id="chatBtn">咲季聊天</button>
            <button class="mode-btn" type="button" id="askBtn">知识问答</button>
            <button class="mode-btn" type="button" id="queryBtn">只检索</button>
            <button class="send-btn" type="button" id="sendBtn">↑</button>
          </div>
        </div>
        <div class="details">
          <details id="sourcesDetails">
            <summary>引用来源</summary>
            <div class="detail-body" id="sources"><div class="empty">还没有引用来源。</div></div>
          </details>
          <details id="contextsDetails">
            <summary>检索片段</summary>
            <div class="detail-body" id="contexts"><div class="empty">还没有检索片段。</div></div>
          </details>
        </div>
        <div class="trace-panel">
          <details id="traceDetails">
            <summary>原文追溯</summary>
            <div class="detail-body" id="traceBody"><div class="empty">点击来源里的“追溯”查看场景卡和原始对话。</div></div>
          </details>
        </div>
        <div class="footer-note">本地模型回复可能会出错；关键设定以右侧引用来源为准。</div>
      </section>
    </main>
  </div>
  <script>
    const questionEl = document.getElementById('question');
    const topKEl = document.getElementById('topK');
    const backendEl = document.getElementById('backend');
    const chatModeEl = document.getElementById('chatMode');
    const messagesEl = document.getElementById('messages');
    const welcomeEl = document.getElementById('welcome');
    const sourcesEl = document.getElementById('sources');
    const contextsEl = document.getElementById('contexts');
    const sourcesDetailsEl = document.getElementById('sourcesDetails');
    const contextsDetailsEl = document.getElementById('contextsDetails');
    const traceDetailsEl = document.getElementById('traceDetails');
    const traceBodyEl = document.getElementById('traceBody');
    const statusEl = document.getElementById('status');
    const conversationListEl = document.getElementById('conversationList');
    const sendBtn = document.getElementById('sendBtn');
    const chatBtn = document.getElementById('chatBtn');
    const askBtn = document.getElementById('askBtn');
    const queryBtn = document.getElementById('queryBtn');
    let busy = false;
    let chatHistory = [];
    let currentConversationId = null;
    let composingText = false;

    document.querySelectorAll('[data-question]').forEach((button) => {
      button.addEventListener('click', () => {
        questionEl.value = button.dataset.question || '';
        questionEl.focus();
      });
    });
    conversationListEl.addEventListener('click', async (event) => {
      const deleteButton = event.target.closest('[data-delete-conversation]');
      if (deleteButton) {
        event.stopPropagation();
        await deleteConversation(deleteButton.dataset.deleteConversation);
        return;
      }
      const button = event.target.closest('[data-conversation-id]');
      if (button) {
        await loadConversation(button.dataset.conversationId);
      }
    });
    document.getElementById('newChat').addEventListener('click', () => resetConversation());
    [sourcesDetailsEl, contextsDetailsEl].forEach((details) => {
      details.addEventListener('toggle', () => {
        if (details.open) details.classList.remove('has-new');
      });
    });
    document.addEventListener('click', async (event) => {
      const voiceButton = event.target.closest('[data-play-tts]');
      if (voiceButton) {
        event.preventDefault();
        await playTTS(voiceButton);
        return;
      }
      const traceButton = event.target.closest('[data-trace-scene-id]');
      if (!traceButton) return;
      event.preventDefault();
      await loadTrace(traceButton.dataset.traceSceneId);
    });
    sendBtn.addEventListener('click', () => runChatSaki());
    chatBtn.addEventListener('click', () => runChatSaki());
    askBtn.addEventListener('click', () => runAsk());
    queryBtn.addEventListener('click', () => runQuery());
    questionEl.addEventListener('compositionstart', () => {
      composingText = true;
    });
    questionEl.addEventListener('compositionend', () => {
      composingText = false;
    });
    questionEl.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' && !event.shiftKey && !event.isComposing && !composingText && event.keyCode !== 229) {
        event.preventDefault();
        runChatSaki();
      }
    });
    loadConversations();

    function payload(extra = {}) {
      return {
        question: questionEl.value.trim(),
        top_k: Number(topKEl.value || 4),
        backend: backendEl.value,
        conversation_id: currentConversationId,
        ...extra,
      };
    }

    async function getJSON(url) {
      const response = await fetch(url);
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || response.statusText);
      return data;
    }

    async function postJSON(url, body) {
      const response = await fetch(url, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(body),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || response.statusText);
      return data;
    }

    async function postNDJSONStream(url, body, onEvent) {
      const response = await fetch(url, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(body),
      });
      if (!response.ok) {
        const text = await response.text();
        throw new Error(text || response.statusText);
      }
      if (!response.body) {
        throw new Error('浏览器不支持流式读取。');
      }
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      while (true) {
        const {value, done} = await reader.read();
        buffer += decoder.decode(value || new Uint8Array(), {stream: !done});
        const lines = buffer.split('\\n');
        buffer = lines.pop() || '';
        for (const line of lines) {
          if (line.trim()) onEvent(JSON.parse(line));
        }
        if (done) break;
      }
      if (buffer.trim()) onEvent(JSON.parse(buffer));
    }

    async function deleteJSON(url) {
      const response = await fetch(url, {method: 'DELETE'});
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || response.statusText);
      return data;
    }

    function resetConversation() {
      currentConversationId = null;
      chatHistory = [];
      messagesEl.innerHTML = '';
      welcomeEl.style.display = 'grid';
      sourcesEl.innerHTML = '<div class="empty">还没有引用来源。</div>';
      contextsEl.innerHTML = '<div class="empty">还没有检索片段。</div>';
      traceBodyEl.innerHTML = '<div class="empty">点击来源里的“追溯”查看场景卡和原始对话。</div>';
      traceDetailsEl.open = false;
      resetDetailNotice(sourcesDetailsEl);
      resetDetailNotice(contextsDetailsEl);
      setStatus('待机中');
      renderActiveConversation();
      questionEl.focus();
    }

    async function loadConversations() {
      try {
        const data = await getJSON('/api/conversations');
        renderConversations(data.conversations || []);
      } catch (error) {
        conversationListEl.innerHTML = `<div class="empty">历史聊天读取失败：${escapeText(error.message)}</div>`;
      }
    }

    function renderConversations(conversations) {
      if (!conversations.length) {
        conversationListEl.innerHTML = '<div class="empty">还没有历史聊天。</div>';
        return;
      }
      conversationListEl.innerHTML = conversations.map((conversation) => `
        <div class="conversation-row ${conversation.id === currentConversationId ? 'active' : ''}">
          <button class="history-item" type="button" data-conversation-id="${escapeText(conversation.id)}" title="${escapeText(conversation.title)}">
            ${escapeText(conversation.title || '新聊天')}
          </button>
          <button class="conversation-delete" type="button" data-delete-conversation="${escapeText(conversation.id)}" title="删除聊天">×</button>
        </div>
      `).join('');
    }

    function renderActiveConversation() {
      conversationListEl.querySelectorAll('.conversation-row').forEach((row) => {
        const button = row.querySelector('[data-conversation-id]');
        row.classList.toggle('active', Boolean(button && button.dataset.conversationId === currentConversationId));
      });
    }

    async function loadConversation(conversationId) {
      if (!conversationId || busy) return;
      setBusy(true, '正在读取历史...');
      try {
        const data = await getJSON(`/api/conversations/${conversationId}`);
        currentConversationId = data.conversation.id;
        renderMessages(data.messages || []);
        chatHistory = rebuildChatHistory(data.messages || []);
        renderLastAssistantMetadata(data.messages || []);
        renderActiveConversation();
        setStatus('历史已加载');
      } catch (error) {
        setStatus(`读取失败：${error.message}`);
      } finally {
        setBusy(false);
      }
    }

    async function deleteConversation(conversationId) {
      if (!conversationId || busy) return;
      if (!confirm('删除这段聊天记录？')) return;
      setBusy(true, '正在删除聊天...');
      try {
        await deleteJSON(`/api/conversations/${conversationId}`);
        if (conversationId === currentConversationId) {
          resetConversation();
        }
        await loadConversations();
        setStatus('聊天已删除');
      } catch (error) {
        setStatus(`删除失败：${error.message}`);
      } finally {
        setBusy(false);
      }
    }

    async function runChatSaki() {
      const body = payload({mode: chatModeEl.value, history: chatHistory});
      if (!body.question || busy) return;
      appendMessage('user', body.question);
      const assistantNode = appendMessage('assistant', '');
      assistantNode.classList.add('streaming');
      questionEl.value = '';
      setBusy(true, '咲季正在思考...');
      let fullMessage = '';
      let doneEvent = null;
      try {
        await postNDJSONStream('/api/chat-saki-stream', body, (event) => {
          if (event.type === 'start') {
            currentConversationId = event.conversation_id || currentConversationId;
            setStatus(event.rag_used ? '已检索资料，正在生成...' : '正在生成...');
            return;
          }
          if (event.type === 'delta') {
            fullMessage += event.text || '';
            updateMessage(assistantNode, fullMessage);
            return;
          }
          if (event.type === 'done') {
            doneEvent = event;
            fullMessage = event.message || fullMessage;
            updateMessage(assistantNode, fullMessage, [
              event.mode ? `模式：${event.mode}` : '',
              event.rag_used ? '已检索 RAG' : '未检索',
            ].filter(Boolean));
            currentConversationId = event.conversation_id || currentConversationId;
            renderSources(event.sources || []);
            renderContexts(event.contexts || []);
            return;
          }
          if (event.type === 'error') {
            throw new Error(event.error || '流式生成失败');
          }
        });
        if (!doneEvent) {
          throw new Error('流式生成没有正常结束');
        }
        assistantNode.classList.remove('streaming');
        chatHistory.push({user: body.question, assistant: fullMessage});
        chatHistory = chatHistory.slice(-8);
        await loadConversations();
        setStatus('回复完成');
      } catch (error) {
        assistantNode.classList.remove('streaming');
        updateMessage(assistantNode, fullMessage ? `${fullMessage}\n\n出错了：${error.message}` : `出错了：${error.message}`);
        setStatus('请求失败');
      } finally {
        setBusy(false);
      }
    }

    async function runAsk() {
      const body = payload();
      if (!body.question || busy) return;
      appendMessage('user', body.question);
      questionEl.value = '';
      setBusy(true, '正在检索并生成答案...');
      try {
        const data = await postJSON('/api/ask', body);
        currentConversationId = data.conversation_id || currentConversationId;
        appendMessage('assistant', data.answer || '', ['知识问答']);
        renderSources(data.sources || []);
        renderContexts(data.contexts || []);
        await loadConversations();
        setStatus('问答完成');
      } catch (error) {
        appendMessage('assistant', `出错了：${error.message}`);
        setStatus('请求失败');
      } finally {
        setBusy(false);
      }
    }

    async function runQuery() {
      const body = payload();
      if (!body.question || busy) return;
      setBusy(true, '正在检索片段...');
      try {
        const data = await postJSON('/api/query', body);
        renderSources([]);
        renderContexts(data.results || []);
        appendMessage('assistant', `检索到 ${(data.results || []).length} 条片段，已放到下方“检索片段”。`, ['只检索']);
        setStatus('检索完成');
      } catch (error) {
        appendMessage('assistant', `出错了：${error.message}`);
        setStatus('请求失败');
      } finally {
        setBusy(false);
      }
    }

    function appendMessage(role, text, meta = []) {
      welcomeEl.style.display = 'none';
      const node = document.createElement('div');
      node.className = `message ${role}`;
      const avatar = role === 'assistant'
        ? '<img class="bubble-avatar" src="/assets/saki-avatar.jpg" alt="咲季">'
        : '<div class="bubble-avatar">你</div>';
      const metaHtml = meta.length
        ? `<div class="bubble-meta">${meta.map((item) => `<span class="tag">${escapeText(item)}</span>`).join('')}</div>`
        : '';
      const actionsHtml = role === 'assistant'
        ? `<div class="bubble-actions"><button class="voice-btn" type="button" data-play-tts ${text.trim() ? '' : 'disabled'}>播放语音</button></div>`
        : '';
      node.dataset.voiceText = text;
      node.innerHTML = `${avatar}<div class="bubble"><span class="bubble-text">${escapeText(text)}</span>${metaHtml}${actionsHtml}</div>`;
      messagesEl.appendChild(node);
      scrollChatToBottom();
      return node;
    }

    function updateMessage(node, text, meta = []) {
      node.dataset.voiceText = text;
      const textEl = node.querySelector('.bubble-text');
      if (textEl) textEl.textContent = text;
      const bubble = node.querySelector('.bubble');
      if (!bubble) return;
      const oldMeta = bubble.querySelector('.bubble-meta');
      if (oldMeta) oldMeta.remove();
      if (meta.length) {
        const metaEl = document.createElement('div');
        metaEl.className = 'bubble-meta';
        metaEl.innerHTML = meta.map((item) => `<span class="tag">${escapeText(item)}</span>`).join('');
        const actions = bubble.querySelector('.bubble-actions');
        bubble.insertBefore(metaEl, actions || null);
      }
      const voiceButton = bubble.querySelector('[data-play-tts]');
      if (voiceButton) voiceButton.disabled = !text.trim();
      scrollChatToBottom();
    }

    async function playTTS(button) {
      const messageNode = button.closest('.message');
      if (!messageNode || messageNode.classList.contains('streaming')) {
        setStatus('回复生成完再播放语音');
        return;
      }
      const text = (messageNode.dataset.voiceText || messageNode.querySelector('.bubble-text')?.textContent || '').trim();
      if (!text) return;
      const previousLabel = button.textContent;
      button.disabled = true;
      button.textContent = '生成中...';
      setStatus('正在生成咲季语音...');
      try {
        const data = await postJSON('/api/tts/saki', {text});
        const audio = new Audio(data.audio_url);
        button.textContent = data.cached ? '播放中' : '已生成';
        await audio.play();
        audio.addEventListener('ended', () => {
          button.disabled = false;
          button.textContent = previousLabel;
          setStatus('语音播放完成');
        }, {once: true});
        audio.addEventListener('error', () => {
          button.disabled = false;
          button.textContent = previousLabel;
          setStatus('语音播放失败');
        }, {once: true});
      } catch (error) {
        button.disabled = false;
        button.textContent = previousLabel;
        setStatus(`语音失败：${error.message}`);
      }
    }

    function renderMessages(messages) {
      messagesEl.innerHTML = '';
      if (!messages.length) {
        welcomeEl.style.display = 'grid';
        return;
      }
      welcomeEl.style.display = 'none';
      messages.forEach((message) => {
        const metadata = message.metadata || {};
        const meta = [];
        if (message.role === 'assistant') {
          if (metadata.kind === 'rag-ask') meta.push('知识问答');
          if (metadata.mode) meta.push(`模式：${metadata.mode}`);
          if (metadata.rag_used === true) meta.push('已检索 RAG');
          if (metadata.rag_used === false) meta.push('未检索');
        }
        appendMessage(message.role, message.content, meta);
      });
      requestAnimationFrame(scrollChatToBottom);
    }

    function scrollChatToBottom() {
      const chatScroll = document.getElementById('chatScroll');
      chatScroll.scrollTop = chatScroll.scrollHeight;
    }

    function rebuildChatHistory(messages) {
      const turns = [];
      let pendingUser = null;
      messages.forEach((message) => {
        if (message.role === 'user') {
          pendingUser = message.content;
        } else if (message.role === 'assistant' && pendingUser) {
          turns.push({user: pendingUser, assistant: message.content});
          pendingUser = null;
        }
      });
      return turns.slice(-8);
    }

    function renderLastAssistantMetadata(messages) {
      const assistant = [...messages].reverse().find((message) => message.role === 'assistant');
      const metadata = assistant ? assistant.metadata || {} : {};
      renderSources(metadata.sources || []);
      renderContexts(metadata.contexts || []);
    }

    function setBusy(nextBusy, label = '') {
      busy = nextBusy;
      [sendBtn, chatBtn, askBtn, queryBtn].forEach((button) => button.disabled = nextBusy);
      if (label) setStatus(label);
    }

    function setStatus(text) {
      statusEl.textContent = text;
    }

    function resetDetailNotice(detailsEl) {
      detailsEl.open = false;
      detailsEl.classList.remove('has-new');
    }

    function markDetailNotice(detailsEl, hasItems) {
      detailsEl.open = false;
      detailsEl.classList.toggle('has-new', Boolean(hasItems));
    }

    async function loadTrace(sceneId) {
      if (!sceneId) return;
      traceDetailsEl.open = true;
      traceBodyEl.innerHTML = '<div class="empty">正在读取原文追溯...</div>';
      try {
        const data = await getJSON(`/api/trace?scene_id=${encodeURIComponent(sceneId)}`);
        renderTrace(data);
      } catch (error) {
        traceBodyEl.innerHTML = `<div class="empty">追溯失败：${escapeText(error.message)}</div>`;
      }
    }

    function renderTrace(data) {
      const summary = data.summary || {};
      const dialogue = data.dialogue_lines || [];
      const evidence = Array.isArray(summary.key_evidence) ? summary.key_evidence : [];
      traceBodyEl.innerHTML = `
        <div class="trace-grid">
          <div class="trace-meta">
            <span class="badge">${escapeText(data.scene_id || '')}</span>
            <span>${escapeText(summary.title || '')}</span>
            <span>${escapeText(summary.source_file || '')}</span>
            <span>${escapeText(summary.chapter || '')}</span>
            <span>${escapeText(summary.scene || '')}</span>
          </div>
          <div class="trace-card"><strong>摘要</strong>\n${escapeText(summary.summary || '无摘要。')}</div>
          <div class="trace-card"><strong>咲季定位</strong>\n${escapeText(summary.saki_role || summary.saki_presence || '无。')}</div>
          ${evidence.length ? `<div class="trace-card"><strong>证据摘录</strong>\n${evidence.map((item) => `- ${escapeText(item.source_row || '')} ${escapeText(item.speaker || '')}: ${escapeText(item.text_zh || '')}`).join('\\n')}</div>` : ''}
          <div class="trace-card"><strong>场景卡</strong>\n${escapeText(compact(data.scene_card || '未找到场景卡。', 2200))}</div>
          <div class="trace-card dialogue-card"><strong>原始对话</strong>${dialogue.length ? `<div class="dialogue-list">${dialogue.map(renderDialogueLine).join('')}</div>` : '<div class="empty">未找到原始对话。</div>'}</div>
        </div>
      `;
    }

    function renderDialogueLine(line) {
      return `
        <div class="dialogue-line">
          <div><span class="badge">${escapeText(line.source_row || '')}</span> <strong>${escapeText(line.speaker || '')}</strong></div>
          <div>${escapeText(line.text_zh || '')}</div>
          <div class="dialogue-ja">${escapeText(line.text_ja || '')}</div>
        </div>
      `;
    }

    function renderSources(sources) {
      if (!sources.length) {
        sourcesEl.innerHTML = '<div class="empty">这次没有返回引用来源。</div>';
        markDetailNotice(sourcesDetailsEl, false);
        return;
      }
      sourcesEl.innerHTML = sources.map((source, index) => `
        <div class="source">
          <div class="source-head">
            <span class="badge">${index + 1}</span>
            <span>${escapeText(source.label || source.source_path || '')}</span>
            <span>${escapeText(source.topic || '')}</span>
            <span>${formatDistance(source.distance)}</span>
            <span class="scene-list">${escapeText(formatSceneIds(source.scene_ids || []))}</span>
            ${renderTraceButtons(source.scene_ids || [])}
          </div>
        </div>
      `).join('');
      markDetailNotice(sourcesDetailsEl, true);
    }

    function renderContexts(contexts) {
      if (!contexts.length) {
        contextsEl.innerHTML = '<div class="empty">这次没有返回检索片段。</div>';
        markDetailNotice(contextsDetailsEl, false);
        return;
      }
      contextsEl.innerHTML = contexts.map((item) => {
        const meta = item.metadata || {};
        return `
          <div class="context">
            <div class="source-head">
              <span class="badge">#${escapeText(item.rank || '')}</span>
              <span>${escapeText(meta.source_path || '')}</span>
              <span>${escapeText(meta.topic || '')}</span>
              <span>${formatDistance(item.distance)}</span>
              ${renderTraceButtons(meta.scene_ids || meta.scene_id || '')}
            </div>
            <div class="snippet">${escapeText(compact(item.text || ''))}</div>
          </div>
        `;
      }).join('');
      markDetailNotice(contextsDetailsEl, true);
    }

    function sceneIdList(value) {
      const raw = Array.isArray(value) ? value.join(' ') : String(value || '');
      const matches = raw.match(/scene-\\d{5}/g) || [];
      return [...new Set(matches)];
    }

    function formatSceneIds(value) {
      const ids = sceneIdList(value);
      if (!ids.length) return '';
      return ids.length === 1 ? ids[0] : `${ids.length} scenes: ${ids.join(', ')}`;
    }

    function renderTraceButtons(value) {
      const ids = sceneIdList(value);
      if (!ids.length) return '';
      return `<div class="trace-actions">${ids.map((sceneId) => `<button class="trace-btn" type="button" data-trace-scene-id="${escapeText(sceneId)}">${escapeText(sceneId)}</button>`).join('')}</div>`;
    }

    function compact(text, maxLength = 900) {
      const clean = text.replace(/\\s+/g, ' ').trim();
      return clean.length > maxLength ? `${clean.slice(0, maxLength).trim()}...` : clean;
    }

    function formatDistance(value) {
      return typeof value === 'number' ? value.toFixed(4) : 'n/a';
    }

    function escapeText(value) {
      return String(value).replace(/[&<>"']/g, (char) => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
      }[char]));
    }
  </script>
</body>
</html>"""
    return (
        html.replace("__SUGGESTIONS__", suggestions_html)
        .replace("__TOP_K__", str(settings.config.rag.top_k))
        .replace("__BACKEND_OPTIONS__", backend_options)
        .replace("__CHAT_MODE_OPTIONS__", chat_mode_options)
        .replace("__COLLECTION__", escape_html(settings.collection_name))
        .replace("__EMBED_MODEL__", escape_html(settings.embedding_model))
        .replace("__CHAT_MODEL__", escape_html(settings.chat_model))
    )
