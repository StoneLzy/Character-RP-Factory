from __future__ import annotations

import json
import threading
import webbrowser
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

from .config import ProjectConfig
from .rag_chat import ask_rag
from .rag_index import RagSearchResult, query_rag_index


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
    class RagWebUIHandler(BaseHTTPRequestHandler):
        server_version = "CRPFWebUI/0.1"

        def do_GET(self) -> None:  # noqa: N802 - http.server naming
            path = urlparse(self.path).path
            if path == "/":
                self.write_html(render_index_html(settings))
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
            self.write_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802 - http.server naming
            path = urlparse(self.path).path
            try:
                payload = self.read_json()
                if path == "/api/query":
                    self.handle_query(payload)
                    return
                if path == "/api/ask":
                    self.handle_ask(payload)
                    return
                self.write_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)
            except ValueError as exc:
                self.write_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            except Exception as exc:
                self.write_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

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
            self.write_json(
                {
                    "question": answer.question,
                    "answer": answer.answer,
                    "sources": [
                        {
                            "label": source.label,
                            "source_path": source.source_path,
                            "topic": source.topic,
                            "scene_ids": list(source.scene_ids),
                            "distance": source.distance,
                        }
                        for source in answer.sources
                    ],
                    "contexts": [serialize_search_result(result) for result in answer.contexts],
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


def serialize_search_result(result: RagSearchResult) -> dict[str, Any]:
    return {
        "rank": result.rank,
        "distance": result.distance,
        "text": result.text,
        "metadata": result.metadata,
    }


def render_index_html(settings: WebUISettings) -> str:
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
    textarea:focus, input:focus {{
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(15, 118, 110, 0.16);
    }}
    .row {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }}
    input {{
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
      <div class="actions">
        <button id="askBtn">问答</button>
        <button id="queryBtn" class="secondary">检索</button>
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
        <h2>回答</h2>
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
    const statusEl = document.getElementById('status');
    const answerEl = document.getElementById('answer');
    const sourcesEl = document.getElementById('sources');
    const contextsEl = document.getElementById('contexts');
    const askBtn = document.getElementById('askBtn');
    const queryBtn = document.getElementById('queryBtn');

    document.querySelectorAll('.example').forEach((button) => {{
      button.addEventListener('click', () => {{
        question.value = button.dataset.question;
      }});
    }});

    askBtn.addEventListener('click', () => runAsk());
    queryBtn.addEventListener('click', () => runQuery());

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

    function setBusy(disabled, text) {{
      askBtn.disabled = disabled;
      queryBtn.disabled = disabled;
      statusEl.className = 'status';
      statusEl.textContent = text;
    }}

    function setError(message) {{
      askBtn.disabled = false;
      queryBtn.disabled = false;
      statusEl.className = 'status error';
      statusEl.textContent = message;
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
