from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Iterable, Iterator

import requests


SUPPORTED_MODEL_PROVIDERS = {"ollama", "openai_compatible"}


@dataclass(frozen=True)
class ModelProviderConfig:
    provider: str = "ollama"
    model: str = ""
    base_url: str = ""
    api_key_env: str = ""


def normalize_provider(provider: str) -> str:
    normalized = (provider or "ollama").strip().lower()
    if normalized in {"openai", "openai-compatible", "openai_compat", "api"}:
        normalized = "openai_compatible"
    if normalized not in SUPPORTED_MODEL_PROVIDERS:
        raise ValueError(f"Unsupported model provider: {provider}")
    return normalized


def complete_chat(
    config: ModelProviderConfig,
    prompt: str,
    system: str,
    temperature: float = 0.2,
    num_ctx: int = 8192,
    num_predict: int = 900,
    timeout: int = 300,
    json_mode: bool = False,
) -> str:
    provider = normalize_provider(config.provider)
    if provider == "ollama":
        content = complete_ollama_chat(config, prompt, system, temperature, num_ctx, num_predict, timeout, json_mode)
    else:
        content = complete_openai_compatible_chat(config, prompt, system, temperature, num_predict, timeout, json_mode)
    content = strip_thinking(content)
    if not content:
        raise RuntimeError("Chat response is empty")
    return content


def stream_chat(
    config: ModelProviderConfig,
    prompt: str,
    system: str,
    temperature: float = 0.2,
    num_ctx: int = 8192,
    num_predict: int = 900,
    timeout: int = 300,
) -> Iterator[str]:
    provider = normalize_provider(config.provider)
    if provider == "ollama":
        yield from stream_ollama_chat(config, prompt, system, temperature, num_ctx, num_predict, timeout)
    else:
        yield from stream_openai_compatible_chat(config, prompt, system, temperature, num_predict, timeout)


def embed_texts(
    config: ModelProviderConfig,
    texts: Iterable[str],
    batch_size: int = 16,
    timeout: int = 120,
) -> list[list[float]]:
    inputs = [text for text in texts]
    if not inputs:
        return []
    provider = normalize_provider(config.provider)
    if provider == "ollama":
        return ollama_embed_texts(inputs, config, batch_size=batch_size, timeout=timeout)
    return openai_compatible_embed_texts(inputs, config, batch_size=batch_size, timeout=timeout)


def complete_ollama_chat(
    config: ModelProviderConfig,
    prompt: str,
    system: str,
    temperature: float,
    num_ctx: int,
    num_predict: int,
    timeout: int,
    json_mode: bool,
) -> str:
    url = config.base_url.rstrip("/") + "/api/chat"
    payload = {
        "model": config.model,
        "stream": False,
        "think": False,
        "options": {
            "temperature": temperature,
            "top_p": 0.9,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
        },
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    }
    if json_mode:
        payload["format"] = "json"
    raw = post_json(url, payload, timeout=timeout, disable_proxy=True)
    return str(raw.get("message", {}).get("content", "")).strip()


def stream_ollama_chat(
    config: ModelProviderConfig,
    prompt: str,
    system: str,
    temperature: float,
    num_ctx: int,
    num_predict: int,
    timeout: int,
) -> Iterator[str]:
    url = config.base_url.rstrip("/") + "/api/chat"
    payload = {
        "model": config.model,
        "stream": True,
        "think": False,
        "options": {
            "temperature": temperature,
            "top_p": 0.9,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
        },
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    }
    request = urllib_request(url, payload, disable_proxy=True)
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


def complete_openai_compatible_chat(
    config: ModelProviderConfig,
    prompt: str,
    system: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
    json_mode: bool,
) -> str:
    url = config.base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": config.model,
        "stream": False,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    raw = post_json(url, payload, timeout=timeout, api_key_env=config.api_key_env)
    choices = raw.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(f"OpenAI-compatible chat response missing choices: {raw}")
    return str(choices[0].get("message", {}).get("content", "")).strip()


def stream_openai_compatible_chat(
    config: ModelProviderConfig,
    prompt: str,
    system: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
) -> Iterator[str]:
    url = config.base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": config.model,
        "stream": True,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    }
    request = urllib_request(url, payload, api_key_env=config.api_key_env)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line.removeprefix("data:").strip()
                if data == "[DONE]":
                    break
                raw = json.loads(data)
                if raw.get("error"):
                    raise RuntimeError(str(raw["error"]))
                choices = raw.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                content = str(delta.get("content") or "")
                if content:
                    yield content
    except urllib.error.URLError as exc:
        raise RuntimeError(f"OpenAI-compatible chat stream request failed: {exc}") from exc


def ollama_embed_texts(
    texts: list[str],
    config: ModelProviderConfig,
    batch_size: int,
    timeout: int,
) -> list[list[float]]:
    session = requests.Session()
    session.trust_env = False
    embeddings: list[list[float]] = []
    endpoint = f"{config.base_url.rstrip('/')}/api/embed"

    for offset in range(0, len(texts), batch_size):
        batch = texts[offset : offset + batch_size]
        response = session.post(
            endpoint,
            json={"model": config.model, "input": batch},
            timeout=timeout,
        )
        if response.status_code == 404:
            embeddings.extend(ollama_embed_texts_legacy(session, batch, config, timeout))
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
    config: ModelProviderConfig,
    timeout: int,
) -> list[list[float]]:
    endpoint = f"{config.base_url.rstrip('/')}/api/embeddings"
    embeddings: list[list[float]] = []
    for text in texts:
        response = session.post(endpoint, json={"model": config.model, "prompt": text}, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
        embedding = payload.get("embedding")
        if not isinstance(embedding, list):
            raise RuntimeError(f"Ollama legacy embedding response missing 'embedding': {payload}")
        embeddings.append([float(value) for value in embedding])
    return embeddings


def openai_compatible_embed_texts(
    texts: list[str],
    config: ModelProviderConfig,
    batch_size: int,
    timeout: int,
) -> list[list[float]]:
    embeddings: list[list[float]] = []
    endpoint = f"{config.base_url.rstrip('/')}/embeddings"
    for offset in range(0, len(texts), batch_size):
        batch = texts[offset : offset + batch_size]
        payload = {"model": config.model, "input": batch}
        raw = post_json(endpoint, payload, timeout=timeout, api_key_env=config.api_key_env)
        data = raw.get("data")
        if not isinstance(data, list):
            raise RuntimeError(f"OpenAI-compatible embedding response missing data: {raw}")
        data.sort(key=lambda item: int(item.get("index", 0)))
        for item in data:
            embedding = item.get("embedding")
            if not isinstance(embedding, list):
                raise RuntimeError(f"OpenAI-compatible embedding item missing embedding: {item}")
            embeddings.append([float(value) for value in embedding])
    return embeddings


def post_json(
    url: str,
    payload: dict[str, object],
    timeout: int,
    api_key_env: str = "",
    disable_proxy: bool = False,
) -> dict[str, object]:
    request = urllib_request(url, payload, api_key_env=api_key_env, disable_proxy=disable_proxy)
    try:
        if disable_proxy:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Request failed for {url}: {exc}") from exc


def urllib_request(
    url: str,
    payload: dict[str, object],
    api_key_env: str = "",
    disable_proxy: bool = False,
) -> urllib.request.Request:
    headers = {"Content-Type": "application/json"}
    api_key = resolve_api_key(api_key_env)
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )


def resolve_api_key(api_key_env: str) -> str:
    return os.environ.get(api_key_env, "") if api_key_env else ""


def strip_thinking(content: str) -> str:
    return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
