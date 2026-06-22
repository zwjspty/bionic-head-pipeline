from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from time import perf_counter
from uuid import UUID

from pydantic import ValidationError

from bionic_head.config import OllamaSettings
from bionic_head.domain.errors import ErrorCode, PipelineException
from bionic_head.domain.models import DiagnosticResult, LLMEvent, LLMResult, TurnContext

try:
    import httpx
except ImportError:  # pragma: no cover - exercised only without the llm extra
    httpx = None  # type: ignore[assignment]


SYSTEM_PROMPT = (
    "Return one JSON object with reply, emotion, intensity. "
    "emotion must be one of neutral,friendly,happy,sad,angry,surprised,thinking,calm. "
    "intensity must be 0.0 to 1.0."
)


def _ollama_error(
    *,
    code: ErrorCode,
    message: str,
    retryable: bool,
) -> PipelineException:
    return PipelineException(
        code=code,
        stage="llm",
        provider="ollama",
        retryable=retryable,
        message=message,
    )


def _provider_unavailable() -> PipelineException:
    return _ollama_error(
        code=ErrorCode.PROVIDER_UNAVAILABLE,
        message="Ollama HTTP client is unavailable",
        retryable=False,
    )


def _provider_failed() -> PipelineException:
    return _ollama_error(
        code=ErrorCode.PROVIDER_FAILED,
        message="Ollama provider request failed",
        retryable=True,
    )


def _provider_timeout() -> PipelineException:
    return _ollama_error(
        code=ErrorCode.PROVIDER_TIMEOUT,
        message="Ollama provider request timed out",
        retryable=True,
    )


def _invalid_output() -> PipelineException:
    return _ollama_error(
        code=ErrorCode.OUTPUT_VALIDATION_FAILED,
        message="Ollama returned invalid structured output",
        retryable=False,
    )


class _ReplyStringStreamer:
    def __init__(self) -> None:
        self._emitted = 0

    def feed(self, json_text: str) -> str:
        decoded = _decode_reply_prefix(json_text)
        if decoded is None:
            return ""
        delta = decoded[self._emitted :]
        self._emitted = len(decoded)
        return delta


def _decode_reply_prefix(json_text: str) -> str | None:
    key_index = json_text.find('"reply"')
    if key_index < 0:
        return None
    colon_index = json_text.find(":", key_index + len('"reply"'))
    if colon_index < 0:
        return None

    index = colon_index + 1
    while index < len(json_text) and json_text[index].isspace():
        index += 1
    if index >= len(json_text) or json_text[index] != '"':
        return None

    return _decode_json_string_prefix(json_text, index + 1)


def _decode_json_string_prefix(json_text: str, start: int) -> str:
    output: list[str] = []
    index = start
    while index < len(json_text):
        character = json_text[index]
        if character == '"':
            break
        if character != "\\":
            output.append(character)
            index += 1
            continue

        if index + 1 >= len(json_text):
            break
        escape = json_text[index + 1]
        if escape == "u":
            if index + 5 >= len(json_text):
                break
            codepoint = json_text[index + 2 : index + 6]
            try:
                output.append(chr(int(codepoint, 16)))
            except ValueError:
                break
            index += 6
            continue
        mapping = {
            '"': '"',
            "\\": "\\",
            "/": "/",
            "b": "\b",
            "f": "\f",
            "n": "\n",
            "r": "\r",
            "t": "\t",
        }
        output.append(mapping.get(escape, escape))
        index += 2
    return "".join(output)


class OllamaLLMAdapter:
    name = "ollama"

    def __init__(self, settings: OllamaSettings, transport: object | None = None) -> None:
        self.settings = settings
        self._transport = transport
        self._active_responses: dict[UUID, object] = {}
        self._active_lock = asyncio.Lock()

    async def chat(
        self,
        text: str,
        history: list[dict[str, str]],
        context: TurnContext,
    ) -> LLMResult:
        final: LLMResult | None = None
        async for event in self.chat_stream(text, history, context):
            if event.kind == "final":
                final = event.result
        if final is None:
            raise _invalid_output()
        return final

    def chat_stream(
        self,
        text: str,
        history: list[dict[str, str]],
        context: TurnContext,
    ) -> AsyncIterator[LLMEvent]:
        return self._chat_stream(text, history, context)

    async def _chat_stream(
        self,
        text: str,
        history: list[dict[str, str]],
        context: TurnContext,
    ) -> AsyncIterator[LLMEvent]:
        if httpx is None:
            raise _provider_unavailable()

        context.cancellation.raise_if_cancelled()
        accumulated = ""
        reply_streamer = _ReplyStringStreamer()
        client = self._client()
        try:
            async with client:
                async with client.stream(
                    "POST",
                    "/api/chat",
                    json=self._build_payload(text, history),
                ) as response:
                    async with self._active_lock:
                        self._active_responses[context.turn_id] = response
                    try:
                        if response.status_code >= 400:
                            raise _provider_failed()
                        async for line in response.aiter_lines():
                            context.cancellation.raise_if_cancelled()
                            if not line.strip():
                                continue
                            data = self._parse_stream_line(line)
                            if "error" in data:
                                raise _provider_failed()
                            content = self._content_from_stream_line(data)
                            if content:
                                accumulated += content
                                token = reply_streamer.feed(accumulated)
                                if token:
                                    yield LLMEvent(kind="token", text=token)
                            if data.get("done") is True:
                                break
                    finally:
                        async with self._active_lock:
                            self._active_responses.pop(context.turn_id, None)
        except asyncio.CancelledError:
            raise
        except PipelineException:
            raise
        except httpx.TimeoutException as exc:
            raise _provider_timeout() from exc
        except httpx.RequestError as exc:
            raise _ollama_error(
                code=ErrorCode.PROVIDER_UNAVAILABLE,
                message="Ollama API is unavailable",
                retryable=True,
            ) from exc

        result = self._parse_result(accumulated)
        yield LLMEvent(kind="final", result=result)

    def _client(self) -> "httpx.AsyncClient":
        if httpx is None:
            raise _provider_unavailable()
        return httpx.AsyncClient(
            base_url=str(self.settings.base_url),
            timeout=self.settings.timeout_seconds,
            transport=self._transport,
        )

    def _build_payload(self, text: str, history: list[dict[str, str]]) -> dict[str, object]:
        messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        for item in history:
            role = item.get("role")
            content = item.get("content")
            if role and content:
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": text})
        return {
            "model": self.settings.model,
            "stream": True,
            "format": "json",
            "messages": messages,
        }

    def _parse_stream_line(self, line: str) -> dict[str, object]:
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            raise _provider_failed() from exc
        if not isinstance(data, dict):
            raise _provider_failed()
        return data

    def _content_from_stream_line(self, data: dict[str, object]) -> str:
        message = data.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content
        response = data.get("response")
        if isinstance(response, str):
            return response
        return ""

    def _parse_result(self, accumulated: str) -> LLMResult:
        try:
            payload = json.loads(accumulated)
        except json.JSONDecodeError as exc:
            raise _invalid_output() from exc
        try:
            return LLMResult.model_validate(payload)
        except ValidationError as exc:
            raise _invalid_output() from exc

    async def diagnostics(self) -> DiagnosticResult:
        started = perf_counter()
        if httpx is None:
            return self._diagnostic(
                available=False,
                started=started,
                message="httpx is not installed; install the llm extra",
            )

        try:
            async with self._client() as client:
                response = await client.get("/api/tags")
            if response.status_code >= 400:
                return self._diagnostic(
                    available=False,
                    started=started,
                    message="Ollama tags endpoint returned an error",
                )
            payload = response.json()
        except httpx.TimeoutException:
            return self._diagnostic(
                available=False,
                started=started,
                message="Ollama diagnostics timed out",
            )
        except httpx.RequestError:
            return self._diagnostic(
                available=False,
                started=started,
                message="Ollama API is unreachable",
            )
        except ValueError:
            return self._diagnostic(
                available=False,
                started=started,
                message="Ollama tags response is invalid",
            )

        model_names = self._model_names(payload)
        if self.settings.model not in model_names:
            return self._diagnostic(
                available=False,
                started=started,
                message=f"Ollama model is not available: {self.settings.model}",
            )
        return self._diagnostic(
            available=True,
            started=started,
            message=f"Ollama model ready: {self.settings.model}",
        )

    def _model_names(self, payload: object) -> set[str]:
        if not isinstance(payload, dict):
            return set()
        models = payload.get("models")
        if not isinstance(models, list):
            return set()
        names: set[str] = set()
        for model in models:
            if not isinstance(model, dict):
                continue
            for key in ("name", "model"):
                value = model.get(key)
                if isinstance(value, str):
                    names.add(value)
        return names

    def _diagnostic(
        self,
        *,
        available: bool,
        started: float,
        message: str,
    ) -> DiagnosticResult:
        return DiagnosticResult(
            adapter="llm",
            provider=self.name,
            available=available,
            latency_ms=(perf_counter() - started) * 1000.0,
            message=message,
        )

    async def cancel(self, turn_id: UUID) -> None:
        async with self._active_lock:
            response = self._active_responses.pop(turn_id, None)
        if response is not None:
            await response.aclose()  # type: ignore[attr-defined]
