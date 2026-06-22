import asyncio
import json

import httpx
import pytest

from bionic_head.adapters.ollama import OllamaLLMAdapter
from bionic_head.adapters.registry import build_registry
from bionic_head.config import OllamaSettings
from bionic_head.domain.errors import ErrorCode, PipelineException


def _settings() -> OllamaSettings:
    return OllamaSettings(
        base_url="http://ollama.test:11434",
        model="qwen2.5:3b",
        timeout_seconds=2,
    )


def _ndjson_response(lines: list[dict[str, object]], status_code: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code,
        content=b"\n".join(
            json.dumps(line, ensure_ascii=False).encode("utf-8") for line in lines
        ),
    )


@pytest.mark.asyncio
async def test_streams_tokens_and_parses_final_emotion(turn_context) -> None:
    requests: list[httpx.Request] = []
    lines = [
        {"message": {"content": '{"reply":"你好'}, "done": False},
        {"message": {"content": '！","emotion":"friendly","intensity":0.8}'}, "done": False},
        {"done": True},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _ndjson_response(lines)

    adapter = OllamaLLMAdapter(
        settings=_settings(),
        transport=httpx.MockTransport(handler),
    )

    events = [event async for event in adapter.chat_stream("你好", [], turn_context)]

    assert "".join(event.text for event in events if event.kind == "token") == "你好！"
    assert events[-1].kind == "final"
    assert events[-1].result is not None
    assert events[-1].result.reply == "你好！"
    assert events[-1].result.emotion.value == "friendly"
    assert events[-1].result.intensity == pytest.approx(0.8)

    body = json.loads(requests[0].content)
    assert requests[0].method == "POST"
    assert requests[0].url.path == "/api/chat"
    assert body["model"] == "qwen2.5:3b"
    assert body["stream"] is True
    assert body["format"] == "json"
    assert body["messages"][0]["role"] == "system"
    assert body["messages"][-1] == {"role": "user", "content": "你好"}


@pytest.mark.asyncio
async def test_chat_consumes_stream_and_returns_final_result(turn_context) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["messages"][1] == {"role": "assistant", "content": "之前的回复"}
        return _ndjson_response(
            [
                {
                    "message": {
                        "content": '{"reply":"收到","emotion":"calm","intensity":0.4}'
                    },
                    "done": False,
                },
                {"done": True},
            ]
        )

    adapter = OllamaLLMAdapter(
        settings=_settings(),
        transport=httpx.MockTransport(handler),
    )

    result = await adapter.chat(
        "继续",
        [{"role": "assistant", "content": "之前的回复"}],
        turn_context,
    )

    assert result.reply == "收到"
    assert result.emotion.value == "calm"
    assert result.intensity == pytest.approx(0.4)


@pytest.mark.asyncio
async def test_malformed_ndjson_maps_to_provider_failed(turn_context) -> None:
    adapter = OllamaLLMAdapter(
        settings=_settings(),
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, content=b'{"done": false}\nnot-json')
        ),
    )

    with pytest.raises(PipelineException) as raised:
        [event async for event in adapter.chat_stream("你好", [], turn_context)]

    assert raised.value.code is ErrorCode.PROVIDER_FAILED
    assert raised.value.stage == "llm"
    assert raised.value.provider == "ollama"


@pytest.mark.asyncio
async def test_http_failure_maps_to_provider_failed_with_safe_message(turn_context) -> None:
    adapter = OllamaLLMAdapter(
        settings=_settings(),
        transport=httpx.MockTransport(
            lambda request: httpx.Response(500, content=b"/private/ollama/error")
        ),
    )

    with pytest.raises(PipelineException) as raised:
        [event async for event in adapter.chat_stream("你好", [], turn_context)]

    assert raised.value.code is ErrorCode.PROVIDER_FAILED
    assert "/private/ollama/error" not in raised.value.safe_message


@pytest.mark.asyncio
async def test_timeout_maps_to_provider_timeout(turn_context) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("secret timeout detail", request=request)

    adapter = OllamaLLMAdapter(
        settings=_settings(),
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(PipelineException) as raised:
        [event async for event in adapter.chat_stream("你好", [], turn_context)]

    assert raised.value.code is ErrorCode.PROVIDER_TIMEOUT
    assert "secret" not in raised.value.safe_message


@pytest.mark.asyncio
async def test_invalid_model_json_maps_to_output_validation_failed(turn_context) -> None:
    adapter = OllamaLLMAdapter(
        settings=_settings(),
        transport=httpx.MockTransport(
            lambda request: _ndjson_response(
                [
                    {
                        "message": {
                            "content": '{"reply":"嗨","emotion":"alien","intensity":2}'
                        },
                        "done": False,
                    },
                    {"done": True},
                ]
            )
        ),
    )

    with pytest.raises(PipelineException) as raised:
        [event async for event in adapter.chat_stream("你好", [], turn_context)]

    assert raised.value.code is ErrorCode.OUTPUT_VALIDATION_FAILED
    assert raised.value.retryable is False


@pytest.mark.asyncio
async def test_pre_cancelled_context_does_not_call_ollama(turn_context) -> None:
    turn_context.cancellation.cancel()

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("request should not be sent")

    adapter = OllamaLLMAdapter(
        settings=_settings(),
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(asyncio.CancelledError):
        [event async for event in adapter.chat_stream("你好", [], turn_context)]


@pytest.mark.asyncio
async def test_diagnostics_checks_tags_without_generation() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={"models": [{"name": "qwen2.5:3b"}, {"model": "llama3"}]},
        )

    adapter = OllamaLLMAdapter(
        settings=_settings(),
        transport=httpx.MockTransport(handler),
    )

    result = await adapter.diagnostics()

    assert result.adapter == "llm"
    assert result.provider == "ollama"
    assert result.available is True
    assert result.latency_ms >= 0
    assert requests[0].method == "GET"
    assert requests[0].url.path == "/api/tags"


@pytest.mark.asyncio
async def test_diagnostics_reports_missing_model_unavailable() -> None:
    adapter = OllamaLLMAdapter(
        settings=_settings(),
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"models": [{"name": "llama3"}]})
        ),
    )

    result = await adapter.diagnostics()

    assert result.available is False
    assert "qwen2.5:3b" in result.message


def test_registry_builds_ollama_llm_with_other_mock_providers(mock_settings) -> None:
    settings = mock_settings.model_copy(deep=True)
    settings.adapters.llm.provider = "ollama"

    registry = build_registry(settings)

    assert registry.llm.name == "ollama"
    assert registry.asr.name == "mock"
