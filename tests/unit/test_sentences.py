from __future__ import annotations

from bionic_head.core.sentences import SentenceBuffer


def test_punctuation_and_max_chars_emit_segments() -> None:
    buffer = SentenceBuffer(max_chars=4)

    assert buffer.push("你好。") == ["你好。"]
    assert buffer.push("12345") == ["1234"]
    assert buffer.flush() == "5"


def test_keeps_residual_and_strips_only_surrounding_whitespace() -> None:
    buffer = SentenceBuffer(max_chars=10)

    assert buffer.push("  你，好") == []
    assert buffer.push("！  next") == ["你，好！"]
    assert buffer.flush() == "next"
