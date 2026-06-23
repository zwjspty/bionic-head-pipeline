from bionic_head.core.sentences import SentenceBuffer


def test_sentence_buffer_waits_for_min_chars_before_punctuation_flush() -> None:
    buffer = SentenceBuffer(max_chars=24, min_chars=8)

    assert buffer.push("你好！") == []
    assert buffer.push("很高兴。") == ["你好！很高兴。"]


def test_sentence_buffer_forces_flush_at_max_chars_without_punctuation() -> None:
    buffer = SentenceBuffer(max_chars=6, min_chars=4)

    assert buffer.push("一二三") == []
    assert buffer.push("四五六") == ["一二三四五六"]
    assert buffer.flush() is None
