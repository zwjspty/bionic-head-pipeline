from __future__ import annotations


PUNCTUATION = "。！？!?；;\n"


class SentenceBuffer:
    def __init__(self, *, max_chars: int, min_chars: int = 1) -> None:
        if max_chars < 1:
            raise ValueError("max_chars must be at least 1")
        if min_chars < 1:
            raise ValueError("min_chars must be at least 1")
        self.max_chars = max_chars
        self.min_chars = min_chars
        self._buffer = ""

    def push(self, token: str) -> list[str]:
        self._buffer += token
        return self._drain_ready()

    def flush(self) -> str | None:
        segment = self._buffer.strip()
        self._buffer = ""
        return segment or None

    def _drain_ready(self) -> list[str]:
        segments: list[str] = []
        while self._buffer:
            punctuation_index = self._first_eligible_punctuation_index()
            if punctuation_index is not None:
                split_at = punctuation_index + 1
            elif len(self._buffer) >= self.max_chars:
                split_at = self.max_chars
            else:
                break

            raw_segment = self._buffer[:split_at]
            self._buffer = self._buffer[split_at:]
            segment = raw_segment.strip()
            if segment:
                segments.append(segment)
        return segments

    def _first_eligible_punctuation_index(self) -> int | None:
        for index, character in enumerate(self._buffer):
            if character in PUNCTUATION and index + 1 >= self.min_chars:
                return index
        return None
