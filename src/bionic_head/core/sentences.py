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
            if self._buffer[-1] in PUNCTUATION and len(self._buffer) >= self.min_chars - 1:
                split_at = len(self._buffer)
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

    def _first_punctuation_index(self) -> int | None:
        indexes = [self._buffer.find(mark) for mark in PUNCTUATION if mark in self._buffer]
        if not indexes:
            return None
        return min(index for index in indexes if index >= 0)
