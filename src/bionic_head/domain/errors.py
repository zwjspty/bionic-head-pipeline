from __future__ import annotations

from enum import Enum

try:  # Python 3.10 fallback for local verification
    from enum import StrEnum
except ImportError:  # pragma: no cover
    class StrEnum(str, Enum):
        pass


class ErrorCode(StrEnum):
    INVALID_REQUEST = "invalid_request"
    INVALID_AUDIO_FORMAT = "invalid_audio_format"
    NO_SPEECH_DETECTED = "no_speech_detected"
    SESSION_LIMIT_REACHED = "session_limit_reached"
    PROTOCOL_VIOLATION = "protocol_violation"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    PROVIDER_TIMEOUT = "provider_timeout"
    PROVIDER_FAILED = "provider_failed"
    OUTPUT_VALIDATION_FAILED = "output_validation_failed"
    TURN_CANCELLED = "turn_cancelled"
    INTERNAL_ERROR = "internal_error"


class PipelineException(Exception):
    def __init__(
        self,
        *,
        code: ErrorCode,
        stage: str,
        provider: str | None,
        retryable: bool,
        message: str,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.stage = stage
        self.provider = provider
        self.retryable = retryable
        self.safe_message = message

    def to_detail(self) -> dict[str, object]:
        return {
            "code": self.code.value,
            "stage": self.stage,
            "provider": self.provider,
            "retryable": self.retryable,
            "message": self.safe_message,
        }
