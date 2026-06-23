from __future__ import annotations

from bionic_head.adapters.morpheus import MorpheusAudio2FaceAdapter
from bionic_head.config import EmoTalkSettings


class EmoTalkAudio2FaceAdapter(MorpheusAudio2FaceAdapter):
    name = "emotalk"
    label = "EmoTalk"
    output_prefix = "emotalk"

    @classmethod
    def from_settings(
        cls,
        settings: EmoTalkSettings,
        *,
        grace_seconds: float,
    ) -> "EmoTalkAudio2FaceAdapter":
        return cls(
            executable=settings.executable,
            args=list(settings.args),
            cwd=settings.cwd,
            output_npy_glob=settings.output_npy_glob,
            output_json_glob=settings.output_json_glob,
            timeout_seconds=settings.timeout_seconds,
            grace_seconds=grace_seconds,
        )
