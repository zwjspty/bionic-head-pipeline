from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from bionic_head.core.cancellation import CancellationToken
from bionic_head.domain.errors import ErrorCode, PipelineException


@dataclass(frozen=True)
class CompletedCommand:
    returncode: int
    stdout: bytes
    stderr: bytes
    debug_stderr: str


def _safe_process_error(
    *,
    code: ErrorCode,
    message: str,
    retryable: bool,
) -> PipelineException:
    return PipelineException(
        code=code,
        stage="process",
        provider=None,
        retryable=retryable,
        message=message,
    )


def _validate_args(args: Sequence[str]) -> list[str]:
    if not args:
        raise _safe_process_error(
            code=ErrorCode.PROVIDER_UNAVAILABLE,
            message="External provider command is unavailable",
            retryable=False,
        )
    validated: list[str] = []
    for arg in args:
        if not isinstance(arg, str) or not arg:
            raise _safe_process_error(
                code=ErrorCode.PROVIDER_UNAVAILABLE,
                message="External provider command is unavailable",
                retryable=False,
            )
        validated.append(arg)
    return validated


async def _terminate_process(
    process: asyncio.subprocess.Process,
    *,
    grace_seconds: float,
) -> None:
    if process.returncode is not None:
        return
    process.terminate()
    try:
        await asyncio.wait_for(process.wait(), timeout=grace_seconds)
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()
    except ProcessLookupError:
        return


async def run_command(
    *,
    args: Sequence[str],
    cwd: Path | str | None,
    stdin: bytes | None,
    timeout_seconds: float,
    cancellation: CancellationToken,
    grace_seconds: float,
) -> CompletedCommand:
    validated_args = _validate_args(args)

    try:
        process = await asyncio.create_subprocess_exec(
            *validated_args,
            cwd=cwd,
            stdin=asyncio.subprocess.PIPE
            if stdin is not None
            else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
    except FileNotFoundError as exc:
        raise _safe_process_error(
            code=ErrorCode.PROVIDER_UNAVAILABLE,
            message="External provider executable is unavailable",
            retryable=False,
        ) from exc
    except PermissionError as exc:
        raise _safe_process_error(
            code=ErrorCode.PROVIDER_UNAVAILABLE,
            message="External provider executable is unavailable",
            retryable=False,
        ) from exc
    except OSError as exc:
        raise _safe_process_error(
            code=ErrorCode.PROVIDER_UNAVAILABLE,
            message="External provider command is unavailable",
            retryable=False,
        ) from exc

    communicate_task = asyncio.create_task(process.communicate(stdin))
    cancel_task = asyncio.create_task(cancellation.wait())

    try:
        done, pending = await asyncio.wait(
            {communicate_task, cancel_task},
            timeout=timeout_seconds,
            return_when=asyncio.FIRST_COMPLETED,
        )

        if communicate_task in done:
            cancel_task.cancel()
            stdout, stderr = await communicate_task
        elif cancel_task in done:
            communicate_task.cancel()
            await _terminate_process(process, grace_seconds=grace_seconds)
            raise asyncio.CancelledError
        else:
            for task in pending:
                task.cancel()
            await _terminate_process(process, grace_seconds=grace_seconds)
            raise _safe_process_error(
                code=ErrorCode.PROVIDER_TIMEOUT,
                message="External provider command timed out",
                retryable=True,
            )
    finally:
        if not cancel_task.done():
            cancel_task.cancel()

    stderr_text = stderr.decode("utf-8", errors="replace")
    if process.returncode != 0:
        raise _safe_process_error(
            code=ErrorCode.PROVIDER_FAILED,
            message="External provider command failed",
            retryable=True,
        )

    return CompletedCommand(
        returncode=process.returncode or 0,
        stdout=stdout,
        stderr=stderr,
        debug_stderr=stderr_text,
    )
