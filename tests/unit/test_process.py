import asyncio
import sys

import pytest

from bionic_head.core.cancellation import CancellationToken
from bionic_head.core.process import run_command
from bionic_head.domain.errors import ErrorCode, PipelineException


@pytest.mark.asyncio
async def test_command_captures_stdout_and_stderr(tmp_path) -> None:
    result = await run_command(
        args=[
            sys.executable,
            "-c",
            "import sys; print('ok'); print('note', file=sys.stderr)",
        ],
        cwd=tmp_path,
        stdin=None,
        timeout_seconds=2,
        cancellation=CancellationToken(),
        grace_seconds=0.1,
    )
    assert result.returncode == 0
    assert result.stdout.strip() == b"ok"
    assert result.stderr.strip() == b"note"
    assert result.debug_stderr == "note\n"


@pytest.mark.asyncio
async def test_command_writes_stdin(tmp_path) -> None:
    result = await run_command(
        args=[sys.executable, "-c", "import sys; print(sys.stdin.read().upper())"],
        cwd=tmp_path,
        stdin=b"hello",
        timeout_seconds=2,
        cancellation=CancellationToken(),
        grace_seconds=0.1,
    )
    assert result.stdout.strip() == b"HELLO"


@pytest.mark.asyncio
async def test_nonzero_exit_maps_to_provider_failed_with_safe_message(tmp_path) -> None:
    with pytest.raises(PipelineException) as raised:
        await run_command(
            args=[
                sys.executable,
                "-c",
                "import sys; print('/private/provider/secret', file=sys.stderr); sys.exit(7)",
            ],
            cwd=tmp_path,
            stdin=None,
            timeout_seconds=2,
            cancellation=CancellationToken(),
            grace_seconds=0.1,
        )

    assert raised.value.code is ErrorCode.PROVIDER_FAILED
    assert raised.value.stage == "process"
    assert raised.value.retryable is True
    assert "/private/provider/secret" not in raised.value.safe_message


@pytest.mark.asyncio
async def test_timeout_maps_to_provider_timeout(tmp_path) -> None:
    with pytest.raises(PipelineException) as raised:
        await run_command(
            args=[sys.executable, "-c", "import time; time.sleep(5)"],
            cwd=tmp_path,
            stdin=None,
            timeout_seconds=0.05,
            cancellation=CancellationToken(),
            grace_seconds=0.05,
        )
    assert raised.value.code is ErrorCode.PROVIDER_TIMEOUT
    assert raised.value.retryable is True


@pytest.mark.asyncio
async def test_cancellation_terminates_process(tmp_path) -> None:
    token = CancellationToken()
    task = asyncio.create_task(
        run_command(
            args=[sys.executable, "-c", "import time; time.sleep(5)"],
            cwd=tmp_path,
            stdin=None,
            timeout_seconds=5,
            cancellation=token,
            grace_seconds=0.1,
        )
    )

    await asyncio.sleep(0.05)
    token.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_missing_executable_maps_to_provider_unavailable(tmp_path) -> None:
    with pytest.raises(PipelineException) as raised:
        await run_command(
            args=["/definitely/not/a/provider-command"],
            cwd=tmp_path,
            stdin=None,
            timeout_seconds=2,
            cancellation=CancellationToken(),
            grace_seconds=0.1,
        )

    assert raised.value.code is ErrorCode.PROVIDER_UNAVAILABLE
    assert raised.value.retryable is False
    assert "/definitely/not/a/provider-command" not in raised.value.safe_message


@pytest.mark.asyncio
@pytest.mark.parametrize("args", [[], [""], [sys.executable, ""]])
async def test_argument_array_validation(args, tmp_path) -> None:
    with pytest.raises(PipelineException) as raised:
        await run_command(
            args=args,
            cwd=tmp_path,
            stdin=None,
            timeout_seconds=2,
            cancellation=CancellationToken(),
            grace_seconds=0.1,
        )

    assert raised.value.code is ErrorCode.PROVIDER_UNAVAILABLE
