"""Exercise Pixelator REQ acknowledgement separately from terminal PUB events."""

import asyncio
import json
from contextlib import suppress
from pathlib import Path

import pytest
import zmq
import zmq.asyncio

from sxntools.pixelator.client import PixelatorClient, PixelatorControllerError, PixelatorProtocolError, PixelatorScanAborted
from sxntools.pixelator.protocol import ScanRequest
from sxntools.pixelator.simulator import PixelatorSimulator

REQUEST = ScanRequest.from_definition({"scanType": "Sample"})


def _client(sim: PixelatorSimulator, *, request_timeout: float = 1.0, finish_timeout: float = 1.0) -> PixelatorClient:
    return PixelatorClient(
        sim.request_endpoint,
        sim.publisher_endpoint,
        request_timeout=request_timeout,
        start_timeout=1.0,
        finish_timeout=finish_timeout,
    )


def test_acknowledgement_does_not_complete_held_scan(tmp_path: Path) -> None:
    with PixelatorSimulator(tmp_path, auto_finish=False) as sim:

        async def run() -> None:
            client = _client(sim)
            await client.connect()
            try:
                assert await client.abort_scan() is None
                await client.start_scan(REQUEST)
                completion = asyncio.create_task(client.finish_scan())
                with pytest.raises(TimeoutError):
                    await asyncio.wait_for(asyncio.shield(completion), 0.03)
                assert not completion.done()
                assert sim.provisional_path is not None and sim.provisional_path.exists()
                sim.release_scan()
                result = await completion
                assert result.flag == 0
                assert sim.final_path is not None and result.filename == str(sim.final_path)
                assert sim.final_path.exists() and not sim.provisional_path.exists()
            finally:
                await client.close()
                await client.close()

        asyncio.run(run())


def test_abort_while_scan_is_held(tmp_path: Path) -> None:
    with PixelatorSimulator(tmp_path, auto_finish=False) as sim:

        async def run() -> None:
            client = _client(sim)
            await client.connect()
            try:
                await client.start_scan(REQUEST)
                terminal = await client.abort_scan()
                assert terminal is not None and terminal.flag == 1
                assert sim.final_path is None and sim.provisional_path is not None
                assert terminal.filename == str(sim.provisional_path)
                assert await client.abort_scan() == terminal
                with pytest.raises(PixelatorScanAborted) as error:
                    await client.finish_scan()
                assert error.value.result == terminal
            finally:
                await client.close()

        asyncio.run(run())


def test_controller_error_is_not_a_successful_file(tmp_path: Path) -> None:
    with PixelatorSimulator(tmp_path, outcome="error") as sim:

        async def run() -> None:
            client = _client(sim)
            await client.connect()
            try:
                await client.start_scan(REQUEST)
                with pytest.raises(PixelatorControllerError) as error:
                    await client.finish_scan()
                assert error.value.result.flag == 2
                assert sim.final_path is None and sim.provisional_path is not None
                assert error.value.result.filename == str(sim.provisional_path)
            finally:
                await client.close()

        asyncio.run(run())


def test_duplicate_request_is_rejected_before_another_command(tmp_path: Path) -> None:
    with PixelatorSimulator(tmp_path, auto_finish=False) as sim:

        async def run() -> None:
            client = _client(sim)
            await client.connect()
            try:
                await client.start_scan(REQUEST)
                first_path = sim.provisional_path
                with pytest.raises(RuntimeError, match="outstanding"):
                    await client.start_scan(REQUEST)
                assert sim.provisional_path == first_path
                sim.release_scan()
                assert (await client.finish_scan()).flag == 0
            finally:
                await client.close()

        asyncio.run(run())


def test_missing_terminal_times_out_but_abort_still_observes_flag(tmp_path: Path) -> None:
    with PixelatorSimulator(tmp_path, auto_finish=False) as sim:

        async def run() -> None:
            client = _client(sim, finish_timeout=0.05)
            await client.connect()
            try:
                await client.start_scan(REQUEST)
                with pytest.raises(TimeoutError):
                    await client.finish_scan()
                with pytest.raises(RuntimeError, match="uncertain"):
                    await client.start_scan(REQUEST)
                terminal = await client.abort_scan()
                assert terminal is not None and terminal.flag == 1
                assert terminal.filename == str(sim.provisional_path)
                with pytest.raises(PixelatorScanAborted) as error:
                    await client.finish_scan()
                assert error.value.result == terminal
            finally:
                await client.close()

        asyncio.run(run())


def test_closing_with_missing_terminal_keeps_uncertain_latch(tmp_path: Path) -> None:
    with PixelatorSimulator(tmp_path, auto_finish=False) as sim:

        async def run() -> None:
            client = _client(sim, finish_timeout=0.05)
            await client.connect()
            await client.start_scan(REQUEST)
            with pytest.raises(TimeoutError):
                await client.finish_scan()
            await client.close()
            with pytest.raises(RuntimeError, match="uncertain"):
                await client.connect()
            with pytest.raises(RuntimeError, match="uncertain"):
                await client.start_scan(REQUEST)
            assert sim.final_path is None

        asyncio.run(run())


def test_request_timeout_does_not_resend_a_possibly_executed_scan() -> None:
    async def run() -> None:
        context = zmq.asyncio.Context()
        reply = context.socket(zmq.REP)
        publisher = context.socket(zmq.XPUB)
        request_port = reply.bind_to_random_port("tcp://127.0.0.1")
        publisher_port = publisher.bind_to_random_port("tcp://127.0.0.1")
        client = PixelatorClient(
            f"tcp://127.0.0.1:{request_port}",
            f"tcp://127.0.0.1:{publisher_port}",
            request_timeout=0.05,
            start_timeout=0.05,
            finish_timeout=1.0,
        )
        received: list[list[str]] = []
        first_received = asyncio.Event()
        allow_late_reply = asyncio.Event()

        async def controller() -> None:
            assert await asyncio.wait_for(publisher.recv(), 2.0) == b"\x01"
            received.append(json.loads(await asyncio.wait_for(reply.recv(), 2.0)))
            first_received.set()
            await allow_late_reply.wait()
            await reply.send_json(["status", "ok"])
            received.append(json.loads(await asyncio.wait_for(reply.recv(), 2.0)))
            await reply.send_json(["status", "ok"])
            await publisher.send_json(["scanFinished", {"filename": None, "flag": 1}])

        task = asyncio.create_task(controller())
        try:
            await client.connect()
            with pytest.raises(TimeoutError):
                await client.start_scan(REQUEST)
            assert first_received.is_set()
            allow_late_reply.set()
            terminal = await client.abort_scan()
            await task
            assert received == [["scanRequest", REQUEST.json_definition], ["abortScan"]]
            assert terminal is not None and terminal.flag == 1
        finally:
            await client.close()
            if not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
            reply.close(linger=0)
            publisher.close(linger=0)
            context.term()

    asyncio.run(run())


def test_missing_started_event_times_out_and_can_abort() -> None:
    async def run() -> None:
        context = zmq.asyncio.Context()
        reply = context.socket(zmq.REP)
        publisher = context.socket(zmq.XPUB)
        request_port = reply.bind_to_random_port("tcp://127.0.0.1")
        publisher_port = publisher.bind_to_random_port("tcp://127.0.0.1")
        client = PixelatorClient(
            f"tcp://127.0.0.1:{request_port}",
            f"tcp://127.0.0.1:{publisher_port}",
            request_timeout=1.0,
            start_timeout=0.05,
            finish_timeout=1.0,
        )
        received: list[list[str]] = []

        async def controller() -> None:
            assert await asyncio.wait_for(publisher.recv(), 2.0) == b"\x01"
            received.append(json.loads(await asyncio.wait_for(reply.recv(), 2.0)))
            await reply.send_json(["status", "ok"])
            received.append(json.loads(await asyncio.wait_for(reply.recv(), 2.0)))
            await reply.send_json(["status", "ok"])
            await publisher.send_json(["scanFinished", {"filename": None, "flag": 1}])

        task = asyncio.create_task(controller())
        try:
            await client.connect()
            with pytest.raises(TimeoutError):
                await client.start_scan(REQUEST)
            with pytest.raises(RuntimeError, match="uncertain"):
                await client.start_scan(REQUEST)
            terminal = await client.abort_scan()
            await task
            assert received == [["scanRequest", REQUEST.json_definition], ["abortScan"]]
            assert terminal is not None and terminal.flag == 1
        finally:
            await client.close()
            if not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
            reply.close(linger=0)
            publisher.close(linger=0)
            context.term()

    asyncio.run(run())


@pytest.mark.parametrize(
    ("reply_message", "events", "failure_phase"),
    [
        (["status", "no"], [], "start"),
        (["status", "ok"], [["scanStarted", "invalid"]], "start"),
        (["status", "ok"], [["scanStarted", {}], ["scanFinished", {"filename": "/final.h5", "flag": 9}]], "finish"),
        (["status", "ok"], [["scanStarted", {}], ["scanFinished", {"filename": "", "flag": 0}]], "finish"),
    ],
    ids=["non-ok-ack", "malformed-started", "unknown-terminal-flag", "empty-success-filename"],
)
def test_protocol_violations_never_complete_successfully(
    reply_message: list[str], events: list[list[object]], failure_phase: str
) -> None:
    async def run() -> None:
        context = zmq.asyncio.Context()
        reply = context.socket(zmq.REP)
        publisher = context.socket(zmq.XPUB)
        request_port = reply.bind_to_random_port("tcp://127.0.0.1")
        publisher_port = publisher.bind_to_random_port("tcp://127.0.0.1")
        client = PixelatorClient(
            f"tcp://127.0.0.1:{request_port}",
            f"tcp://127.0.0.1:{publisher_port}",
            request_timeout=1.0,
            start_timeout=1.0,
            finish_timeout=1.0,
        )

        async def controller() -> None:
            assert await asyncio.wait_for(publisher.recv(), 2.0) == b"\x01"
            assert json.loads(await asyncio.wait_for(reply.recv(), 2.0)) == ["scanRequest", REQUEST.json_definition]
            for event in events:
                await publisher.send_json(event)
            await reply.send_json(reply_message)

        task = asyncio.create_task(controller())
        try:
            await client.connect()
            if failure_phase == "start":
                with pytest.raises(PixelatorProtocolError):
                    await client.start_scan(REQUEST)
            else:
                await client.start_scan(REQUEST)
                with pytest.raises(PixelatorProtocolError):
                    await client.finish_scan()
            await task
        finally:
            await client.close()
            if not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
            reply.close(linger=0)
            publisher.close(linger=0)
            context.term()

    asyncio.run(run())
