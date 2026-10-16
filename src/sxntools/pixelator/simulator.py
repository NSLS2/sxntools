"""Hardware-free Pixelator REP/XPUB simulator; never binds outside loopback."""

import asyncio
import json
import os
from concurrent.futures import Future
from datetime import date
from pathlib import Path
from threading import Thread
from types import TracebackType
from typing import Any, Literal, Self
from uuid import uuid4

import h5py
import numpy as np
import zmq
import zmq.asyncio

FILENAME = f"Sample_Image_{date.today():%Y-%m-%d}_001.hdf5"


class PixelatorSimulator:
    """Serve simulated scan commands and events from a background event loop.

    Only one isolated client is supported: the public controller protocol has
    no scan identifiers with which to attribute another client's events.

    ``archive=False`` leaves a successful file in the discard directory.
    """

    def __init__(
        self,
        directory: Path,
        *,
        outcome: Literal["success", "abort", "error"] = "success",
        auto_finish: bool = True,
        archive: bool = True,
    ) -> None:
        self.directory = directory
        self.outcome = outcome
        self.auto_finish = auto_finish
        self.archive = archive
        self.request_endpoint: str
        self.publisher_endpoint: str
        self.provisional_path: Path | None = None
        self.final_path: Path | None = None
        self._ready: Future[None] = Future()
        self._thread: Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop: asyncio.Event | None = None
        self._release: asyncio.Event | None = None
        self._completion_task: asyncio.Task[None] | None = None
        self._abort_requested = False
        self._failure: BaseException | None = None

    def __enter__(self) -> Self:
        """Start localhost REP/XPUB sockets and propagate startup errors."""
        self._thread = Thread(target=self._run, name="pixelator-simulator", daemon=True)
        self._thread.start()
        try:
            self._ready.result(timeout=5)
        except BaseException:
            self._thread.join(timeout=5)
            raise
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Stop the server and close its sockets without commanding an abort."""
        if self._loop is not None and self._stop is not None:
            self._loop.call_soon_threadsafe(self._stop.set)
        if self._thread is not None:
            self._thread.join(timeout=5)
            if self._thread.is_alive():
                raise RuntimeError("Pixelator simulator thread did not stop")
        self._thread = None
        self._loop = None
        if self._failure is not None and exc_type is None:
            raise RuntimeError("Pixelator simulator failed") from self._failure

    def release_scan(self) -> None:
        """Allow a held scan to finish on the simulator's event loop."""
        if self._loop is None or self._release is None:
            raise RuntimeError("No simulated scan is available to release")
        self._loop.call_soon_threadsafe(self._release.set)

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._serve())
        except BaseException as exc:
            self._failure = exc
            if not self._ready.done():
                self._ready.set_exception(exc)
        finally:
            loop.close()
            asyncio.set_event_loop(None)

    async def _serve(self) -> None:
        self._stop = asyncio.Event()
        subscribed = asyncio.Event()
        context = zmq.asyncio.Context()
        request = context.socket(zmq.REP)
        publisher = context.socket(zmq.XPUB)
        tasks: list[asyncio.Task[None]] = []
        try:
            request_port = request.bind_to_random_port("tcp://127.0.0.1")
            publisher_port = publisher.bind_to_random_port("tcp://127.0.0.1")
            self.request_endpoint = f"tcp://127.0.0.1:{request_port}"
            self.publisher_endpoint = f"tcp://127.0.0.1:{publisher_port}"
            tasks = [
                asyncio.create_task(self._subscriptions(publisher, subscribed)),
                asyncio.create_task(self._requests(request, publisher, subscribed)),
            ]
            for task in tasks:
                task.add_done_callback(self._record_failure)
            self._ready.set_result(None)
            await self._stop.wait()
        finally:
            for task in tasks:
                task.cancel()
            if self._completion_task is not None:
                self._completion_task.cancel()
                tasks.append(self._completion_task)
            await asyncio.gather(*tasks, return_exceptions=True)
            request.close(linger=0)
            publisher.close(linger=0)
            context.term()

    def _record_failure(self, task: asyncio.Task[None]) -> None:
        if not task.cancelled() and (error := task.exception()) is not None:
            self._failure = error
            if self._stop is not None:
                self._stop.set()

    async def _subscriptions(self, publisher: zmq.asyncio.Socket, subscribed: asyncio.Event) -> None:
        while True:
            # XPUB subscription notices provide deterministic loopback readiness;
            # a real Pixelator PUB does not provide this handshake to clients.
            notice = await publisher.recv()
            if notice == b"\x01":
                subscribed.set()
            elif notice == b"\x00":
                subscribed.clear()

    async def _requests(self, request: zmq.asyncio.Socket, publisher: zmq.asyncio.Socket, subscribed: asyncio.Event) -> None:
        while True:
            # UTF-8 JSON in a single ZMQ frame is an unverified Pixelator codec
            # assumption shared only with the simulator and client.
            message = json.loads((await request.recv()).decode("utf-8"))
            if message == ["abortScan"]:
                await self._send(request, ["status", "ok"])
                if self._completion_task is not None and not self._completion_task.done():
                    self._abort_requested = True
                    if self._release is None:
                        raise RuntimeError("Active simulated scan lost its release event")
                    self._release.set()
            elif isinstance(message, list) and len(message) == 2 and message[0] == "scanRequest":
                if self._completion_task is not None and not self._completion_task.done():
                    await self._send(request, ["status", "busy"])
                    continue
                definition = json.loads(message[1])
                self._abort_requested = False
                release = asyncio.Event()
                self._release = release
                if self.auto_finish:
                    release.set()
                self.final_path = None
                await self._send(request, ["status", "ok"])
                await subscribed.wait()
                provisional = self._write_file()
                self.provisional_path = provisional
                await self._send(publisher, ["scanStarted", definition])
                await self._send(publisher, ["filename", FILENAME, str(provisional.parent)])
                await self._send(
                    publisher,
                    [
                        "scanStatus",
                        {"scanType": "Sample Scan", "status": "running", "current": {"point": 0}, "total": {"point": 12}},
                    ],
                )
                # Timing and release behavior are simulator choices, not a
                # documented controller scan-duration or completion guarantee.
                self._completion_task = asyncio.create_task(self._complete(publisher, provisional, release))
                self._completion_task.add_done_callback(self._record_failure)
            else:
                await self._send(request, ["status", "unsupported"])

    def _write_file(self) -> Path:
        # UUID directories isolate test scans; a real controller uses a dated directory.
        provisional = self.directory / uuid4().hex / "discard" / FILENAME
        provisional.parent.mkdir(parents=True)
        with h5py.File(provisional, "x") as file:
            # Synthetic integrity fixture only: NXstxm names NXdata/data but
            # does not establish Pixelator's entry name or on-disk layout.
            file.create_dataset("/entry/data/data", data=np.arange(12, dtype=np.int32).reshape(3, 4))
        return provisional

    async def _complete(self, publisher: zmq.asyncio.Socket, provisional: Path, release: asyncio.Event) -> None:
        await release.wait()
        if self._abort_requested or self.outcome == "abort":
            flag: Literal[0, 1, 2] = 1
            filename = str(provisional)
        elif self.outcome == "error":
            flag = 2
            filename = str(provisional)
        else:
            flag = 0
            if self.archive:
                final = provisional.parent.parent / FILENAME
                # Both paths are on the same filesystem; linking first prevents
                # rename() from overwriting any existing final-path data.
                os.link(provisional, final)
                provisional.unlink()
            else:
                final = provisional
            self.final_path = final
            filename = str(final)
        await self._send(publisher, ["scanFinished", {"filename": filename, "flag": flag}])

    @staticmethod
    async def _send(socket: zmq.asyncio.Socket, message: list[Any]) -> None:
        await socket.send(json.dumps(message, allow_nan=False).encode("utf-8"))
