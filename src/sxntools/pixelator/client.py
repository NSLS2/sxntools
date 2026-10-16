"""Async Pixelator REQ/SUB transport for one isolated controller client."""

import asyncio
import json
from contextlib import suppress
from typing import Any

import zmq
import zmq.asyncio

from .protocol import ScanRequest, ScanResult


class PixelatorProtocolError(ValueError):
    """A relevant publisher event or request reply violated the protocol."""


class PixelatorScanAborted(RuntimeError):  # noqa: N818 (documented public name)
    """The controller reported a user-requested abort (flag 1)."""

    def __init__(self, result: ScanResult) -> None:
        self.result = result
        super().__init__(f"Pixelator scan aborted: {result.filename}")


class PixelatorControllerError(RuntimeError):
    """The controller reported a scan error (flag 2)."""

    def __init__(self, result: ScanResult) -> None:
        self.result = result
        super().__init__(f"Pixelator controller error: {result.filename}")


class PixelatorClient:
    """Track one scan over REQ/REP and PUB/SUB without inferring completion from REP.

    The public protocol has no run ID. Do not share an isolated controller with
    other scan clients or connect this implementation to live equipment yet.
    """

    def __init__(
        self,
        request_endpoint: str,
        publisher_endpoint: str,
        *,
        request_timeout: float,
        start_timeout: float,
        finish_timeout: float,
    ) -> None:
        self.request_endpoint = request_endpoint
        self.publisher_endpoint = publisher_endpoint
        self.request_timeout = request_timeout
        self.start_timeout = start_timeout
        self.finish_timeout = finish_timeout
        self._context: zmq.asyncio.Context | None = None
        self._request: zmq.asyncio.Socket | None = None
        self._subscriber: zmq.asyncio.Socket | None = None
        self._listener_task: asyncio.Task[None] | None = None
        self._request_lock = asyncio.Lock()
        self._started: asyncio.Future[None] | None = None
        self._finished: asyncio.Future[ScanResult] | None = None
        self._terminal: ScanResult | None = None
        self._start_failure: Exception | None = None
        self._active = False
        self._uncertain = False
        self._abort_sent = False

    @property
    def scan_active(self) -> bool:
        """Whether a sent request still lacks a valid terminal event."""
        return self._active

    async def connect(self) -> None:
        """Subscribe before any scan request, with a single publisher reader."""
        if self._context is not None:
            return
        if self._uncertain:
            raise RuntimeError("Cannot reconnect after closing with an uncertain scan outcome")
        context = zmq.asyncio.Context()
        try:
            request = context.socket(zmq.REQ)
            request.connect(self.request_endpoint)
            subscriber = context.socket(zmq.SUB)
            # The wiki specifies JSON lists, not ZMQ frame/byte encoding: one
            # UTF-8 JSON-array frame is an unverified simulator-only assumption.
            subscriber.setsockopt(zmq.SUBSCRIBE, b"")
            subscriber.connect(self.publisher_endpoint)
        except BaseException:
            context.destroy(linger=0)
            raise
        self._context = context
        self._request = request
        self._subscriber = subscriber
        self._listener_task = asyncio.create_task(self._listen(subscriber))

    async def start_scan(self, request: ScanRequest) -> None:
        """Wait for both the REP acknowledgement and published scanStarted."""
        if self._uncertain or self._active or (self._finished is not None and not self._finished.done()):
            raise RuntimeError("A Pixelator scan is already outstanding or has an uncertain outcome")
        if self._context is None:
            raise RuntimeError("Connect the Pixelator client before starting a scan")
        loop = asyncio.get_running_loop()
        self._started = loop.create_future()
        self._finished = loop.create_future()
        self._terminal = None
        self._start_failure = None
        self._abort_sent = False
        self._active = True
        try:
            await self._exchange(["scanRequest", request.json_definition])
        except (TimeoutError, PixelatorProtocolError) as exc:
            self._start_failure = exc
            if self._terminal is None:
                self._uncertain = True
            raise
        try:
            await asyncio.wait_for(asyncio.shield(self._started), self.start_timeout)
        except TimeoutError as exc:
            self._start_failure = exc
            if self._terminal is None:
                self._uncertain = True
            raise

    async def finish_scan(self) -> ScanResult:
        """Wait for scanFinished and raise on a controller abort or error."""
        if self._finished is None:
            raise RuntimeError("No Pixelator scan is outstanding")
        if self._start_failure is not None:
            raise self._start_failure
        try:
            result = await asyncio.wait_for(asyncio.shield(self._finished), self.finish_timeout)
        except TimeoutError:
            if self._terminal is None:
                self._uncertain = True
            raise
        if result.flag == 1:
            raise PixelatorScanAborted(result)
        if result.flag == 2:
            raise PixelatorControllerError(result)
        return result

    async def abort_scan(self) -> ScanResult | None:
        """Request abort at most once and await the reported terminal result."""
        if self._terminal is not None:
            return self._terminal
        if not self._active:
            if self._uncertain:
                raise RuntimeError("Cannot abort an uncertain scan after closing its transport")
            return None
        if self._finished is None:
            raise RuntimeError("An active scan has no terminal event subscription")
        if not self._abort_sent:
            async with self._request_lock:
                if self._terminal is not None:
                    return self._terminal
                if not self._abort_sent:
                    self._abort_sent = True
                    try:
                        await self._exchange_locked(["abortScan"])
                    except (TimeoutError, PixelatorProtocolError):
                        if self._terminal is None:
                            self._uncertain = True
                        raise
        try:
            return await asyncio.wait_for(asyncio.shield(self._finished), self.finish_timeout)
        except TimeoutError:
            if self._terminal is None:
                self._uncertain = True
            raise

    async def close(self) -> None:
        """Release ZMQ resources without issuing controller commands."""
        if self._context is None:
            return
        if self._active:
            self._uncertain = True
        if self._listener_task is not None:
            self._listener_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._listener_task
        for future in (self._started, self._finished):
            if future is not None:
                if not future.done():
                    future.cancel()
                elif not future.cancelled():
                    future.exception()
        if self._request is not None:
            self._request.close(linger=0)
        if self._subscriber is not None:
            self._subscriber.close(linger=0)
        self._context.term()
        self._context = None
        self._request = None
        self._subscriber = None
        self._listener_task = None
        self._started = None
        self._finished = None
        self._terminal = None
        self._active = False

    async def _exchange(self, message: list[str]) -> None:
        async with self._request_lock:
            await self._exchange_locked(message)

    async def _exchange_locked(self, message: list[str]) -> None:
        request = self._request
        if request is None:
            raise RuntimeError("Pixelator REQ socket is not connected")

        async def send_receive() -> bytes:
            # Unverified framing assumption; see the matching simulator codec.
            await request.send(json.dumps(message).encode("utf-8"))
            return await request.recv()

        try:
            reply = await asyncio.wait_for(send_receive(), self.request_timeout)
        except TimeoutError:
            # The command may already have executed. Recover REQ's EFSM state,
            # but never resend a scanRequest (or an abortScan).
            self._replace_request()
            raise
        try:
            if json.loads(reply.decode("utf-8")) != ["status", "ok"]:
                raise PixelatorProtocolError(f"Unexpected Pixelator reply: {reply!r}")
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise PixelatorProtocolError("Invalid Pixelator REQ reply") from exc

    def _replace_request(self) -> None:
        if self._context is None or self._request is None:
            return
        self._request.close(linger=0)
        self._request = self._context.socket(zmq.REQ)
        self._request.connect(self.request_endpoint)

    async def _listen(self, subscriber: zmq.asyncio.Socket) -> None:
        try:
            while True:
                try:
                    message = json.loads((await subscriber.recv()).decode("utf-8"))
                    self._on_event(message)
                except (UnicodeError, json.JSONDecodeError, PixelatorProtocolError) as exc:
                    if self._active:
                        error = exc if isinstance(exc, PixelatorProtocolError) else PixelatorProtocolError(str(exc))
                        self._fail_scan(error)
        except asyncio.CancelledError:
            raise
        except zmq.ZMQError as exc:
            if self._active:
                self._fail_scan(PixelatorProtocolError(f"Pixelator publisher connection failed: {exc}"))

    def _on_event(self, message: Any) -> None:
        if self._started is None or self._terminal is not None:
            return
        if not isinstance(message, list) or not message or not isinstance(message[0], str):
            raise PixelatorProtocolError("Pixelator publisher message is not a tagged list")
        kind = message[0]
        if kind == "scanStarted":
            if len(message) != 2 or not isinstance(message[1], dict):
                raise PixelatorProtocolError("Invalid scanStarted definition")
            if not self._started.done():
                self._started.set_result(None)
        elif kind == "scanFinished":
            if len(message) != 2 or not isinstance(message[1], dict):
                raise PixelatorProtocolError("Invalid scanFinished event")
            body = message[1]
            filename = body.get("filename")
            flag = body.get("flag")
            if "filename" not in body or (filename is not None and not isinstance(filename, str)):
                raise PixelatorProtocolError("Invalid terminal filename")
            if type(flag) is not int or flag not in (0, 1, 2):
                raise PixelatorProtocolError(f"Unknown Pixelator scanFinished flag: {flag!r}")
            if flag == 0 and not filename:
                raise PixelatorProtocolError("Successful scanFinished is missing its final filename")
            result = ScanResult(filename, flag)
            self._terminal = result
            self._active = False
            self._uncertain = False
            if self._finished is not None and not self._finished.done():
                self._finished.set_result(result)
        # Provisional filenames, scanStatus, and live chunks are not file assets.

    def _fail_scan(self, error: PixelatorProtocolError) -> None:
        self._uncertain = True
        if self._started is not None and not self._started.done():
            self._started.set_exception(error)
        if self._finished is not None and not self._finished.done():
            self._finished.set_exception(error)
