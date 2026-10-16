"""Ophyd-async Pixelator flyer with opaque file stream assets."""

from collections.abc import AsyncIterator, Callable
from functools import cached_property
from typing import cast
from uuid import uuid4

from bluesky.protocols import StreamAsset
from event_model.documents.event_descriptor import DataKey
from ophyd_async.core import FlyableLogic, StandardFlyable

from .client import PixelatorClient
from .protocol import ScanRequest


class PixelatorFlyableLogic(FlyableLogic[ScanRequest, ScanRequest]):
    """Coordinate transport and final-file mapping across one fly scan."""

    def __init__(self, client: PixelatorClient, path_to_uri: Callable[[str], str]) -> None:
        self.client = client
        self.path_to_uri = path_to_uri
        self.final_file_uri: str | None = None
        self.asset_emitted = False

    async def on_stage(self) -> None:
        """Subscribe before sending a scan command; staging never aborts."""
        await self.client.connect()

    async def on_prepare(self, value: ScanRequest) -> ScanRequest:
        """Discard the preceding run's data without touching the controller."""
        self.final_file_uri = None
        self.asset_emitted = False
        return value

    async def on_kickoff(self, ctx: ScanRequest) -> ScanRequest:
        """Start only after the scan request is acknowledged and published."""
        await self.client.start_scan(ctx)
        return ctx

    async def on_complete(self, ctx: ScanRequest) -> None:
        """Map a successful terminal filename to the catalog URI."""
        result = await self.client.finish_scan()
        # PixelatorClient rejects an empty terminal filename for flag 0.
        uri = self.path_to_uri(cast(str, result.filename))
        if not uri:
            raise ValueError("path_to_uri returned an empty URI for the final Pixelator filename")
        self.final_file_uri = uri

    async def stop(self) -> None:
        """Ask for abort only while the controller has no known terminal event."""
        if self.client.scan_active:
            await self.client.abort_scan()

    async def on_unstage(self) -> None:
        """Stop an active scan and always release its transport."""
        try:
            await self.stop()
        finally:
            await self.client.close()


class PixelatorFlyer(StandardFlyable[ScanRequest, ScanRequest]):
    """One Pixelator scan per run, with a complete file as an opaque asset."""

    def __init__(self, client: PixelatorClient, path_to_uri: Callable[[str], str], *, name: str = "pixelator") -> None:
        super().__init__(name=name)
        self._client = client
        self._path_to_uri = path_to_uri

    @cached_property
    def flyable_logic(self) -> PixelatorFlyableLogic:
        """Share terminal and per-run asset state across flyer methods."""
        return PixelatorFlyableLogic(self._client, self._path_to_uri)

    async def stop(self, success: bool = True) -> None:
        """Satisfy Bluesky Stoppable without wrapping stop in a Status."""
        await self.flyable_logic.stop()

    async def describe_collect(self) -> dict[str, DataKey]:
        """Describe the file only after successful terminal completion."""
        uri = self.flyable_logic.final_file_uri
        if uri is None:
            raise RuntimeError("Cannot describe a Pixelator file before successful completion")
        return {
            "pixelator_file": {
                "source": uri,
                "dtype": "array",
                "dtype_numpy": "|u1",
                "shape": [None],
                "external": "STREAM:",
            }
        }

    async def get_index(self) -> int:
        """Report a single file only after successful terminal URI mapping."""
        return int(self.flyable_logic.final_file_uri is not None)

    async def collect_asset_docs(self, index: int | None = None) -> AsyncIterator[StreamAsset]:
        """Emit one StreamResource/StreamDatum pair per successful run."""
        logic = self.flyable_logic
        if logic.asset_emitted or logic.final_file_uri is None or (index is not None and index < 1):
            return
        resource_uid = str(uuid4())
        logic.asset_emitted = True
        yield (
            "stream_resource",
            {
                "uid": resource_uid,
                "data_key": "pixelator_file",
                "mimetype": "application/octet-stream",
                "uri": logic.final_file_uri,
                "parameters": {},
            },
        )
        yield (
            "stream_datum",
            {
                "uid": f"{resource_uid}/0",
                "stream_resource": resource_uid,
                "descriptor": "",
                "indices": {"start": 0, "stop": 1},
                "seq_nums": {"start": 0, "stop": 0},
            },
        )
