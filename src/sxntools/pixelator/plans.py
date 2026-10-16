"""Bluesky plans for Pixelator acquisitions."""

from collections.abc import Mapping
from typing import Any

from bluesky import plan_stubs as bps
from bluesky import preprocessors as bpp
from bluesky.utils import MsgGenerator

from .flyer import PixelatorFlyer
from .protocol import ScanRequest


def pixelator_scan(
    flyer: PixelatorFlyer, definition: Mapping[str, Any], *, md: Mapping[str, Any] | None = None
) -> MsgGenerator:
    """Run one scan and publish its completed file as a stream asset.

    The request definition is caller-supplied; controller validation is not
    replicated here. The documented wire framing is simulator-only until a
    captured controller exchange confirms interoperability.
    """
    request = ScanRequest.from_definition(definition)

    @bpp.stage_decorator([flyer])
    @bpp.run_decorator(md=md)
    def acquire() -> MsgGenerator:
        yield from bps.prepare(flyer, request, wait=True)
        yield from bps.clear_checkpoint()
        yield from bps.kickoff(flyer, wait=True)
        yield from bps.complete(flyer, wait=True)
        # The final file must be known before its stream is described.
        # Bluesky types declare_stream for Readable, but collect=True uses describe_collect.
        yield from bps.declare_stream(flyer, name="primary", collect=True)  # pyright: ignore[reportArgumentType]
        yield from bps.collect(flyer, name="primary", return_payload=False)

    return acquire()
