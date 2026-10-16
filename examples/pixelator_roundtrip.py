"""Run a simulator-only Pixelator/ZMQ -> Bluesky -> Tiled file round trip."""

import json
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

import h5py
import numpy as np
from bluesky import RunEngine
from bluesky_tiled_plugins import TiledWriter
from tiled.client import from_uri
from tiled.server import SimpleTiledServer

from sxntools.pixelator.client import PixelatorClient
from sxntools.pixelator.flyer import PixelatorFlyer
from sxntools.pixelator.plans import pixelator_scan
from sxntools.pixelator.simulator import PixelatorSimulator


def main() -> None:
    """Check that the complete final HDF5 file is readable from Tiled."""
    definition = json.loads(Path(__file__).with_name("pixelator_request.json").read_text(encoding="utf-8"))
    with TemporaryDirectory() as temporary:
        asset_directory = Path(temporary)
        with PixelatorSimulator(asset_directory, archive=True) as sim:
            client = PixelatorClient(
                sim.request_endpoint,
                sim.publisher_endpoint,
                request_timeout=2.0,
                start_timeout=2.0,
                finish_timeout=5.0,
            )
            flyer = PixelatorFlyer(client, lambda filename: Path(filename).resolve().as_uri())
            with SimpleTiledServer(readable_storage=[str(asset_directory)]) as server:
                tiled_client = from_uri(server.uri)
                with tiled_client.context:
                    engine = RunEngine({})
                    engine.subscribe(TiledWriter(tiled_client, batch_size=1))
                    (uid,) = cast(tuple[str, ...], engine(pixelator_scan(flyer, definition)))
                    run = tiled_client[uid]
                    if run.metadata["stop"]["exit_status"] != "success":
                        raise AssertionError("Tiled did not ingest a successful stop document")
                    if sim.final_path is None or sim.provisional_path is None:
                        raise AssertionError("Simulator did not report both provisional and final filenames")
                    final_uri = sim.final_path.resolve().as_uri()
                    if final_uri == sim.provisional_path.resolve().as_uri():
                        raise AssertionError("Archived scan was not moved out of discard")
                    buffers = {}
                    (key,) = run["primary/pixelator_file"].raw_export(buffers)
                    payload = buffers[key].getvalue()
                    if payload != sim.final_path.read_bytes():
                        raise AssertionError("Downloaded bytes differ from the final HDF5 file")
                    with h5py.File(BytesIO(payload), "r") as file:
                        dataset = cast(h5py.Dataset, file["/entry/data/data"])
                        np.testing.assert_array_equal(dataset[:], np.arange(12).reshape(3, 4))
                    print(uid, final_uri, "Pixelator/ZMQ -> Tiled round trip OK: 12 values")


if __name__ == "__main__":
    main()
