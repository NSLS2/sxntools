"""Prove final Pixelator file assets survive Bluesky and Tiled ingestion."""

import json
from io import BytesIO
from pathlib import Path
from typing import Any, Literal, cast

import h5py
import numpy as np
import pytest
from bluesky import RunEngine
from bluesky.utils import FailedStatus
from bluesky_tiled_plugins import TiledWriter
from tiled.client import from_uri
from tiled.server import SimpleTiledServer

from sxntools.pixelator.client import PixelatorClient, PixelatorControllerError, PixelatorScanAborted
from sxntools.pixelator.flyer import PixelatorFlyer
from sxntools.pixelator.plans import pixelator_scan
from sxntools.pixelator.simulator import PixelatorSimulator

REQUEST_FIXTURE = Path(__file__).resolve().parents[1] / "examples" / "pixelator_request.json"


def _definition() -> dict[str, Any]:
    return json.loads(REQUEST_FIXTURE.read_text(encoding="utf-8"))


def _flyer(sim: PixelatorSimulator) -> PixelatorFlyer:
    client = PixelatorClient(
        sim.request_endpoint,
        sim.publisher_endpoint,
        request_timeout=2.0,
        start_timeout=2.0,
        finish_timeout=5.0,
    )
    return PixelatorFlyer(client, lambda filename: Path(filename).resolve().as_uri())


@pytest.mark.parametrize("archive", [True, False], ids=["archived", "discard"])
def test_two_successful_runs_keep_distinct_final_assets_and_readable_bytes(tmp_path: Path, archive: bool) -> None:
    with PixelatorSimulator(tmp_path, archive=archive) as sim:
        with SimpleTiledServer(readable_storage=[str(tmp_path)]) as server:
            tiled_client = from_uri(server.uri)
            with tiled_client.context:
                engine = RunEngine({})
                flyer = _flyer(sim)
                documents: list[tuple[str, dict[str, Any]]] = []
                described_with_file: list[bool] = []

                def record_document(name: str, doc: Any) -> None:
                    if name == "descriptor":
                        described_with_file.append(flyer.flyable_logic.final_file_uri is not None)
                    documents.append((name, doc))

                engine.subscribe(record_document)
                engine.subscribe(TiledWriter(tiled_client, batch_size=1))
                results: list[tuple[str, str, str]] = []
                for number in range(2):
                    (uid,) = cast(tuple[str, ...], engine(pixelator_scan(flyer, _definition())))
                    run = tiled_client[uid]
                    assert run.metadata["stop"]["exit_status"] == "success"
                    assert sim.provisional_path is not None and sim.final_path is not None
                    final_uri = sim.final_path.resolve().as_uri()
                    assert (final_uri != sim.provisional_path.resolve().as_uri()) is archive
                    resources = [doc for name, doc in documents if name == "stream_resource" and doc["run_start"] == uid]
                    assert len(resources) == 1
                    resource = resources[0]
                    assert resource["data_key"] == "pixelator_file"
                    assert resource["mimetype"] == "application/octet-stream"
                    assert resource["uri"] == final_uri
                    primary_descriptor = next(
                        doc
                        for name, doc in documents
                        if name == "descriptor" and doc["run_start"] == uid and doc["name"] == "primary"
                    )
                    assert described_with_file[-1]
                    assert primary_descriptor["data_keys"]["pixelator_file"]["source"] == final_uri
                    datums = [
                        doc
                        for name, doc in documents
                        if name == "stream_datum" and doc["stream_resource"] == resource["uid"]
                    ]
                    assert len(datums) == 1 and datums[0]["descriptor"] == primary_descriptor["uid"]
                    descriptors = [doc for name, doc in documents if name == "descriptor" and doc["run_start"] == uid]
                    assert descriptors == [primary_descriptor]
                    assert not any(name == "event" for name, _ in documents)
                    buffers = {}
                    (key,) = run["primary/pixelator_file"].raw_export(buffers)
                    payload = buffers[key].getvalue()
                    assert payload == sim.final_path.read_bytes()
                    if number == 0:
                        with h5py.File(BytesIO(payload), "r") as file:
                            dataset = cast(h5py.Dataset, file["/entry/data/data"])
                            np.testing.assert_array_equal(dataset[:], np.arange(12).reshape(3, 4))
                    results.append((uid, resource["uid"], resource["uri"]))
                assert len({uid for uid, _, _ in results}) == 2
                assert len({resource_uid for _, resource_uid, _ in results}) == 2
                assert len({uri for _, _, uri in results}) == 2


@pytest.mark.parametrize(
    ("outcome", "error_type", "flag"),
    [("abort", PixelatorScanAborted, 1), ("error", PixelatorControllerError, 2)],
)
def test_failed_scans_have_distinct_causes_and_no_successful_assets(
    tmp_path: Path, outcome: Literal["abort", "error"], error_type: type[RuntimeError], flag: int
) -> None:
    with PixelatorSimulator(tmp_path, outcome=outcome) as sim:
        with SimpleTiledServer(readable_storage=[str(tmp_path)]) as server:
            tiled_client = from_uri(server.uri)
            with tiled_client.context:
                engine = RunEngine({})
                documents: list[tuple[str, dict[str, Any]]] = []
                engine.subscribe(lambda name, doc: documents.append((name, doc)))
                engine.subscribe(TiledWriter(tiled_client, batch_size=1))
                with pytest.raises(FailedStatus) as failed:
                    engine(pixelator_scan(_flyer(sim), _definition()))
                cause = failed.value.__cause__
                assert isinstance(cause, (PixelatorScanAborted, PixelatorControllerError))
                assert type(cause) is error_type
                assert cause.result.flag == flag
                assert sim.final_path is None and sim.provisional_path is not None
                assert cause.result.filename == str(sim.provisional_path)
                starts = [doc for name, doc in documents if name == "start"]
                assert len(starts) == 1
                uid = starts[0]["uid"]
                stops = [doc for name, doc in documents if name == "stop"]
                assert len(stops) == 1 and stops[0]["run_start"] == uid
                assert stops[0]["exit_status"] != "success"
                assert tiled_client[uid].metadata["stop"]["exit_status"] != "success"
                assert not any(name == "descriptor" for name, _ in documents)
                assert not [doc for name, doc in documents if name in ("stream_resource", "stream_datum")]
