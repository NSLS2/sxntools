"""Scan values exchanged with the Pixelator controller."""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class ScanRequest:
    """Caller-supplied scan definition serialized once for the ZMQ request."""

    json_definition: str

    @classmethod
    def from_definition(cls, definition: Mapping[str, Any]) -> "ScanRequest":
        """Freeze a JSON-compatible definition without imposing a scan schema."""
        return cls(json.dumps(definition, allow_nan=False))


@dataclass(frozen=True)
class ScanResult:
    """Controller's terminal scan filename and outcome flag."""

    filename: str | None
    flag: Literal[0, 1, 2]
