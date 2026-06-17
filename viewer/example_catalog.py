from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExampleSpec:
    example_id: str
    name: str
    method: str
    directory: Path
    ply_path: Path
    camera_path: Path | None
    expected_ply_bytes: int | None
    source_candidates: tuple[Path, ...]
    metadata: dict[str, Any]


def load_example_specs(examples_root: Path, project_root: Path) -> list[ExampleSpec]:
    specs = []
    if not examples_root.is_dir():
        return specs
    for manifest_path in sorted(examples_root.glob("*/scene.json")):
        payload = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        directory = manifest_path.parent
        camera_name = payload.get("cameras")
        specs.append(
            ExampleSpec(
                example_id=str(payload.get("id") or directory.name),
                name=str(payload.get("name") or directory.name),
                method=str(payload.get("method") or "Custom"),
                directory=directory,
                ply_path=directory / str(payload.get("ply") or "point_cloud.ply"),
                camera_path=directory / str(camera_name) if camera_name else None,
                expected_ply_bytes=_optional_int(payload.get("ply_bytes")),
                source_candidates=tuple(
                    project_root / str(candidate) for candidate in payload.get("local_source_candidates", [])
                ),
                metadata=payload,
            )
        )
    return specs


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)
