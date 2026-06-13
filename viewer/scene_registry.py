from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from viewer.camera_io import load_cameras, read_ply_summary


@dataclass(frozen=True)
class SceneRecord:
    scene_id: str
    label: str
    method: str
    ply_path: Path
    camera_path: Path | None
    cameras: list[dict[str, Any]]
    ply_summary: dict[str, Any]
    source: str

    def public_payload(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "label": self.label,
            "method": self.method,
            "source": self.source,
            "model_url": f"/scene-files/{self.scene_id}/model.ply",
            "cameras": self.cameras,
            "ply": self.ply_summary,
        }


class SceneRegistry:
    def __init__(self) -> None:
        self._records: dict[str, SceneRecord] = {}
        self._lock = threading.Lock()

    def register(
        self,
        *,
        label: str,
        method: str,
        ply_path: Path,
        camera_path: Path | None,
        source: str,
        scene_id: str | None = None,
    ) -> SceneRecord:
        ply_path = ply_path.resolve(strict=True)
        if ply_path.suffix.lower() != ".ply":
            raise ValueError("Model file must use the .ply extension")
        if camera_path is not None:
            camera_path = camera_path.resolve(strict=True)
        record = SceneRecord(
            scene_id=scene_id or uuid.uuid4().hex,
            label=label.strip() or ply_path.stem,
            method=method.strip() or "Custom",
            ply_path=ply_path,
            camera_path=camera_path,
            cameras=load_cameras(camera_path),
            ply_summary=read_ply_summary(ply_path),
            source=source,
        )
        with self._lock:
            self._records[record.scene_id] = record
        return record

    def get(self, scene_id: str) -> SceneRecord:
        with self._lock:
            record = self._records.get(scene_id)
        if record is None:
            raise KeyError(scene_id)
        return record

    def all(self) -> list[SceneRecord]:
        with self._lock:
            return list(self._records.values())
