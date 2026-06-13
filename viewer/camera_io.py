from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def load_cameras(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    suffix = path.suffix.lower()
    if suffix == ".json":
        return _load_json_cameras(path)
    if suffix == ".csv":
        return _load_csv_cameras(path)
    if suffix == ".txt":
        return _load_colmap_images(path)
    raise ValueError("Camera file must be JSON, CSV, or COLMAP images.txt")


def _load_json_cameras(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    raw_cameras = payload.get("cameras", []) if isinstance(payload, dict) else payload
    if not isinstance(raw_cameras, list):
        raise ValueError("Camera JSON must contain a list or a 'cameras' list")

    cameras = []
    for index, raw in enumerate(raw_cameras):
        if not isinstance(raw, dict):
            continue
        name = str(raw.get("image") or raw.get("img_name") or raw.get("image_name") or index)
        width = _optional_float(raw.get("width"))
        height = _optional_float(raw.get("height"))
        fx = _optional_float(raw.get("fx"))
        fy = _optional_float(raw.get("fy"))
        cx = _optional_float(raw.get("cx"))
        cy = _optional_float(raw.get("cy"))

        if "world_from_camera_3x4" in raw:
            transform = np.asarray(raw["world_from_camera_3x4"], dtype=np.float64)
            rotation = transform[:3, :3]
            position = transform[:3, 3]
            intrinsic = raw.get("intrinsic_3x3")
            if intrinsic is not None:
                K = np.asarray(intrinsic, dtype=np.float64)
                fx, fy, cx, cy = map(float, (K[0, 0], K[1, 1], K[0, 2], K[1, 2]))
        elif "position" in raw and "rotation" in raw:
            position = np.asarray(raw["position"], dtype=np.float64)
            rotation = np.asarray(raw["rotation"], dtype=np.float64)
        elif "camera_center_world" in raw:
            position = np.asarray(raw["camera_center_world"], dtype=np.float64)
            rotation = None
        else:
            continue

        cameras.append(
            _camera_record(index, name, position, rotation, width, height, fx, fy, cx, cy)
        )
    return cameras


def _load_csv_cameras(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    cameras = []
    for index, row in enumerate(rows):
        name = str(row.get("image") or row.get("img_name") or row.get("image_name") or index)
        if all(key in row for key in ("camera_center_world_x", "camera_center_world_y", "camera_center_world_z")):
            position = np.array(
                [row["camera_center_world_x"], row["camera_center_world_y"], row["camera_center_world_z"]],
                dtype=np.float64,
            )
        elif all(key in row for key in ("x", "y", "z")):
            position = np.array([row["x"], row["y"], row["z"]], dtype=np.float64)
        else:
            raise ValueError("CSV must contain camera_center_world_x/y/z or x/y/z columns")
        cameras.append(_camera_record(index, name, position, None, None, None, None, None, None, None))
    return cameras


def _load_colmap_images(path: Path) -> list[dict[str, Any]]:
    cameras = []
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    data_lines = [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]
    for line_index in range(0, len(data_lines), 2):
        fields = data_lines[line_index].split()
        if len(fields) < 10:
            continue
        qw, qx, qy, qz = map(float, fields[1:5])
        translation = np.asarray(list(map(float, fields[5:8])), dtype=np.float64)
        world_to_camera = _quaternion_to_rotation(qw, qx, qy, qz)
        camera_to_world = world_to_camera.T
        position = -camera_to_world @ translation
        cameras.append(
            _camera_record(len(cameras), fields[9], position, camera_to_world, None, None, None, None, None, None)
        )
    return cameras


def read_ply_summary(path: Path) -> dict[str, Any]:
    header_lines: list[str] = []
    with path.open("rb") as file:
        for _ in range(512):
            raw = file.readline()
            if not raw:
                break
            line = raw.decode("ascii", errors="replace").strip()
            header_lines.append(line)
            if line == "end_header":
                break
    if not header_lines or header_lines[0] != "ply" or "end_header" not in header_lines:
        raise ValueError("The selected file is not a valid PLY file")
    vertex_count = 0
    properties = []
    ply_format = "unknown"
    for line in header_lines:
        if line.startswith("format "):
            ply_format = line.split()[1]
        elif line.startswith("element vertex "):
            vertex_count = int(line.split()[2])
        elif line.startswith("property "):
            properties.append(line.split()[-1])
    is_gaussian = all(name in properties for name in ("f_dc_0", "opacity", "scale_0", "rot_0"))
    return {
        "file_name": path.name,
        "file_size": path.stat().st_size,
        "vertex_count": vertex_count,
        "format": ply_format,
        "properties": properties,
        "is_gaussian_splat": is_gaussian,
    }


def _camera_record(
    index: int,
    name: str,
    position: np.ndarray,
    rotation: np.ndarray | None,
    width: float | None,
    height: float | None,
    fx: float | None,
    fy: float | None,
    cx: float | None,
    cy: float | None,
) -> dict[str, Any]:
    position = np.asarray(position, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(position)):
        raise ValueError(f"Camera {name} has an invalid position")
    rotation_list = None
    if rotation is not None:
        rotation_array = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
        if not np.all(np.isfinite(rotation_array)):
            raise ValueError(f"Camera {name} has an invalid rotation")
        rotation_list = rotation_array.tolist()
    return {
        "index": index,
        "name": name,
        "position": position.tolist(),
        "rotation": rotation_list,
        "width": width,
        "height": height,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
    }


def _quaternion_to_rotation(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm == 0:
        raise ValueError("COLMAP quaternion has zero norm")
    qw, qx, qy, qz = (value / norm for value in (qw, qx, qy, qz))
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)
