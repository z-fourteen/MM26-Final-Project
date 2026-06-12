from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class PinholeCamera:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    source: str = "fallback"

    @property
    def K(self) -> np.ndarray:
        return np.array(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )


def estimate_simple_pinhole(width: int, height: int, focal_scale: float = 1.2) -> PinholeCamera:
    focal = focal_scale * float(max(width, height))
    return PinholeCamera(
        width=width,
        height=height,
        fx=focal,
        fy=focal,
        cx=width / 2.0,
        cy=height / 2.0,
        source="fallback_scale",
    )


def load_camera_from_feature(
    feature_path: Path,
    image_path: Path | None = None,
    focal_scale: float = 1.2,
    prefer_exif: bool = True,
    override: dict | None = None,
) -> PinholeCamera:
    feature_data = np.load(feature_path)
    width, height = [int(value) for value in feature_data["image_size"]]
    if override is not None:
        return PinholeCamera(
            width=int(override.get("width", width)),
            height=int(override.get("height", height)),
            fx=float(override["fx"]),
            fy=float(override.get("fy", override["fx"])),
            cx=float(override.get("cx", width / 2.0)),
            cy=float(override.get("cy", height / 2.0)),
            source=str(override.get("source", "override")),
        )
    if prefer_exif and image_path is not None:
        focal = estimate_focal_pixels_from_exif(image_path, width=width, height=height)
        if focal is not None:
            return PinholeCamera(
                width=width,
                height=height,
                fx=focal,
                fy=focal,
                cx=width / 2.0,
                cy=height / 2.0,
                source="exif",
            )
    return estimate_simple_pinhole(width=width, height=height, focal_scale=focal_scale)


def estimate_focal_pixels_from_exif(image_path: Path, width: int, height: int) -> float | None:
    try:
        from PIL import Image, ExifTags
    except Exception:
        return None
    try:
        with Image.open(image_path) as image:
            exif = image.getexif()
    except Exception:
        return None
    if not exif:
        return None
    tag_by_name = {name: tag for tag, name in ExifTags.TAGS.items()}

    focal_35mm = _exif_float(exif.get(tag_by_name.get("FocalLengthIn35mmFilm")))
    if focal_35mm and focal_35mm > 0:
        sensor_35mm = 36.0
        return float(focal_35mm) / sensor_35mm * float(max(width, height))

    focal_mm = _exif_float(exif.get(tag_by_name.get("FocalLength")))
    x_res = _exif_float(exif.get(tag_by_name.get("FocalPlaneXResolution")))
    y_res = _exif_float(exif.get(tag_by_name.get("FocalPlaneYResolution")))
    unit = exif.get(tag_by_name.get("FocalPlaneResolutionUnit"))
    unit_mm = {2: 25.4, 3: 10.0, 4: 1.0, 5: 0.001}.get(int(unit), None) if unit is not None else None
    if focal_mm and focal_mm > 0 and x_res and y_res and unit_mm:
        fx = float(focal_mm) * float(x_res) / unit_mm
        fy = float(focal_mm) * float(y_res) / unit_mm
        if np.isfinite(fx) and np.isfinite(fy) and fx > 0 and fy > 0:
            return float((fx + fy) * 0.5)
    return None


def _exif_float(value) -> float | None:
    if value is None:
        return None
    try:
        if isinstance(value, tuple) and len(value) == 2:
            return float(value[0]) / float(value[1])
        return float(value)
    except Exception:
        return None


def camera_intrinsics_path(sparse_dir: Path) -> Path:
    return sparse_dir / "camera_intrinsics.json"


def load_camera_overrides(sparse_dir: Path) -> dict[str, dict]:
    path = camera_intrinsics_path(sparse_dir)
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(record["image_name"]): record
        for record in payload.get("cameras", [])
        if "image_name" in record and "fx" in record
    }


def write_camera_overrides(sparse_dir: Path, cameras: dict[str, PinholeCamera], source: str) -> Path:
    path = camera_intrinsics_path(sparse_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": source,
        "cameras": [
            {
                "image_name": image_name,
                "width": camera.width,
                "height": camera.height,
                "fx": camera.fx,
                "fy": camera.fy,
                "cx": camera.cx,
                "cy": camera.cy,
                "source": source,
            }
            for image_name, camera in sorted(cameras.items())
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def projection_matrix(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=np.float64).reshape(3, 1)
    return K @ np.hstack([R.astype(np.float64), t])


def project_points(K: np.ndarray, R: np.ndarray, t: np.ndarray, points3d: np.ndarray) -> np.ndarray:
    points3d = np.asarray(points3d, dtype=np.float64)
    camera_points = (R @ points3d.T + np.asarray(t, dtype=np.float64).reshape(3, 1)).T
    projected = (K @ camera_points.T).T
    return projected[:, :2] / projected[:, 2:3]


def camera_depths(R: np.ndarray, t: np.ndarray, points3d: np.ndarray) -> np.ndarray:
    points3d = np.asarray(points3d, dtype=np.float64)
    camera_points = (R @ points3d.T + np.asarray(t, dtype=np.float64).reshape(3, 1)).T
    return camera_points[:, 2]


def reprojection_errors(
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    points3d: np.ndarray,
    observations: np.ndarray,
) -> np.ndarray:
    projected = project_points(K, R, t, points3d)
    return np.linalg.norm(projected - observations, axis=1)
