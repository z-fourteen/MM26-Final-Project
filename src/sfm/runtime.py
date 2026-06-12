from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.sfm.camera import PinholeCamera, load_camera_from_feature, load_camera_overrides
from src.sfm.config import load_scene_config, resolve_project_path
from src.sfm.features import list_images
from src.sfm.matching import load_features


@dataclass(frozen=True)
class SfMRuntimeContext:
    scene_path: str
    config: dict
    image_dir: Path
    feature_dir: Path
    verified_dir: Path
    sparse_dir: Path
    output_dir: Path
    report_dir: Path
    image_paths: list[Path]
    keypoints_by_name: dict[str, np.ndarray]
    cameras: dict[str, PinholeCamera]


def load_runtime_context(scene_path: str, focal_scale: float = 1.2) -> SfMRuntimeContext:
    config = load_scene_config(scene_path)
    scene = config["scene"]
    default = config["default"]
    camera_config = default.get("camera", {})
    image_dir = resolve_project_path(scene["image_dir"])
    feature_dir = resolve_project_path(scene["feature_dir"])
    verified_dir = resolve_project_path(scene["verified_dir"])
    sparse_dir = resolve_project_path(scene["sparse_dir"])
    output_dir = resolve_project_path(scene["output_dir"])
    report_dir = output_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list_images(image_dir)
    keypoints_by_name = {}
    cameras = {}
    camera_overrides = load_camera_overrides(sparse_dir)
    for image_path in image_paths:
        feature_path = feature_dir / f"{image_path.stem}.npz"
        features = load_features(feature_path)
        keypoints_by_name[image_path.name] = features.keypoints
        cameras[image_path.name] = load_camera_from_feature(
            feature_path=feature_path,
            image_path=image_path,
            focal_scale=focal_scale,
            prefer_exif=bool(camera_config.get("estimate_focal_from_exif", True)),
            override=camera_overrides.get(image_path.name),
        )

    return SfMRuntimeContext(
        scene_path=scene_path,
        config=config,
        image_dir=image_dir,
        feature_dir=feature_dir,
        verified_dir=verified_dir,
        sparse_dir=sparse_dir,
        output_dir=output_dir,
        report_dir=report_dir,
        image_paths=image_paths,
        keypoints_by_name=keypoints_by_name,
        cameras=cameras,
    )
