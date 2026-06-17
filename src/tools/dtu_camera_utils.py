#!/usr/bin/env python3
"""Camera helpers for the preprocessed DTU benchmark scenes."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def camera_center_from_projection(projection: np.ndarray) -> np.ndarray:
    """Return the world-space camera center from a 3x4 projection matrix."""
    projection = np.asarray(projection, dtype=np.float64)[:3, :4]
    return np.linalg.solve(projection[:, :3], -projection[:, 3])


def load_dtu_camera_centers(
    cameras_path: str | Path, image_names: list[str] | None = None
) -> tuple[np.ndarray, list[str]]:
    """Load raw DTU camera centers in the same millimetre frame as the GT scan."""
    cameras_path = Path(cameras_path)
    with np.load(cameras_path) as cameras:
        indices = sorted(
            int(key.rsplit("_", 1)[1])
            for key in cameras.files
            if key.startswith("world_mat_") and "_inv_" not in key
        )
        if image_names is not None:
            indices = [int(Path(name).stem) for name in image_names]
        centers = np.stack(
            [camera_center_from_projection(cameras[f"world_mat_{index}"]) for index in indices]
        )
    names = [f"{index:04d}.png" for index in indices]
    return centers, names


def camera_centers_from_extrinsics(extrinsics: np.ndarray) -> np.ndarray:
    """Convert OpenCV world-to-camera extrinsics to world-space centers."""
    extrinsics = np.asarray(extrinsics, dtype=np.float64)
    rotations = extrinsics[:, :3, :3]
    translations = extrinsics[:, :3, 3]
    return -np.einsum("nij,nj->ni", np.swapaxes(rotations, 1, 2), translations)
