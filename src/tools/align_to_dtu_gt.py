#!/usr/bin/env python3
"""Align a predicted point cloud to DTU GT via camera-center Umeyama Sim(3).

Uses VGGT-predicted camera extrinsics and DTU GT camera projection matrices
to compute a closed-form Sim(3) alignment, then transforms the point cloud
so it lives in the DTU GT millimetre frame.

After alignment, evaluate with::

    python -m src.tools.evaluate_pointcloud --alignment none ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.tools.dtu_camera_utils import camera_centers_from_extrinsics, load_dtu_camera_centers
from src.tools.evaluate_pointcloud import read_ply_vertices
from src.tools.umeyama import transform_points, umeyama


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-npz", required=True, help="VGGT predictions.npz with extrinsic and image_paths")
    parser.add_argument("--gt-cameras", required=True, help="DTU cameras.npz with world_mat_* entries")
    parser.add_argument("--input-ply", required=True, help="Point cloud PLY to transform")
    parser.add_argument("--output-ply", required=True, help="Path for aligned point cloud PLY")
    parser.add_argument("--output-meta", default="", help="Optional JSON metadata path")
    return parser.parse_args()


def write_binary_ply(path: Path, points: np.ndarray) -> None:
    """Write xyz points as binary PLY (gray color)."""
    points = np.asarray(points, dtype="<f4")
    colors = np.full((len(points), 3), 200, dtype=np.uint8)
    dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                       ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    vertices = np.empty(len(points), dtype=dtype)
    for i, name in enumerate(("x", "y", "z")):
        vertices[name] = points[:, i]
    for i, name in enumerate(("red", "green", "blue")):
        vertices[name] = colors[:, i]
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(vertices)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    with path.open("wb") as handle:
        handle.write(header.encode("ascii"))
        vertices.tofile(handle)


def main() -> int:
    args = parse_args()

    # Load VGGT camera centers
    with np.load(args.predictions_npz) as pred:
        image_names = [str(name) for name in pred["image_paths"].tolist()]
        pred_centers = camera_centers_from_extrinsics(pred["extrinsic"])

    # Load DTU GT camera centers (matched to image_names)
    gt_centers, gt_names = load_dtu_camera_centers(args.gt_cameras, image_names)
    if image_names != gt_names:
        raise ValueError("Image/camera name mismatch between predictions and GT cameras")

    # Umeyama Sim(3) alignment
    scale, rotation, translation = umeyama(pred_centers, gt_centers)
    center_rmse = float(np.sqrt(np.mean(
        np.sum((transform_points(pred_centers, scale, rotation, translation) - gt_centers) ** 2, axis=1)
    )))

    # Load and transform point cloud
    points = read_ply_vertices(Path(args.input_ply))
    aligned_points = transform_points(points, scale, rotation, translation)

    # Save
    write_binary_ply(Path(args.output_ply), aligned_points)

    meta = {
        "method": "camera-center Umeyama Sim(3)",
        "scale": scale,
        "rotation": rotation.tolist(),
        "translation": translation.tolist(),
        "camera_center_rmse_mm": center_rmse,
        "num_cameras": len(image_names),
        "input_ply": str(args.input_ply),
        "output_ply": str(args.output_ply),
    }
    meta_path = Path(args.output_meta) if args.output_meta else Path(args.output_ply).with_suffix(".alignment_meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
