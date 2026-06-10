#!/usr/bin/env python3
"""Prepare a GraphDECO 3DGS scene initialized from a VGGT point-cloud PLY.

VGGT's ``points_depth.ply`` is an ordinary colored point cloud with only
``x/y/z/r/g/b``. GraphDECO's gaussian-splatting loader expects
``sparse/0/points3D.ply`` to also contain zero-normal fields ``nx/ny/nz``.
This script reuses an existing COLMAP-style scene's cameras/images and swaps
in a converted VGGT PLY as the 3DGS initialization.
"""

import argparse
import os
import shutil
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a gaussian-splatting scene from VGGT cameras/images plus a VGGT PLY."
    )
    parser.add_argument(
        "--source_scene",
        required=True,
        help="Existing COLMAP-style scene containing images/ and sparse/0/{cameras,images}.txt.",
    )
    parser.add_argument("--source_ply", required=True, help="VGGT points_depth.ply to use as initialization.")
    parser.add_argument("--output_dir", required=True, help="Output scene directory for gaussian-splatting train.py.")
    parser.add_argument(
        "--max_points",
        type=int,
        default=200000,
        help="Maximum points to keep. Use 0 or a negative value to keep all points.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed used when subsampling points.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove output_dir first if it already exists.",
    )
    return parser.parse_args()


def copy_scene_skeleton(source_scene: Path, output_dir: Path, overwrite: bool):
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{output_dir} already exists; pass --overwrite to replace it")
        shutil.rmtree(output_dir)

    images_src = source_scene / "images"
    sparse_src = source_scene / "sparse" / "0"
    if not images_src.is_dir():
        raise FileNotFoundError(f"Missing image directory: {images_src}")
    if not sparse_src.is_dir():
        raise FileNotFoundError(f"Missing sparse/0 directory: {sparse_src}")

    shutil.copytree(images_src, output_dir / "images")
    sparse_dst = output_dir / "sparse" / "0"
    sparse_dst.mkdir(parents=True, exist_ok=True)
    for name in ("cameras.txt", "images.txt", "test.txt"):
        src = sparse_src / name
        if src.exists():
            shutil.copy2(src, sparse_dst / name)
    return sparse_dst


def read_vggt_ply(source_ply: Path):
    vertices = PlyData.read(source_ply)["vertex"]
    xyz = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T.astype(np.float32)
    rgb = np.vstack([vertices["red"], vertices["green"], vertices["blue"]]).T.astype(np.uint8)
    keep = np.isfinite(xyz).all(axis=1)
    return xyz[keep], rgb[keep]


def write_graphdeco_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray):
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]
    elements = np.empty(len(xyz), dtype=dtype)
    normals = np.zeros_like(xyz, dtype=np.float32)
    attributes = np.concatenate([xyz, normals, rgb.astype(np.float32)], axis=1)
    elements[:] = list(map(tuple, attributes))
    PlyData([PlyElement.describe(elements, "vertex")], text=False).write(path)


def main():
    args = parse_args()
    source_scene = Path(args.source_scene)
    source_ply = Path(args.source_ply)
    output_dir = Path(args.output_dir)

    sparse_dst = copy_scene_skeleton(source_scene, output_dir, args.overwrite)
    xyz, rgb = read_vggt_ply(source_ply)
    original_count = len(xyz)

    if args.max_points and args.max_points > 0 and original_count > args.max_points:
        rng = np.random.default_rng(args.seed)
        keep = rng.choice(original_count, size=args.max_points, replace=False)
        xyz = xyz[keep]
        rgb = rgb[keep]

    output_ply = sparse_dst / "points3D.ply"
    write_graphdeco_ply(output_ply, xyz, rgb)

    print(f"source_scene: {source_scene}")
    print(f"source_ply: {source_ply}")
    print(f"output_scene: {output_dir}")
    print(f"points: {original_count} -> {len(xyz)}")
    print(f"graphdeco_init_ply: {output_ply}")


if __name__ == "__main__":
    # Keep relative paths stable when launched from another directory.
    os.environ.setdefault("PYTHONHASHSEED", "0")
    main()
