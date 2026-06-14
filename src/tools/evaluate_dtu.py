#!/usr/bin/env python3
"""Evaluate pose-aligned VGGT point clouds with the DTU point protocol."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.spatial import cKDTree

from src.tools.dtu_camera_utils import camera_centers_from_extrinsics, load_dtu_camera_centers
from src.tools.umeyama import transform_points, umeyama


PLY_TYPES = {
    "char": "i1",
    "uchar": "u1",
    "int8": "i1",
    "uint8": "u1",
    "short": "<i2",
    "ushort": "<u2",
    "int16": "<i2",
    "uint16": "<u2",
    "int": "<i4",
    "uint": "<u4",
    "int32": "<i4",
    "uint32": "<u4",
    "float": "<f4",
    "float32": "<f4",
    "double": "<f8",
    "float64": "<f8",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-id", type=int)
    parser.add_argument("--prediction-dir")
    parser.add_argument("--scan-dir")
    parser.add_argument("--sample-set", default="dtu/SampleSet/MVS Data")
    parser.add_argument("--points-root", default="dtu/Points")
    parser.add_argument("--output-dir", default="outputs/vggt_dtu_benchmark")
    parser.add_argument("--downsample-distance", type=float, default=0.2)
    parser.add_argument("--search-max-distance", type=float, default=60.0)
    parser.add_argument("--stat-max-distance", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--aggregate", action="store_true")
    return parser.parse_args()


def read_ply_vertices(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    with path.open("rb") as handle:
        if handle.readline().strip() != b"ply":
            raise ValueError(f"Not a PLY file: {path}")
        fmt = ""
        vertex_count = None
        vertex_properties: list[tuple[str, str]] = []
        current_element = None
        while True:
            raw = handle.readline()
            if not raw:
                raise ValueError(f"Incomplete PLY header: {path}")
            line = raw.decode("ascii").strip()
            fields = line.split()
            if fields[:1] == ["format"]:
                fmt = fields[1]
            elif fields[:1] == ["element"]:
                current_element = fields[1]
                if current_element == "vertex":
                    vertex_count = int(fields[2])
            elif fields[:1] == ["property"] and current_element == "vertex":
                if fields[1] == "list":
                    raise ValueError(f"List property in vertex element is unsupported: {path}")
                vertex_properties.append((fields[2], fields[1]))
            elif line == "end_header":
                break
        if vertex_count is None or not vertex_properties:
            raise ValueError(f"PLY has no vertex element: {path}")

        names = [name for name, _ in vertex_properties]
        if fmt == "binary_little_endian":
            dtype = np.dtype([(name, PLY_TYPES[kind]) for name, kind in vertex_properties])
            data = np.fromfile(handle, dtype=dtype, count=vertex_count)
            xyz = np.column_stack([data[name] for name in ("x", "y", "z")]).astype(np.float64)
            rgb = None
            if all(name in names for name in ("red", "green", "blue")):
                rgb = np.column_stack([data[name] for name in ("red", "green", "blue")]).astype(np.uint8)
            return xyz, rgb
        if fmt == "ascii":
            data = np.loadtxt(handle, max_rows=vertex_count)
            xyz = data[:, [names.index(name) for name in ("x", "y", "z")]].astype(np.float64)
            rgb = None
            if all(name in names for name in ("red", "green", "blue")):
                rgb = data[:, [names.index(name) for name in ("red", "green", "blue")]].astype(np.uint8)
            return xyz, rgb
        raise ValueError(f"Unsupported PLY format {fmt}: {path}")


def write_binary_ply(path: Path, points: np.ndarray, colors: np.ndarray | None = None) -> None:
    points = np.asarray(points, dtype="<f4")
    if colors is None:
        colors = np.full((len(points), 3), 200, dtype=np.uint8)
    colors = np.asarray(colors, dtype=np.uint8)
    dtype = np.dtype(
        [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    )
    vertices = np.empty(len(points), dtype=dtype)
    for index, name in enumerate(("x", "y", "z")):
        vertices[name] = points[:, index]
    for index, name in enumerate(("red", "green", "blue")):
        vertices[name] = colors[:, index]
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(vertices)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    with path.open("wb") as handle:
        handle.write(header.encode("ascii"))
        vertices.tofile(handle)


def radius_downsample(
    points: np.ndarray, radius: float, seed: int, chunk_size: int = 10_000
) -> np.ndarray:
    """Match DTU reducePts_haa: random greedy suppression within a radius."""
    if radius <= 0 or len(points) == 0:
        return np.arange(len(points), dtype=np.int64)
    tree = cKDTree(points)
    active = np.ones(len(points), dtype=bool)
    order = np.random.default_rng(seed).permutation(len(points))
    for start in range(0, len(order), chunk_size):
        indices = order[start : start + chunk_size]
        neighbours = tree.query_ball_point(points[indices], radius, workers=1)
        for index, close_indices in zip(indices, neighbours):
            if active[index]:
                active[np.asarray(close_indices, dtype=np.int64)] = False
                active[index] = True
    return np.flatnonzero(active)


def bounded_nearest_distances(
    target: np.ndarray,
    source: np.ndarray,
    bounding_box: np.ndarray,
    max_distance: float,
) -> np.ndarray:
    """Equivalent distances for DTU statistics below the official cutoff."""
    distances = np.full(len(source), max_distance, dtype=np.float64)
    span = bounding_box[1] - bounding_box[0]
    covered_high = bounding_box[0] + (np.floor(span / max_distance) + 1.0) * max_distance
    covered = np.all((source >= bounding_box[0]) & (source < covered_high), axis=1)
    if covered.any() and len(target):
        tree = cKDTree(target)
        values, _ = tree.query(
            source[covered], k=1, distance_upper_bound=max_distance, workers=-1
        )
        values[~np.isfinite(values)] = max_distance
        distances[covered] = values
    return distances


def points_in_observation_mask(
    points: np.ndarray, mask: np.ndarray, bounding_box: np.ndarray, resolution: float
) -> np.ndarray:
    voxels = np.floor((points - bounding_box[0]) / resolution + 0.5).astype(np.int64)
    valid = np.all((voxels >= 0) & (voxels < np.asarray(mask.shape)), axis=1)
    result = np.zeros(len(points), dtype=bool)
    selected = voxels[valid]
    result[valid] = mask[selected[:, 0], selected[:, 1], selected[:, 2]].astype(bool)
    return result


def evaluate_scan(args: argparse.Namespace) -> dict:
    if args.scan_id is None or not args.prediction_dir or not args.scan_dir:
        raise ValueError("--scan-id, --prediction-dir, and --scan-dir are required")
    started = time.time()
    scan_id = args.scan_id
    prediction_dir = Path(args.prediction_dir)
    scan_dir = Path(args.scan_dir)
    output_dir = Path(args.output_dir) / f"scan{scan_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    predicted_points, colors = read_ply_vertices(prediction_dir / "points_depth.ply")
    with np.load(prediction_dir / "predictions.npz") as predictions:
        image_names = [str(name) for name in predictions["image_paths"].tolist()]
        predicted_centers = camera_centers_from_extrinsics(predictions["extrinsic"])
    gt_centers, gt_names = load_dtu_camera_centers(scan_dir / "cameras.npz", image_names)
    if image_names != gt_names:
        raise ValueError(f"Image/camera mismatch for scan {scan_id}")

    scale, rotation, translation = umeyama(predicted_centers, gt_centers)
    aligned_centers = transform_points(predicted_centers, scale, rotation, translation)
    aligned_points = transform_points(predicted_points, scale, rotation, translation)
    alignment_rmse = float(np.sqrt(np.mean(np.sum((aligned_centers - gt_centers) ** 2, axis=1))))

    keep = radius_downsample(aligned_points, args.downsample_distance, args.seed)
    aligned_points = aligned_points[keep]
    if colors is not None:
        colors = colors[keep]
    write_binary_ply(output_dir / "points_aligned_downsampled.ply", aligned_points, colors)

    gt_path = Path(args.points_root) / "stl" / f"stl{scan_id:03d}_total.ply"
    gt_points, _ = read_ply_vertices(gt_path)
    mask_path = Path(args.sample_set) / "ObsMask" / f"ObsMask{scan_id}_10.mat"
    mask_data = loadmat(mask_path)
    observation_mask = mask_data["ObsMask"]
    bounding_box = mask_data["BB"].astype(np.float64)
    resolution = float(mask_data["Res"].reshape(-1)[0])
    plane = loadmat(Path(args.sample_set) / "ObsMask" / f"Plane{scan_id}.mat")["P"].reshape(-1)

    data_distances = bounded_nearest_distances(
        gt_points, aligned_points, bounding_box, args.search_max_distance
    )
    stl_distances = bounded_nearest_distances(
        aligned_points, gt_points, bounding_box, args.search_max_distance
    )
    data_valid = points_in_observation_mask(
        aligned_points, observation_mask, bounding_box, resolution
    ) & (data_distances < args.stat_max_distance)
    stl_above_plane = (np.column_stack([gt_points, np.ones(len(gt_points))]) @ plane) > 0
    stl_valid = stl_above_plane & (stl_distances < args.stat_max_distance)
    if not data_valid.any() or not stl_valid.any():
        raise RuntimeError(f"No valid DTU distances for scan {scan_id}")

    accuracy = float(data_distances[data_valid].mean())
    completeness = float(stl_distances[stl_valid].mean())
    report = {
        "scan_id": scan_id,
        "accuracy_mm": accuracy,
        "completeness_mm": completeness,
        "overall_mm": (accuracy + completeness) / 2.0,
        "input_points": int(len(predicted_points)),
        "downsampled_points": int(len(aligned_points)),
        "accuracy_points": int(data_valid.sum()),
        "completeness_points": int(stl_valid.sum()),
        "camera_alignment_rmse_mm": alignment_rmse,
        "sim3_scale": scale,
        "sim3_rotation": rotation.tolist(),
        "sim3_translation": translation.tolist(),
        "alignment": "Umeyama Sim(3), VGGT camera centers to DTU GT camera centers",
        "downsample_distance_mm": args.downsample_distance,
        "search_max_distance_mm": args.search_max_distance,
        "stat_max_distance_mm": args.stat_max_distance,
        "seconds": time.time() - started,
    }
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def aggregate_reports(output_dir: Path) -> dict:
    reports = []
    for path in sorted(output_dir.glob("scan*/metrics.json"), key=lambda p: int(p.parent.name[4:])):
        reports.append(json.loads(path.read_text(encoding="utf-8")))
    if not reports:
        raise RuntimeError(f"No scan reports found under {output_dir}")
    summary = {
        "num_scans": len(reports),
        "scan_ids": [report["scan_id"] for report in reports],
        "mean_accuracy_mm": float(np.mean([report["accuracy_mm"] for report in reports])),
        "mean_completeness_mm": float(np.mean([report["completeness_mm"] for report in reports])),
        "mean_overall_mm": float(np.mean([report["overall_mm"] for report in reports])),
        "alignment": "Umeyama Sim(3), VGGT camera centers to DTU GT camera centers",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["scan_id", "accuracy_mm", "completeness_mm", "overall_mm", "camera_alignment_rmse_mm"],
        )
        writer.writeheader()
        for report in reports:
            writer.writerow({key: report[key] for key in writer.fieldnames})
    print(json.dumps(summary, indent=2))
    return summary


def main() -> int:
    args = parse_args()
    if args.aggregate:
        aggregate_reports(Path(args.output_dir))
    else:
        evaluate_scan(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
