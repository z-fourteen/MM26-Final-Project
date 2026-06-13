from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.sfm.config import load_scene_config, resolve_project_path
from src.sfm.features import list_images
from src.sfm.camera import load_camera_from_feature, load_camera_overrides
from src.sfm.matching import load_features
from src.sfm.reconstruction import load_reconstruction_state


def main() -> int:
    parser = argparse.ArgumentParser(description="Export current reconstruction as COLMAP text sparse model.")
    parser.add_argument("--scene", required=True, help="Path to configs/scenes/<scene>.yaml")
    parser.add_argument("--output-dir", default="", help="Defaults to outputs/<scene>/colmap/sparse/0")
    parser.add_argument("--focal-scale", type=float, default=1.2)
    parser.add_argument(
        "--filter-points-robust-bbox",
        action="store_true",
        help="Export only points inside a robust coordinate bounding box.",
    )
    parser.add_argument("--point-bbox-lower-percentile", type=float, default=1.0)
    parser.add_argument("--point-bbox-upper-percentile", type=float, default=99.0)
    parser.add_argument("--point-bbox-padding-ratio", type=float, default=0.1)
    parser.add_argument("--point-bbox-min-points", type=int, default=100)
    parser.add_argument("--point-filter-report", default="", help="Optional JSON report for exported point filtering.")
    args = parser.parse_args()

    config = load_scene_config(args.scene)
    default = config["default"]
    scene = config["scene"]
    image_dir = resolve_project_path(scene["image_dir"])
    feature_dir = resolve_project_path(scene["feature_dir"])
    sparse_dir = resolve_project_path(scene["sparse_dir"])
    output_dir = Path(args.output_dir) if args.output_dir else resolve_project_path(scene["output_dir"]) / "colmap" / "sparse" / "0"
    output_dir.mkdir(parents=True, exist_ok=True)

    state = load_reconstruction_state(sparse_dir)
    image_paths = list_images(image_dir)
    image_id_by_name = {image_path.name: index + 1 for index, image_path in enumerate(image_paths)}
    image_path_by_name = {image_path.name: image_path for image_path in image_paths}
    registered_names = [name for name in image_id_by_name if name in state.registered_images]

    camera_records = {}
    camera_overrides = load_camera_overrides(sparse_dir)
    for image_name in registered_names:
        camera = load_camera_from_feature(
            feature_path=feature_dir / f"{Path(image_name).stem}.npz",
            image_path=image_path_by_name.get(image_name),
            focal_scale=args.focal_scale,
            prefer_exif=bool(default.get("camera", {}).get("estimate_focal_from_exif", True)),
            override=camera_overrides.get(image_name),
        )
        camera_records[image_id_by_name[image_name]] = (
            camera.width,
            camera.height,
            camera.fx,
            camera.fy,
            camera.cx,
            camera.cy,
        )

    keypoints_by_name = {
        image_name: load_features(feature_dir / f"{Path(image_name).stem}.npz").keypoints
        for image_name in registered_names
    }
    observations_by_image: dict[str, list[dict]] = {image_name: [] for image_name in registered_names}
    point_tracks: dict[int, list[tuple[str, int, int]]] = {}
    for observation in state.observations:
        image_name = str(observation["image_name"])
        if image_name not in image_id_by_name or image_name not in state.registered_images:
            continue
        point_id = int(observation["point3D_id"])
        if point_id < 0 or point_id >= int(state.points3d.shape[0]):
            continue
        keypoint_idx = int(observation["keypoint_idx"])
        point_tracks.setdefault(point_id, []).append((image_name, image_id_by_name[image_name], keypoint_idx))

    point_tracks, point_filter_report = filter_point_tracks_for_export(
        state=state,
        point_tracks=point_tracks,
        enabled=bool(args.filter_points_robust_bbox),
        lower_percentile=float(args.point_bbox_lower_percentile),
        upper_percentile=float(args.point_bbox_upper_percentile),
        padding_ratio=float(args.point_bbox_padding_ratio),
        min_points=int(args.point_bbox_min_points),
    )
    kept_point_ids = set(point_tracks)
    for observation in state.observations:
        image_name = str(observation["image_name"])
        if image_name not in image_id_by_name or image_name not in state.registered_images:
            continue
        point_id = int(observation["point3D_id"])
        if point_id not in kept_point_ids:
            continue
        observations_by_image.setdefault(image_name, []).append(observation)

    write_cameras(output_dir / "cameras.txt", camera_records)
    point2d_idx_lookup = write_images(
        output_dir / "images.txt",
        state,
        registered_names,
        image_id_by_name,
        keypoints_by_name,
        observations_by_image,
    )
    write_points3d(output_dir / "points3D.txt", state, point_tracks, point2d_idx_lookup)
    if args.point_filter_report:
        report_path = resolve_project_path(args.point_filter_report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(point_filter_report, indent=2), encoding="utf-8")

    print(f"Scene: {scene['scene_name']}")
    print(f"Images exported: {len(registered_names)}")
    print(f"Points exported: {len(point_tracks)}")
    if point_filter_report["enabled"]:
        print(f"Points filtered: {point_filter_report['input_points']} -> {point_filter_report['output_points']}")
    print(f"Output: {output_dir}")
    return 0


def filter_point_tracks_for_export(
    *,
    state,
    point_tracks: dict[int, list[tuple[str, int, int]]],
    enabled: bool,
    lower_percentile: float,
    upper_percentile: float,
    padding_ratio: float,
    min_points: int,
) -> tuple[dict[int, list[tuple[str, int, int]]], dict]:
    point_ids = np.array(sorted(point_tracks), dtype=np.int64)
    report = {
        "enabled": bool(enabled),
        "method": "robust_bbox_percentile",
        "input_points": int(len(point_ids)),
        "output_points": int(len(point_ids)),
        "removed_points": 0,
        "lower_percentile": float(lower_percentile),
        "upper_percentile": float(upper_percentile),
        "padding_ratio": float(padding_ratio),
        "min_points": int(min_points),
        "applied": False,
        "reason": "",
    }
    if not enabled:
        report["reason"] = "disabled"
        return point_tracks, report
    if lower_percentile < 0.0 or upper_percentile > 100.0 or lower_percentile >= upper_percentile:
        raise ValueError("Point bbox percentiles must satisfy 0 <= lower < upper <= 100.")
    if len(point_ids) < min_points:
        report["reason"] = "too_few_points"
        return point_tracks, report

    points = state.points3d[point_ids].astype(np.float64)
    finite_mask = np.isfinite(points).all(axis=1)
    finite_points = points[finite_mask]
    finite_ids = point_ids[finite_mask]
    if len(finite_ids) < min_points:
        report["reason"] = "too_few_finite_points"
        return point_tracks, report

    lower = np.percentile(finite_points, lower_percentile, axis=0)
    upper = np.percentile(finite_points, upper_percentile, axis=0)
    padding = np.maximum(upper - lower, 0.0) * max(0.0, padding_ratio)
    lower = lower - padding
    upper = upper + padding
    keep_mask = np.logical_and(finite_points >= lower.reshape(1, 3), finite_points <= upper.reshape(1, 3)).all(axis=1)
    kept_ids = {int(point_id) for point_id in finite_ids[keep_mask]}
    filtered_tracks = {point_id: track for point_id, track in point_tracks.items() if int(point_id) in kept_ids}
    kept_points = state.points3d[np.array(sorted(filtered_tracks), dtype=np.int64)].astype(np.float64)
    report.update(
        {
            "output_points": int(len(filtered_tracks)),
            "removed_points": int(len(point_tracks) - len(filtered_tracks)),
            "applied": True,
            "reason": "ok",
            "bounds_min": lower.tolist(),
            "bounds_max": upper.tolist(),
            "input_bbox_min": np.nanmin(points, axis=0).tolist(),
            "input_bbox_max": np.nanmax(points, axis=0).tolist(),
            "output_bbox_min": np.nanmin(kept_points, axis=0).tolist() if len(kept_points) else [],
            "output_bbox_max": np.nanmax(kept_points, axis=0).tolist() if len(kept_points) else [],
        }
    )
    return filtered_tracks, report


def write_cameras(path: Path, camera_records: dict[int, tuple[int, int, float, float, float, float]]) -> None:
    lines = [
        "# Camera list with one line of data per camera:",
        "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]",
        f"# Number of cameras: {len(camera_records)}",
    ]
    for camera_id, (width, height, fx, fy, cx, cy) in sorted(camera_records.items()):
        lines.append(f"{camera_id} PINHOLE {width} {height} {fx:.12g} {fy:.12g} {cx:.12g} {cy:.12g}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_images(
    path: Path,
    state,
    registered_names: list[str],
    image_id_by_name: dict[str, int],
    keypoints_by_name: dict[str, np.ndarray],
    observations_by_image: dict[str, list[dict]],
) -> dict[tuple[str, int, int], int]:
    point2d_idx_lookup = {}
    lines = [
        "# Image list with two lines of data per image:",
        "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
        "#   POINTS2D[] as (X, Y, POINT3D_ID)",
        f"# Number of images: {len(registered_names)}",
    ]
    for image_name in registered_names:
        image_id = image_id_by_name[image_name]
        registered = state.registered_images[image_name]
        qvec = rotation_matrix_to_qvec(registered.R)
        t = registered.t.reshape(3)
        lines.append(
            f"{image_id} "
            f"{qvec[0]:.17g} {qvec[1]:.17g} {qvec[2]:.17g} {qvec[3]:.17g} "
            f"{t[0]:.17g} {t[1]:.17g} {t[2]:.17g} {image_id} {image_name}"
        )
        points2d = []
        for point2d_idx, observation in enumerate(
            sorted(observations_by_image.get(image_name, []), key=lambda item: int(item["keypoint_idx"]))
        ):
            point3d_id = int(observation["point3D_id"]) + 1
            keypoint_idx = int(observation["keypoint_idx"])
            xy = keypoints_by_name[image_name][keypoint_idx, :2]
            point2d_idx_lookup[(image_name, int(observation["point3D_id"]), keypoint_idx)] = point2d_idx
            points2d.extend([f"{float(xy[0]):.12g}", f"{float(xy[1]):.12g}", str(point3d_id)])
        lines.append(" ".join(points2d))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return point2d_idx_lookup


def write_points3d(
    path: Path,
    state,
    point_tracks: dict[int, list[tuple[str, int, int]]],
    point2d_idx_lookup: dict[tuple[str, int, int], int],
) -> None:
    lines = [
        "# 3D point list with one line of data per point:",
        "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)",
        f"# Number of points: {len(point_tracks)}",
    ]
    for point_id in sorted(point_tracks):
        point = state.points3d[point_id]
        color = state.colors[point_id] if point_id < int(state.colors.shape[0]) else np.array([255, 255, 255])
        track_parts = []
        for image_name, image_id, keypoint_idx in sorted(point_tracks[point_id], key=lambda item: (item[1], item[2])):
            point2d_idx = point2d_idx_lookup.get((image_name, point_id, keypoint_idx))
            if point2d_idx is None:
                continue
            track_parts.append(f"{image_id} {point2d_idx}")
        track = " ".join(track_parts)
        lines.append(
            f"{point_id + 1} {point[0]:.17g} {point[1]:.17g} {point[2]:.17g} "
            f"{int(color[0])} {int(color[1])} {int(color[2])} 0 {track}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def rotation_matrix_to_qvec(R: np.ndarray) -> np.ndarray:
    qvec = np.empty(4, dtype=np.float64)
    qvec[0] = 0.5 * np.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2]))
    qvec[1] = 0.5 * np.sign(R[2, 1] - R[1, 2]) * np.sqrt(max(0.0, 1.0 + R[0, 0] - R[1, 1] - R[2, 2]))
    qvec[2] = 0.5 * np.sign(R[0, 2] - R[2, 0]) * np.sqrt(max(0.0, 1.0 - R[0, 0] + R[1, 1] - R[2, 2]))
    qvec[3] = 0.5 * np.sign(R[1, 0] - R[0, 1]) * np.sqrt(max(0.0, 1.0 - R[0, 0] - R[1, 1] + R[2, 2]))
    norm = np.linalg.norm(qvec)
    if norm > 0:
        qvec /= norm
    return qvec


if __name__ == "__main__":
    raise SystemExit(main())
