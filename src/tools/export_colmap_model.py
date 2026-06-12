from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.sfm.config import load_scene_config, resolve_project_path
from src.sfm.features import list_images
from src.sfm.matching import load_features
from src.sfm.reconstruction import load_reconstruction_state


def main() -> int:
    parser = argparse.ArgumentParser(description="Export current reconstruction as COLMAP text sparse model.")
    parser.add_argument("--scene", required=True, help="Path to configs/scenes/<scene>.yaml")
    parser.add_argument("--output-dir", default="", help="Defaults to outputs/<scene>/colmap/sparse/0")
    parser.add_argument("--focal-scale", type=float, default=1.2)
    args = parser.parse_args()

    config = load_scene_config(args.scene)
    scene = config["scene"]
    image_dir = resolve_project_path(scene["image_dir"])
    feature_dir = resolve_project_path(scene["feature_dir"])
    sparse_dir = resolve_project_path(scene["sparse_dir"])
    output_dir = Path(args.output_dir) if args.output_dir else resolve_project_path(scene["output_dir"]) / "colmap" / "sparse" / "0"
    output_dir.mkdir(parents=True, exist_ok=True)

    state = load_reconstruction_state(sparse_dir)
    image_paths = list_images(image_dir)
    image_id_by_name = {image_path.name: index + 1 for index, image_path in enumerate(image_paths)}
    registered_names = [name for name in image_id_by_name if name in state.registered_images]

    camera_records = {}
    for image_name in registered_names:
        feature_data = np.load(feature_dir / f"{Path(image_name).stem}.npz")
        width, height = [int(value) for value in feature_data["image_size"]]
        focal = float(args.focal_scale * max(width, height))
        camera_records[image_id_by_name[image_name]] = (width, height, focal, width / 2.0, height / 2.0)

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
        observations_by_image.setdefault(image_name, []).append(observation)
        point_tracks.setdefault(point_id, []).append((image_name, image_id_by_name[image_name], keypoint_idx))

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

    print(f"Scene: {scene['scene_name']}")
    print(f"Images exported: {len(registered_names)}")
    print(f"Points exported: {len(point_tracks)}")
    print(f"Output: {output_dir}")
    return 0


def write_cameras(path: Path, camera_records: dict[int, tuple[int, int, float, float, float]]) -> None:
    lines = [
        "# Camera list with one line of data per camera:",
        "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]",
        f"# Number of cameras: {len(camera_records)}",
    ]
    for camera_id, (width, height, focal, cx, cy) in sorted(camera_records.items()):
        lines.append(f"{camera_id} SIMPLE_PINHOLE {width} {height} {focal:.12g} {cx:.12g} {cy:.12g}")
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
