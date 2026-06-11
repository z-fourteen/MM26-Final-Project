from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from src.sfm.camera import estimate_simple_pinhole, projection_matrix
from src.sfm.config import load_scene_config, resolve_project_path
from src.sfm.features import list_images
from src.sfm.matching import load_features
from src.sfm.reconstruction import (
    ReconstructionState,
    load_reconstruction_state,
    observation_lookup,
    save_reconstruction_state,
)
from src.sfm.triangulation import robust_triangulate_track


@dataclass(frozen=True)
class TrackObservation:
    image_name: str
    keypoint_idx: int


@dataclass(frozen=True)
class TrackBuildResult:
    tracks: list[list[TrackObservation]]
    valid_edges: set[tuple[str, str]]
    edge_status_counts: dict[str, int]
    edge_model_type_counts: dict[str, int]
    used_edges: int
    skipped_edges: int
    skipped_matches_by_residual: int


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[TrackObservation, TrackObservation] = {}

    def add(self, item: TrackObservation) -> None:
        self.parent.setdefault(item, item)

    def find(self, item: TrackObservation) -> TrackObservation:
        self.add(item)
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: TrackObservation, right: TrackObservation) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left != root_right:
            self.parent[root_right] = root_left

    def groups(self) -> list[list[TrackObservation]]:
        grouped: dict[TrackObservation, list[TrackObservation]] = {}
        for item in list(self.parent):
            grouped.setdefault(self.find(item), []).append(item)
        return list(grouped.values())


def read_rgb(image_path: Path) -> np.ndarray:
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Failed to read image: {image_path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def sample_color(image_rgb: np.ndarray, point2d: np.ndarray) -> np.ndarray:
    height, width = image_rgb.shape[:2]
    x = int(np.clip(round(float(point2d[0])), 0, width - 1))
    y = int(np.clip(round(float(point2d[1])), 0, height - 1))
    return image_rgb[y, x].astype(np.uint8)


def build_registered_tracks(
    verified_dir: Path,
    registered_names: set[str],
    blocked_observations: set[TrackObservation] | None = None,
) -> TrackBuildResult:
    union_find = UnionFind()
    valid_edges: set[tuple[str, str]] = set()
    edge_status_counts: dict[str, int] = {}
    edge_model_type_counts: dict[str, int] = {}
    used_edges = 0
    skipped_edges = 0
    skipped_matches_by_residual = 0
    blocked_observations = blocked_observations or set()
    for verified_path in sorted(verified_dir.glob("*.npz")):
        data = np.load(verified_path)
        image_name1 = str(data["image_name1"])
        image_name2 = str(data["image_name2"])
        if image_name1 not in registered_names or image_name2 not in registered_names:
            continue
        for required_field in ("status", "model_type", "inlier_matches", "homography_ratio"):
            if required_field not in data:
                raise KeyError(f"Phase 6B requires field '{required_field}' in {verified_path}")
        status = str(data["status"])
        model_type = str(data["model_type"])
        edge_status_counts[status] = edge_status_counts.get(status, 0) + 1
        edge_model_type_counts[model_type] = edge_model_type_counts.get(model_type, 0) + 1
        if status not in {"verified", "verified_planar"} or model_type in {"panoramic", "rejected_wtf"}:
            skipped_edges += 1
            continue
        used_edges += 1
        valid_edges.add(pair_key(image_name1, image_name2))
        matches = data["inlier_matches"].astype(np.int32)
        for keypoint_idx1, keypoint_idx2 in matches:
            obs1 = TrackObservation(image_name1, int(keypoint_idx1))
            obs2 = TrackObservation(image_name2, int(keypoint_idx2))
            if obs1 in blocked_observations or obs2 in blocked_observations:
                skipped_matches_by_residual += 1
                continue
            union_find.union(obs1, obs2)
    return TrackBuildResult(
        tracks=union_find.groups(),
        valid_edges=valid_edges,
        edge_status_counts=edge_status_counts,
        edge_model_type_counts=edge_model_type_counts,
        used_edges=used_edges,
        skipped_edges=skipped_edges,
        skipped_matches_by_residual=skipped_matches_by_residual,
    )


def pair_key(image_name1: str, image_name2: str) -> tuple[str, str]:
    return tuple(sorted((image_name1, image_name2)))


def valid_pair_mask_for_observations(
    observations: list[TrackObservation],
    valid_edges: set[tuple[str, str]],
) -> np.ndarray:
    num_observations = len(observations)
    mask = np.zeros((num_observations, num_observations), dtype=bool)
    for idx1 in range(num_observations):
        for idx2 in range(idx1 + 1, num_observations):
            if pair_key(observations[idx1].image_name, observations[idx2].image_name) in valid_edges:
                mask[idx1, idx2] = True
                mask[idx2, idx1] = True
    return mask


def add_observation_if_missing(
    state: ReconstructionState,
    existing: dict[tuple[str, int], int],
    point3d_id: int,
    observation: TrackObservation,
) -> bool:
    key = (observation.image_name, int(observation.keypoint_idx))
    if key in existing:
        return False
    state.observations.append(
        {
            "point3D_id": int(point3d_id),
            "image_name": observation.image_name,
            "keypoint_idx": int(observation.keypoint_idx),
        }
    )
    existing[key] = int(point3d_id)
    return True


def load_blocked_observations_from_residual_report(
    report_path: Path,
    max_observation_error: float,
) -> tuple[set[TrackObservation], int]:
    if not report_path.exists():
        raise FileNotFoundError(f"Residual report does not exist: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    records = report.get("observation_errors", report.get("optimized_observation_errors", []))
    blocked = set()
    for record in records:
        error = float(record["reprojection_error"] if "reprojection_error" in record else record["final_error"])
        if error <= max_observation_error:
            continue
        blocked.add(
            TrackObservation(
                image_name=str(record["image_name"]),
                keypoint_idx=int(record["keypoint_idx"]),
            )
        )
    return blocked, len(records)


def main() -> int:
    parser = argparse.ArgumentParser(description="Triangulate new 3D points from tracks among registered images.")
    parser.add_argument("--scene", required=True, help="Path to configs/scenes/<scene>.yaml")
    parser.add_argument("--focal-scale", type=float, default=1.2)
    parser.add_argument("--max-pair-samples", type=int, default=100)
    parser.add_argument("--min-track-length", type=int, default=0)
    parser.add_argument("--stage", default="registered_rt", choices=["registered_rt", "pre_ba_rt", "post_ba_rt"])
    parser.add_argument("--residual-report", default="", help="Residual report used to gate post-BA RT observations.")
    parser.add_argument("--max-observation-error", type=float, default=8.0)
    parser.add_argument("--report-name", default="triangulation_report.json")
    args = parser.parse_args()

    config = load_scene_config(args.scene)
    default = config["default"]
    scene = config["scene"]

    image_dir = resolve_project_path(scene["image_dir"])
    feature_dir = resolve_project_path(scene["feature_dir"])
    verified_dir = resolve_project_path(scene["verified_dir"])
    sparse_dir = resolve_project_path(scene["sparse_dir"])
    output_dir = resolve_project_path(scene["output_dir"])
    report_dir = output_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    state = load_reconstruction_state(sparse_dir)
    registered_names = set(state.registered_images)
    if len(registered_names) < 2:
        raise ValueError("At least two registered images are required for Phase 6 triangulation.")

    images = list_images(image_dir)
    image_by_name = {image_path.name: image_path for image_path in images}
    keypoints_by_name = {
        image_name: load_features(feature_dir / f"{Path(image_name).stem}.npz").keypoints
        for image_name in registered_names
    }
    cameras = {}
    for image_name in registered_names:
        feature_data = np.load(feature_dir / f"{Path(image_name).stem}.npz")
        width, height = [int(value) for value in feature_data["image_size"]]
        cameras[image_name] = estimate_simple_pinhole(width, height, focal_scale=args.focal_scale)

    images_rgb: dict[str, np.ndarray] = {}
    projections = {
        image_name: projection_matrix(cameras[image_name].K, registered.R, registered.t)
        for image_name, registered in state.registered_images.items()
    }
    blocked_observations: set[TrackObservation] = set()
    residual_observations_scored = 0
    residual_report_path = ""
    if args.residual_report:
        residual_report = Path(args.residual_report)
        residual_report_path = str(residual_report)
        blocked_observations, residual_observations_scored = load_blocked_observations_from_residual_report(
            residual_report,
            max_observation_error=float(args.max_observation_error),
        )
    elif args.stage == "post_ba_rt":
        raise ValueError("post_ba_rt requires --residual-report for observation gating.")

    track_build = build_registered_tracks(
        verified_dir,
        registered_names,
        blocked_observations=blocked_observations,
    )
    tracks = track_build.tracks
    existing = observation_lookup(state)

    max_reproj_error_px = float(default["sfm"].get("max_reproj_error_px", 8.0))
    min_angle_deg = float(default["sfm"].get("min_triangulation_angle_deg", 1.5))
    min_track_length = int(args.min_track_length or default["sfm"].get("min_track_length", 2))

    initial_points = int(state.points3d.shape[0])
    initial_observations = int(len(state.observations))
    new_points: list[np.ndarray] = []
    new_colors: list[np.ndarray] = []
    new_point_observations: list[list[TrackObservation]] = []
    new_point_errors: list[float] = []
    new_point_angles: list[float] = []
    augmented_observations = 0
    skipped_ambiguous_tracks = 0
    skipped_conflicting_point_tracks = 0
    skipped_short = 0
    skipped_no_valid_view_pair = 0
    skipped_geometry = 0
    skipped_observations_by_residual = 0

    for track in tqdm(tracks, desc=f"Triangulating {scene['scene_name']}"):
        unique_by_image: dict[str, TrackObservation] = {}
        ambiguous = False
        for observation in track:
            if observation.image_name in unique_by_image:
                ambiguous = True
                break
            unique_by_image[observation.image_name] = observation
        if ambiguous:
            skipped_ambiguous_tracks += 1
            continue
        observations = sorted(unique_by_image.values(), key=lambda item: item.image_name)
        if len(observations) < min_track_length:
            skipped_short += 1
            continue

        linked_point_ids = {
            existing[(observation.image_name, observation.keypoint_idx)]
            for observation in observations
            if (observation.image_name, observation.keypoint_idx) in existing
        }
        if len(linked_point_ids) > 1:
            skipped_conflicting_point_tracks += 1
            continue
        if len(linked_point_ids) == 1:
            point3d_id = next(iter(linked_point_ids))
            for observation in observations:
                if observation in blocked_observations:
                    skipped_observations_by_residual += 1
                    continue
                if add_observation_if_missing(state, existing, point3d_id, observation):
                    augmented_observations += 1
            continue

        points2d = np.stack(
            [
                keypoints_by_name[observation.image_name][observation.keypoint_idx, :2].astype(np.float64)
                for observation in observations
            ]
        )
        image_names = [observation.image_name for observation in observations]
        valid_pair_mask = valid_pair_mask_for_observations(observations, track_build.valid_edges)
        if not np.any(np.triu(valid_pair_mask, k=1)):
            skipped_no_valid_view_pair += 1
            continue
        result = robust_triangulate_track(
            projection_matrices=[projections[image_name] for image_name in image_names],
            rotations=[state.registered_images[image_name].R for image_name in image_names],
            translations=[state.registered_images[image_name].t for image_name in image_names],
            intrinsics=[cameras[image_name].K for image_name in image_names],
            points2d=points2d,
            max_reproj_error_px=max_reproj_error_px,
            min_triangulation_angle_deg=min_angle_deg,
            min_track_length=min_track_length,
            max_pair_samples=args.max_pair_samples,
            valid_pair_mask=valid_pair_mask,
        )
        if result is None:
            skipped_geometry += 1
            continue

        inlier_observations = [observations[int(index)] for index in result.inlier_indices]
        color_source = inlier_observations[0]
        if color_source.image_name not in images_rgb:
            images_rgb[color_source.image_name] = read_rgb(image_by_name[color_source.image_name])
        color_point = keypoints_by_name[color_source.image_name][color_source.keypoint_idx, :2]
        new_points.append(result.point3d)
        new_colors.append(sample_color(images_rgb[color_source.image_name], color_point))
        new_point_observations.append(inlier_observations)
        new_point_errors.append(float(np.median(result.reprojection_errors)))
        new_point_angles.append(float(result.triangulation_angle_deg))

    if new_points:
        start_id = int(state.points3d.shape[0])
        state.points3d = np.vstack([state.points3d, np.asarray(new_points, dtype=np.float64)])
        state.colors = np.vstack([state.colors, np.asarray(new_colors, dtype=np.uint8)])
        for offset, observations in enumerate(new_point_observations):
            point3d_id = start_id + offset
            for observation in observations:
                add_observation_if_missing(state, existing, point3d_id, observation)

    state_path, registered_npz_path = save_reconstruction_state(state, sparse_dir)
    report = {
        "scene_name": scene["scene_name"],
        "rt_stage": args.stage,
        "residual_report_path": residual_report_path,
        "residual_observations_scored": int(residual_observations_scored),
        "max_observation_error": float(args.max_observation_error),
        "blocked_observations_by_residual": len(blocked_observations),
        "registered_images": len(state.registered_images),
        "input_edges_used": track_build.used_edges,
        "input_edges_skipped": track_build.skipped_edges,
        "skipped_matches_by_residual": track_build.skipped_matches_by_residual,
        "edge_status_counts": track_build.edge_status_counts,
        "edge_model_type_counts": track_build.edge_model_type_counts,
        "input_tracks": len(tracks),
        "initial_points3D": initial_points,
        "final_points3D": int(state.points3d.shape[0]),
        "new_points3D": int(state.points3d.shape[0] - initial_points),
        "initial_observations": initial_observations,
        "final_observations": int(len(state.observations)),
        "new_observations": int(len(state.observations) - initial_observations),
        "augmented_existing_observations": int(augmented_observations),
        "skipped_short_tracks": int(skipped_short),
        "skipped_ambiguous_tracks": int(skipped_ambiguous_tracks),
        "skipped_conflicting_point_tracks": int(skipped_conflicting_point_tracks),
        "skipped_no_valid_view_pair_tracks": int(skipped_no_valid_view_pair),
        "skipped_geometry_tracks": int(skipped_geometry),
        "skipped_observations_by_residual": int(skipped_observations_by_residual),
        "track_merge_implemented": False,
        "median_new_point_reprojection_error": float(np.median(new_point_errors)) if new_point_errors else 0.0,
        "mean_new_point_reprojection_error": float(np.mean(new_point_errors)) if new_point_errors else 0.0,
        "median_new_point_angle_deg": float(np.median(new_point_angles)) if new_point_angles else 0.0,
        "mean_new_point_angle_deg": float(np.mean(new_point_angles)) if new_point_angles else 0.0,
        "max_reproj_error_px": max_reproj_error_px,
        "min_triangulation_angle_deg": min_angle_deg,
        "min_track_length": min_track_length,
        "state_path": str(state_path),
        "registered_npz_path": str(registered_npz_path),
        "points_npz_path": str(sparse_dir / "reconstruction_points.npz"),
    }
    report_path = report_dir / args.report_name
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Scene: {scene['scene_name']}")
    print(f"RT stage: {report['rt_stage']}")
    print(f"Registered images: {report['registered_images']}")
    print(f"Tracks: {report['input_tracks']}")
    print(f"Points3D: {report['initial_points3D']} -> {report['final_points3D']}")
    print(f"Observations: {report['initial_observations']} -> {report['final_observations']}")
    print(
        "New point reprojection error median/mean: "
        f"{report['median_new_point_reprojection_error']:.3f} / "
        f"{report['mean_new_point_reprojection_error']:.3f}"
    )
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
