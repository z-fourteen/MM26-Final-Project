from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.sfm.camera import (
    PinholeCamera,
    camera_depths,
    project_points,
    projection_matrix,
    reprojection_errors,
)
from src.sfm.triangulation import triangulate_points_dlt


@dataclass(frozen=True)
class InitializationResult:
    image_name1: str
    image_name2: str
    verified_path: Path
    score: float
    camera1: PinholeCamera
    camera2: PinholeCamera
    R1: np.ndarray
    t1: np.ndarray
    R2: np.ndarray
    t2: np.ndarray
    inlier_matches: np.ndarray
    points3d: np.ndarray
    colors: np.ndarray
    reprojection_errors: np.ndarray
    kept_mask: np.ndarray
    num_candidate_points: int
    num_kept_points: int
    pose_inliers: int
    kept_ratio: float
    cheirality_ratio: float
    median_triangulation_angle_deg: float


def select_initial_pair(edges: list[dict]) -> dict:
    if not edges:
        raise ValueError("Scene graph has no verified edges.")
    candidates = rank_initial_pair_candidates(edges)
    if not candidates:
        raise ValueError("Scene graph has no valid initialization candidates.")
    return candidates[0]


def rank_initial_pair_candidates(edges: list[dict]) -> list[dict]:
    degrees = scene_graph_degrees(edges)
    candidates = []
    for edge in edges:
        if not is_initialization_candidate(edge):
            continue
        candidate = dict(edge)
        candidate["graph_degree_score"] = graph_degree_score(edge, degrees)
        candidate["initialization_score"] = initial_pair_score(candidate)
        candidates.append(candidate)
    return sorted(candidates, key=lambda edge: float(edge["initialization_score"]), reverse=True)


def scene_graph_degrees(edges: list[dict]) -> dict[str, int]:
    degrees: dict[str, int] = {}
    for edge in edges:
        if str(edge["status"]) not in {"verified", "verified_planar", "verified_panoramic"}:
            continue
        image_name1 = str(edge["image_name1"])
        image_name2 = str(edge["image_name2"])
        degrees[image_name1] = degrees.get(image_name1, 0) + 1
        degrees[image_name2] = degrees.get(image_name2, 0) + 1
    return degrees


def graph_degree_score(edge: dict, degrees: dict[str, int]) -> float:
    degree1 = float(degrees.get(str(edge["image_name1"]), 0))
    degree2 = float(degrees.get(str(edge["image_name2"]), 0))
    return float(np.sqrt(max(degree1, 0.0) * max(degree2, 0.0)))


def initial_pair_score(edge: dict) -> float:
    status = str(edge["status"])
    model_type = str(edge["model_type"])
    calibration_status = str(edge["calibration_status"])
    num_inliers = float(edge["num_inliers"])
    inlier_ratio = float(edge["inlier_ratio"])
    homography_ratio = float(edge["homography_ratio"])
    essential_ratio = float(edge["essential_ratio"])
    triangulation_angle = float(edge["median_triangulation_angle_deg"])
    graph_score = float(edge["graph_degree_score"])

    status_bonus = 2.0 if status == "verified" else 0.5
    model_bonus = 2.0 if model_type == "general" else 0.5
    calibration_bonus = 1.0 if calibration_status == "calibrated" else 0.0
    support_score = np.log1p(max(num_inliers, 0.0)) * max(inlier_ratio, 0.0)
    angle_score = min(max(triangulation_angle, 0.0), 30.0) / 30.0
    essential_score = min(max(essential_ratio, 0.0), 1.0)
    graph_score = np.log1p(max(graph_score, 0.0))
    homography_penalty = min(max(homography_ratio, 0.0), 2.0)
    return float(
        2.0 * status_bonus
        + 2.0 * model_bonus
        + 1.5 * calibration_bonus
        + 2.0 * support_score
        + 1.0 * angle_score
        + 1.0 * essential_score
        + 0.5 * graph_score
        - 2.0 * homography_penalty
    )


def is_initialization_candidate(edge: dict) -> bool:
    return (
        str(edge["status"]) in {"verified", "verified_planar"}
        and str(edge["model_type"]) in {"general", "planar"}
        and str(edge["calibration_status"]) in {"calibrated", "uncertain"}
    )


def estimate_relative_pose_from_fundamental(
    F: np.ndarray,
    K1: np.ndarray,
    K2: np.ndarray,
    points1: np.ndarray,
    points2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    E = K2.T @ F @ K1
    inliers, R, t, _mask = cv2.recoverPose(E, points1, points2, K1)
    return R.astype(np.float64), t.reshape(3).astype(np.float64), int(inliers)


def initialize_two_view(
    edge: dict,
    verified_path: Path,
    keypoints1: np.ndarray,
    keypoints2: np.ndarray,
    camera1: PinholeCamera,
    camera2: PinholeCamera,
    image1_rgb: np.ndarray,
    max_reproj_error_px: float,
) -> InitializationResult:
    data = np.load(verified_path)
    F = data["F"].astype(np.float64)
    inlier_matches = data["inlier_matches"].astype(np.int32)
    validate_inlier_matches(
        inlier_matches=inlier_matches,
        keypoints1=keypoints1,
        keypoints2=keypoints2,
        verified_path=verified_path,
    )
    points1 = keypoints1[inlier_matches[:, 0], :2].astype(np.float64)
    points2 = keypoints2[inlier_matches[:, 1], :2].astype(np.float64)

    R1 = np.eye(3, dtype=np.float64)
    t1 = np.zeros(3, dtype=np.float64)
    R2, t2, pose_inliers = estimate_relative_pose_from_fundamental(
        F=F,
        K1=camera1.K,
        K2=camera2.K,
        points1=points1,
        points2=points2,
    )

    P1 = projection_matrix(camera1.K, R1, t1)
    P2 = projection_matrix(camera2.K, R2, t2)
    points3d = triangulate_points_dlt(P1, P2, points1, points2)

    finite_mask = np.isfinite(points3d).all(axis=1)
    depth1 = camera_depths(R1, t1, points3d)
    depth2 = camera_depths(R2, t2, points3d)
    positive_depth_mask = (depth1 > 0.0) & (depth2 > 0.0)
    errors1 = reprojection_errors(camera1.K, R1, t1, points3d, points1)
    errors2 = reprojection_errors(camera2.K, R2, t2, points3d, points2)
    errors = 0.5 * (errors1 + errors2)
    reproj_mask = errors < max_reproj_error_px
    kept_mask = finite_mask & positive_depth_mask & reproj_mask

    kept_points = points3d[kept_mask]
    kept_errors = errors[kept_mask]
    kept_matches = inlier_matches[kept_mask]
    colors = sample_colors(image1_rgb, points1[kept_mask])
    cheirality_ratio = float(np.mean(positive_depth_mask)) if positive_depth_mask.size else 0.0
    kept_ratio = float(kept_points.shape[0] / max(points3d.shape[0], 1))
    median_angle = median_triangulation_angle(R1, t1, R2, t2, kept_points)

    return InitializationResult(
        image_name1=str(data["image_name1"]),
        image_name2=str(data["image_name2"]),
        verified_path=verified_path,
        score=float(edge["initialization_score"]),
        camera1=camera1,
        camera2=camera2,
        R1=R1,
        t1=t1,
        R2=R2,
        t2=t2,
        inlier_matches=kept_matches,
        points3d=kept_points,
        colors=colors,
        reprojection_errors=kept_errors,
        kept_mask=kept_mask,
        num_candidate_points=int(points3d.shape[0]),
        num_kept_points=int(kept_points.shape[0]),
        pose_inliers=pose_inliers,
        kept_ratio=kept_ratio,
        cheirality_ratio=cheirality_ratio,
        median_triangulation_angle_deg=median_angle,
    )


def validate_inlier_matches(
    *,
    inlier_matches: np.ndarray,
    keypoints1: np.ndarray,
    keypoints2: np.ndarray,
    verified_path: Path,
) -> None:
    if inlier_matches.size == 0:
        return
    max_idx1 = int(np.max(inlier_matches[:, 0]))
    max_idx2 = int(np.max(inlier_matches[:, 1]))
    min_idx = int(np.min(inlier_matches))
    if min_idx < 0 or max_idx1 >= int(keypoints1.shape[0]) or max_idx2 >= int(keypoints2.shape[0]):
        raise ValueError(
            "Verified matches are incompatible with current feature files. "
            f"Recompute matches and verification with --force. Path: {verified_path}; "
            f"match index ranges=({min_idx}, {max_idx1}, {max_idx2}); "
            f"keypoint counts=({keypoints1.shape[0]}, {keypoints2.shape[0]})."
        )


def initialization_succeeds(
    result: InitializationResult,
    min_initial_points: int,
    min_initial_kept_ratio: float,
    max_initial_median_reproj_error_px: float,
) -> bool:
    if result.num_kept_points < min_initial_points:
        return False
    if result.kept_ratio < min_initial_kept_ratio:
        return False
    if result.reprojection_errors.size == 0:
        return False
    return float(np.median(result.reprojection_errors)) <= max_initial_median_reproj_error_px


def median_triangulation_angle(
    R1: np.ndarray,
    t1: np.ndarray,
    R2: np.ndarray,
    t2: np.ndarray,
    points3d: np.ndarray,
) -> float:
    if points3d.shape[0] == 0:
        return 0.0
    center1 = -R1.T @ t1.reshape(3)
    center2 = -R2.T @ t2.reshape(3)
    rays1 = points3d - center1.reshape(1, 3)
    rays2 = points3d - center2.reshape(1, 3)
    norms1 = np.linalg.norm(rays1, axis=1)
    norms2 = np.linalg.norm(rays2, axis=1)
    valid = (norms1 > 1e-12) & (norms2 > 1e-12)
    if not np.any(valid):
        return 0.0
    cosines = np.sum(rays1[valid] * rays2[valid], axis=1) / (norms1[valid] * norms2[valid])
    angles = np.degrees(np.arccos(np.clip(cosines, -1.0, 1.0)))
    return float(np.median(angles)) if angles.size else 0.0


def sample_colors(image_rgb: np.ndarray, points: np.ndarray) -> np.ndarray:
    height, width = image_rgb.shape[:2]
    rounded = np.rint(points).astype(np.int32)
    xs = np.clip(rounded[:, 0], 0, width - 1)
    ys = np.clip(rounded[:, 1], 0, height - 1)
    return image_rgb[ys, xs].astype(np.uint8)


def save_initialization(result: InitializationResult, sparse_dir: Path) -> tuple[Path, Path]:
    sparse_dir.mkdir(parents=True, exist_ok=True)
    reset_derived_reconstruction_state(sparse_dir)
    pair_path = sparse_dir / "initial_pair.npz"
    points_path = sparse_dir / "initial_points.npz"

    np.savez_compressed(
        pair_path,
        image_name1=result.image_name1,
        image_name2=result.image_name2,
        verified_path=str(result.verified_path),
        score=np.float64(result.score),
        K1=result.camera1.K,
        K2=result.camera2.K,
        R1=result.R1,
        t1=result.t1,
        R2=result.R2,
        t2=result.t2,
        inlier_matches=result.inlier_matches,
        kept_mask=result.kept_mask,
    )
    track_image_names = np.array([[result.image_name1, result.image_name2]] * result.num_kept_points)
    np.savez_compressed(
        points_path,
        points3D=result.points3d,
        colors=result.colors,
        track_image_names=track_image_names,
        track_keypoint_indices=result.inlier_matches,
        reprojection_errors=result.reprojection_errors,
    )
    return pair_path, points_path


def reset_derived_reconstruction_state(sparse_dir: Path) -> None:
    for filename in ("reconstruction_state.json", "registered_images.npz", "reconstruction_points.npz"):
        path = sparse_dir / filename
        if path.exists():
            path.unlink()
