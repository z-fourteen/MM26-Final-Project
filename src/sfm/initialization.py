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


def select_initial_pair(edges: list[dict]) -> dict:
    if not edges:
        raise ValueError("Scene graph has no verified edges.")
    return max(edges, key=initial_pair_score)


def initial_pair_score(edge: dict) -> tuple[float, float, float, float, float]:
    status = str(edge.get("status", "verified"))
    model_type = str(edge.get("model_type", "general"))
    calibration_status = str(edge.get("calibration_status", "uncertain"))
    num_inliers = float(edge.get("num_inliers", 0))
    inlier_ratio = float(edge.get("inlier_ratio", 0.0))
    homography_ratio = float(edge.get("homography_ratio", 0.0))
    triangulation_angle = float(edge.get("median_triangulation_angle_deg", 0.0))

    if status == "rejected_wtf" or model_type == "rejected_wtf":
        model_priority = -2.0
    elif model_type == "panoramic":
        model_priority = -1.0
    elif model_type == "general":
        model_priority = 2.0
    elif model_type == "planar":
        model_priority = 0.0
    else:
        model_priority = 1.0 if status == "verified" else -1.0

    calibration_priority = 1.0 if calibration_status == "calibrated" else 0.0
    geometric_support = num_inliers * inlier_ratio
    return (
        model_priority,
        calibration_priority,
        geometric_support,
        triangulation_angle,
        -homography_ratio,
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
    points1 = keypoints1[inlier_matches[:, 0], :2].astype(np.float64)
    points2 = keypoints2[inlier_matches[:, 1], :2].astype(np.float64)

    R1 = np.eye(3, dtype=np.float64)
    t1 = np.zeros(3, dtype=np.float64)
    R2, t2, _pose_inliers = estimate_relative_pose_from_fundamental(
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

    return InitializationResult(
        image_name1=str(data["image_name1"]),
        image_name2=str(data["image_name2"]),
        verified_path=verified_path,
        score=float(edge["num_inliers"]) * float(edge["inlier_ratio"]),
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
    )


def sample_colors(image_rgb: np.ndarray, points: np.ndarray) -> np.ndarray:
    height, width = image_rgb.shape[:2]
    rounded = np.rint(points).astype(np.int32)
    xs = np.clip(rounded[:, 0], 0, width - 1)
    ys = np.clip(rounded[:, 1], 0, height - 1)
    return image_rgb[ys, xs].astype(np.uint8)


def save_initialization(result: InitializationResult, sparse_dir: Path) -> tuple[Path, Path]:
    sparse_dir.mkdir(parents=True, exist_ok=True)
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
