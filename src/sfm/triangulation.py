from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from src.sfm.camera import camera_depths, project_points


@dataclass(frozen=True)
class TriangulatedTrack:
    point3d: np.ndarray
    inlier_indices: np.ndarray
    reprojection_errors: np.ndarray
    triangulation_angle_deg: float


def triangulate_point_dlt(P1: np.ndarray, P2: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    u1, v1 = float(x1[0]), float(x1[1])
    u2, v2 = float(x2[0]), float(x2[1])
    A = np.array(
        [
            u1 * P1[2] - P1[0],
            v1 * P1[2] - P1[1],
            u2 * P2[2] - P2[0],
            v2 * P2[2] - P2[1],
        ],
        dtype=np.float64,
    )
    _, _, vh = np.linalg.svd(A)
    homogeneous = vh[-1]
    if abs(homogeneous[3]) < 1e-12:
        return np.full((3,), np.nan, dtype=np.float64)
    return homogeneous[:3] / homogeneous[3]


def triangulate_points_dlt(P1: np.ndarray, P2: np.ndarray, points1: np.ndarray, points2: np.ndarray) -> np.ndarray:
    if points1.shape != points2.shape:
        raise ValueError(f"Point arrays must have the same shape, got {points1.shape} and {points2.shape}")
    points3d = [
        triangulate_point_dlt(P1, P2, x1, x2)
        for x1, x2 in zip(points1, points2)
    ]
    return np.asarray(points3d, dtype=np.float64)


def triangulate_point_nview(projection_matrices: list[np.ndarray], points2d: np.ndarray) -> np.ndarray:
    if len(projection_matrices) != int(points2d.shape[0]):
        raise ValueError("Number of projection matrices must match number of image observations.")
    if len(projection_matrices) < 2:
        raise ValueError("At least two views are required for triangulation.")

    rows = []
    for P, point in zip(projection_matrices, points2d):
        u, v = float(point[0]), float(point[1])
        rows.append(u * P[2] - P[0])
        rows.append(v * P[2] - P[1])
    A = np.asarray(rows, dtype=np.float64)
    _, _, vh = np.linalg.svd(A)
    homogeneous = vh[-1]
    if abs(homogeneous[3]) < 1e-12:
        return np.full((3,), np.nan, dtype=np.float64)
    return homogeneous[:3] / homogeneous[3]


def max_triangulation_angle_deg(camera_centers: np.ndarray, point3d: np.ndarray) -> float:
    if camera_centers.shape[0] < 2 or not np.isfinite(point3d).all():
        return 0.0
    rays = point3d.reshape(1, 3) - camera_centers.astype(np.float64)
    norms = np.linalg.norm(rays, axis=1, keepdims=True)
    valid = norms[:, 0] > 1e-12
    rays = rays[valid] / np.maximum(norms[valid], 1e-12)
    if rays.shape[0] < 2:
        return 0.0
    cosines = np.clip(rays @ rays.T, -1.0, 1.0)
    upper = np.triu_indices(rays.shape[0], k=1)
    angles = np.degrees(np.arccos(cosines[upper]))
    return float(np.max(angles)) if angles.size else 0.0


def robust_triangulate_track(
    projection_matrices: list[np.ndarray],
    rotations: list[np.ndarray],
    translations: list[np.ndarray],
    intrinsics: list[np.ndarray],
    points2d: np.ndarray,
    max_reproj_error_px: float,
    min_triangulation_angle_deg: float,
    min_track_length: int,
    max_pair_samples: int = 100,
    valid_pair_mask: np.ndarray | None = None,
) -> TriangulatedTrack | None:
    num_observations = int(points2d.shape[0])
    if num_observations < max(2, min_track_length):
        return None

    camera_centers = np.stack(
        [
            -R.astype(np.float64).T @ np.asarray(t, dtype=np.float64).reshape(3)
            for R, t in zip(rotations, translations)
        ]
    )

    if valid_pair_mask is not None and valid_pair_mask.shape != (num_observations, num_observations):
        raise ValueError(
            f"valid_pair_mask must have shape {(num_observations, num_observations)}, got {valid_pair_mask.shape}"
        )
    pairs = [
        (idx1, idx2)
        for idx1, idx2 in combinations(range(num_observations), 2)
        if valid_pair_mask is None or bool(valid_pair_mask[idx1, idx2])
    ]
    if not pairs:
        return None
    if len(pairs) > max_pair_samples:
        pairs = pairs[:max_pair_samples]

    best: tuple[np.ndarray, np.ndarray, np.ndarray, float] | None = None
    for idx1, idx2 in pairs:
        candidate = triangulate_point_nview(
            [projection_matrices[idx1], projection_matrices[idx2]],
            points2d[[idx1, idx2]],
        )
        evaluation = _evaluate_candidate(
            candidate,
            rotations,
            translations,
            intrinsics,
            points2d,
            camera_centers,
            max_reproj_error_px=max_reproj_error_px,
            min_triangulation_angle_deg=min_triangulation_angle_deg,
            min_track_length=min_track_length,
        )
        if evaluation is None:
            continue
        inlier_indices, errors, angle = evaluation
        if best is None:
            best = (candidate, inlier_indices, errors, angle)
            continue
        _best_point, best_inliers, best_errors, best_angle = best
        current_score = (len(inlier_indices), -float(np.median(errors[inlier_indices])), angle)
        best_score = (len(best_inliers), -float(np.median(best_errors[best_inliers])), best_angle)
        if current_score > best_score:
            best = (candidate, inlier_indices, errors, angle)

    if best is None:
        return None

    _candidate, inlier_indices, _errors, _angle = best
    refined = triangulate_point_nview(
        [projection_matrices[index] for index in inlier_indices],
        points2d[inlier_indices],
    )
    evaluation = _evaluate_candidate(
        refined,
        rotations,
        translations,
        intrinsics,
        points2d,
        camera_centers,
        max_reproj_error_px=max_reproj_error_px,
        min_triangulation_angle_deg=min_triangulation_angle_deg,
        min_track_length=min_track_length,
    )
    if evaluation is None:
        return None
    refined_inliers, refined_errors, refined_angle = evaluation
    return TriangulatedTrack(
        point3d=refined.astype(np.float64),
        inlier_indices=refined_inliers.astype(np.int32),
        reprojection_errors=refined_errors[refined_inliers].astype(np.float64),
        triangulation_angle_deg=float(refined_angle),
    )


def robust_triangulate_track_recursive(
    projection_matrices: list[np.ndarray],
    rotations: list[np.ndarray],
    translations: list[np.ndarray],
    intrinsics: list[np.ndarray],
    points2d: np.ndarray,
    max_reproj_error_px: float,
    min_triangulation_angle_deg: float,
    min_track_length: int,
    max_pair_samples: int = 100,
    valid_pair_mask: np.ndarray | None = None,
    min_consensus_size: int = 3,
) -> list[TriangulatedTrack]:
    """Recover multiple independent points from one potentially merged track."""
    num_observations = int(points2d.shape[0])
    effective_min_track_length = max(int(min_track_length), int(min_consensus_size))
    if num_observations < effective_min_track_length:
        return []

    remaining_indices = np.arange(num_observations, dtype=np.int32)
    recovered: list[TriangulatedTrack] = []

    while remaining_indices.shape[0] >= effective_min_track_length:
        subset_mask = None
        if valid_pair_mask is not None:
            subset_mask = valid_pair_mask[np.ix_(remaining_indices, remaining_indices)]
        result = robust_triangulate_track(
            projection_matrices=[projection_matrices[int(index)] for index in remaining_indices],
            rotations=[rotations[int(index)] for index in remaining_indices],
            translations=[translations[int(index)] for index in remaining_indices],
            intrinsics=[intrinsics[int(index)] for index in remaining_indices],
            points2d=points2d[remaining_indices],
            max_reproj_error_px=max_reproj_error_px,
            min_triangulation_angle_deg=min_triangulation_angle_deg,
            min_track_length=effective_min_track_length,
            max_pair_samples=max_pair_samples,
            valid_pair_mask=subset_mask,
        )
        if result is None or result.inlier_indices.shape[0] < effective_min_track_length:
            break

        original_inliers = remaining_indices[result.inlier_indices].astype(np.int32)
        recovered.append(
            TriangulatedTrack(
                point3d=result.point3d.astype(np.float64),
                inlier_indices=original_inliers,
                reprojection_errors=result.reprojection_errors.astype(np.float64),
                triangulation_angle_deg=float(result.triangulation_angle_deg),
            )
        )
        keep_mask = np.ones(remaining_indices.shape[0], dtype=bool)
        keep_mask[result.inlier_indices] = False
        remaining_indices = remaining_indices[keep_mask]

    return recovered


def _evaluate_candidate(
    point3d: np.ndarray,
    rotations: list[np.ndarray],
    translations: list[np.ndarray],
    intrinsics: list[np.ndarray],
    points2d: np.ndarray,
    camera_centers: np.ndarray,
    max_reproj_error_px: float,
    min_triangulation_angle_deg: float,
    min_track_length: int,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    if not np.isfinite(point3d).all():
        return None
    depths = np.asarray(
        [
            camera_depths(R, t, point3d.reshape(1, 3))[0]
            for R, t in zip(rotations, translations)
        ],
        dtype=np.float64,
    )
    projected = np.stack(
        [
            project_points(K, R, t, point3d.reshape(1, 3))[0]
            for K, R, t in zip(intrinsics, rotations, translations)
        ]
    )
    errors = np.linalg.norm(projected - points2d, axis=1)
    inlier_mask = (depths > 0.0) & np.isfinite(errors) & (errors <= max_reproj_error_px)
    inlier_indices = np.flatnonzero(inlier_mask).astype(np.int32)
    if inlier_indices.shape[0] < min_track_length:
        return None
    angle = max_triangulation_angle_deg(camera_centers[inlier_indices], point3d)
    if angle < min_triangulation_angle_deg:
        return None
    return inlier_indices, errors, angle
