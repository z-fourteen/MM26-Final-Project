from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.sfm.bundle_adjustment import error_distribution, per_image_error_summary, squared_residual_cost
from src.sfm.camera import project_points
from src.sfm.reconstruction import ReconstructionState


def evaluate_registered_residuals_in_memory(
    state: ReconstructionState,
    cameras: dict,
    keypoints_by_name: dict[str, np.ndarray],
) -> tuple[list[dict], np.ndarray, dict]:
    valid_observations = []
    errors = []
    residuals = []
    skipped = {
        "unregistered_image": 0,
        "invalid_point_id": 0,
        "missing_keypoints": 0,
        "invalid_keypoint_idx": 0,
        "missing_camera": 0,
    }

    grouped: dict[str, list[dict]] = {}
    for observation in state.observations:
        image_name = str(observation["image_name"])
        point_id = int(observation["point3D_id"])
        keypoint_idx = int(observation["keypoint_idx"])
        if image_name not in state.registered_images:
            skipped["unregistered_image"] += 1
            continue
        if point_id < 0 or point_id >= int(state.points3d.shape[0]):
            skipped["invalid_point_id"] += 1
            continue
        keypoints = keypoints_by_name.get(image_name)
        if keypoints is None:
            skipped["missing_keypoints"] += 1
            continue
        if keypoint_idx < 0 or keypoint_idx >= int(keypoints.shape[0]):
            skipped["invalid_keypoint_idx"] += 1
            continue
        if image_name not in cameras:
            skipped["missing_camera"] += 1
            continue
        grouped.setdefault(image_name, []).append(dict(observation))

    for image_name in sorted(grouped):
        observations = grouped[image_name]
        registered = state.registered_images[image_name]
        camera = cameras[image_name]
        point_ids = np.asarray([int(observation["point3D_id"]) for observation in observations], dtype=np.int32)
        keypoint_indices = np.asarray([int(observation["keypoint_idx"]) for observation in observations], dtype=np.int32)
        points3d = state.points3d[point_ids].astype(np.float64)
        observed = keypoints_by_name[image_name][keypoint_indices, :2].astype(np.float64)
        projected = project_points(camera.K, registered.R, registered.t, points3d)
        batch_residuals = projected - observed
        batch_errors = np.linalg.norm(batch_residuals, axis=1)
        valid_observations.extend(observations)
        residuals.extend(batch_residuals.reshape(-1).astype(float).tolist())
        errors.extend(batch_errors.astype(float).tolist())

    errors_array = np.asarray(errors, dtype=np.float64)
    residuals_array = np.asarray(residuals, dtype=np.float64)
    report = {
        "scope": "registered_observations",
        "num_registered_images": len(state.registered_images),
        "num_points_total": int(state.points3d.shape[0]),
        "num_observations_total": len(state.observations),
        "num_observations_evaluated": len(valid_observations),
        "num_observations_skipped": int(sum(skipped.values())),
        "skipped": skipped,
        "squared_residual_cost": squared_residual_cost(residuals_array),
        "error_summary": error_distribution(errors_array),
        "per_image": per_image_error_summary(valid_observations, errors_array),
    }
    return valid_observations, errors_array, report


def write_residual_report(
    report_path: Path,
    scene_name: str,
    observations: list[dict],
    errors: np.ndarray,
    summary: dict,
    compact: bool = False,
    npz_name: str | None = None,
) -> dict:
    report = {"scene_name": scene_name, **summary}
    if npz_name:
        report["observation_errors_npz"] = npz_name
    if not compact:
        report["observation_errors"] = [
            {
                "point3D_id": int(observation["point3D_id"]),
                "image_name": str(observation["image_name"]),
                "keypoint_idx": int(observation["keypoint_idx"]),
                "reprojection_error": float(error),
            }
            for observation, error in zip(observations, errors)
        ]
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def write_residual_npz(npz_path: Path, observations: list[dict], errors: np.ndarray) -> None:
    image_names = np.asarray([str(observation["image_name"]) for observation in observations])
    point3d_ids = np.asarray([int(observation["point3D_id"]) for observation in observations], dtype=np.int32)
    keypoint_indices = np.asarray([int(observation["keypoint_idx"]) for observation in observations], dtype=np.int32)
    np.savez_compressed(
        npz_path,
        image_names=image_names,
        point3D_ids=point3d_ids,
        keypoint_indices=keypoint_indices,
        reprojection_errors=errors.astype(np.float64),
    )
