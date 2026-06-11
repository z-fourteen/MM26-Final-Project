from __future__ import annotations

import argparse
import json

import numpy as np

from src.sfm.bundle_adjustment import error_distribution, per_image_error_summary, squared_residual_cost
from src.sfm.camera import estimate_simple_pinhole, project_points
from src.sfm.config import load_scene_config, resolve_project_path
from src.sfm.features import list_images
from src.sfm.matching import load_features
from src.sfm.reconstruction import load_reconstruction_state


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate reprojection residuals for observations in the current registered reconstruction state."
    )
    parser.add_argument("--scene", required=True, help="Path to configs/scenes/<scene>.yaml")
    parser.add_argument("--focal-scale", type=float, default=1.2)
    parser.add_argument("--report-name", default="registered_residual_report.json")
    args = parser.parse_args()

    config = load_scene_config(args.scene)
    scene = config["scene"]

    image_dir = resolve_project_path(scene["image_dir"])
    feature_dir = resolve_project_path(scene["feature_dir"])
    sparse_dir = resolve_project_path(scene["sparse_dir"])
    output_dir = resolve_project_path(scene["output_dir"])
    report_dir = output_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    state = load_reconstruction_state(sparse_dir)
    images = list_images(image_dir)
    features_by_name = {
        image_path.name: load_features(feature_dir / f"{image_path.stem}.npz")
        for image_path in images
    }
    keypoints_by_name = {image_name: features.keypoints for image_name, features in features_by_name.items()}

    cameras = {}
    for image_path in images:
        feature_data = np.load(feature_dir / f"{image_path.stem}.npz")
        width, height = [int(value) for value in feature_data["image_size"]]
        cameras[image_path.name] = estimate_simple_pinhole(width, height, focal_scale=args.focal_scale)

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

    for observation in state.observations:
        image_name = str(observation["image_name"])
        point_id = int(observation["point3D_id"])
        keypoint_idx = int(observation["keypoint_idx"])
        registered = state.registered_images.get(image_name)
        if registered is None:
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
        camera = cameras.get(image_name)
        if camera is None:
            skipped["missing_camera"] += 1
            continue

        point3d = state.points3d[point_id].reshape(1, 3)
        observed = keypoints[keypoint_idx, :2].astype(np.float64)
        projected = project_points(camera.K, registered.R, registered.t, point3d)[0]
        residual = projected - observed
        valid_observations.append(dict(observation))
        residuals.extend([float(residual[0]), float(residual[1])])
        errors.append(float(np.linalg.norm(residual)))

    errors_array = np.asarray(errors, dtype=np.float64)
    residuals_array = np.asarray(residuals, dtype=np.float64)
    report = {
        "scene_name": scene["scene_name"],
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
        "observation_errors": [
            {
                "point3D_id": int(observation["point3D_id"]),
                "image_name": str(observation["image_name"]),
                "keypoint_idx": int(observation["keypoint_idx"]),
                "reprojection_error": float(error),
            }
            for observation, error in zip(valid_observations, errors_array)
        ],
    }

    report_path = report_dir / args.report_name
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    summary = report["error_summary"]
    print(f"Scene: {scene['scene_name']}")
    print(f"Scope: registered_observations")
    print(f"Observations evaluated: {report['num_observations_evaluated']} / {report['num_observations_total']}")
    print(f"Median reprojection error: {summary['median']:.3f} px")
    print(f"Mean reprojection error: {summary['mean']:.3f} px")
    print(f"P95 reprojection error: {summary['p95']:.3f} px")
    print(f"Observations > 8 px: {summary['observations_above_8px']}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
