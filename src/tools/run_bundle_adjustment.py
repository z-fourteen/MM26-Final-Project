from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from src.sfm.bundle_adjustment import per_image_error_summary, run_bundle_adjustment
from src.sfm.camera import estimate_simple_pinhole
from src.sfm.config import load_scene_config, resolve_project_path
from src.sfm.features import list_images
from src.sfm.matching import load_features
from src.sfm.reconstruction import load_reconstruction_state, save_reconstruction_state


def main() -> int:
    parser = argparse.ArgumentParser(description="Run small-scale bundle adjustment over the current reconstruction.")
    parser.add_argument("--scene", required=True, help="Path to configs/scenes/<scene>.yaml")
    parser.add_argument("--max-iterations", type=int, default=50)
    parser.add_argument("--loss", default="cauchy", choices=["linear", "soft_l1", "huber", "cauchy", "arctan"])
    parser.add_argument("--f-scale", type=float, default=4.0)
    parser.add_argument("--max-points", type=int, default=2000)
    parser.add_argument("--max-observations", type=int, default=8000)
    parser.add_argument("--min-track-length", type=int, default=2)
    parser.add_argument("--focal-scale", type=float, default=1.2)
    parser.add_argument("--dry-run", action="store_true", help="Run BA and report metrics without writing optimized state.")
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

    fixed_image_name = next(iter(state.registered_images))
    result = run_bundle_adjustment(
        state=state,
        cameras=cameras,
        keypoints_by_name=keypoints_by_name,
        fixed_image_name=fixed_image_name,
        max_iterations=args.max_iterations,
        loss=args.loss,
        f_scale=args.f_scale,
        max_points=args.max_points,
        max_observations=args.max_observations,
        min_track_length=args.min_track_length,
    )

    backup_paths = {}
    if not args.dry_run:
        backup_paths = backup_reconstruction_state(sparse_dir)
        state_path, registered_npz_path = save_reconstruction_state(result.optimized_state, sparse_dir)
    else:
        state_path = sparse_dir / "reconstruction_state.json"
        registered_npz_path = sparse_dir / "registered_images.npz"

    initial_errors = result.initial_errors
    final_errors = result.final_errors
    report = {
        "scene_name": scene["scene_name"],
        "dry_run": bool(args.dry_run),
        "fixed_image_name": fixed_image_name,
        "num_registered_images": len(result.problem.image_names),
        "num_optimized_cameras": len(result.problem.optimizable_image_names),
        "num_points_total": int(state.points3d.shape[0]),
        "num_points_optimized": int(result.problem.point_ids.shape[0]),
        "num_observations_total": len(state.observations),
        "num_observations_optimized": len(result.problem.observations),
        "loss": args.loss,
        "f_scale": args.f_scale,
        "max_iterations": args.max_iterations,
        "max_points": args.max_points,
        "max_observations": args.max_observations,
        "min_track_length": args.min_track_length,
        "optimizer_success": result.success,
        "optimizer_message": result.message,
        "num_function_evaluations": result.num_function_evaluations,
        "initial_squared_residual_cost": result.optimizer_cost_initial,
        "final_squared_residual_cost": result.optimizer_cost_final,
        "final_robust_cost": result.optimizer_robust_cost_final,
        "initial_median_reprojection_error": float(np.median(initial_errors)) if initial_errors.size else 0.0,
        "final_median_reprojection_error": float(np.median(final_errors)) if final_errors.size else 0.0,
        "initial_mean_reprojection_error": float(np.mean(initial_errors)) if initial_errors.size else 0.0,
        "final_mean_reprojection_error": float(np.mean(final_errors)) if final_errors.size else 0.0,
        "initial_max_reprojection_error": float(np.max(initial_errors)) if initial_errors.size else 0.0,
        "final_max_reprojection_error": float(np.max(final_errors)) if final_errors.size else 0.0,
        "initial_observations_above_8px": int(np.sum(initial_errors > 8.0)),
        "final_observations_above_8px": int(np.sum(final_errors > 8.0)),
        "initial_observations_above_16px": int(np.sum(initial_errors > 16.0)),
        "final_observations_above_16px": int(np.sum(final_errors > 16.0)),
        "per_image_before": per_image_error_summary(result.problem.observations, initial_errors),
        "per_image_after": per_image_error_summary(result.problem.observations, final_errors),
        "state_path": str(state_path),
        "registered_npz_path": str(registered_npz_path),
        "backup_paths": backup_paths,
    }
    report_path = report_dir / "bundle_adjustment_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Scene: {scene['scene_name']}")
    print(f"Images optimized/fixed: {report['num_optimized_cameras']} / 1")
    print(f"Points optimized: {report['num_points_optimized']} / {report['num_points_total']}")
    print(f"Observations optimized: {report['num_observations_optimized']} / {report['num_observations_total']}")
    print(
        "Median reprojection error before/after: "
        f"{report['initial_median_reprojection_error']:.3f} / {report['final_median_reprojection_error']:.3f}"
    )
    print(
        "Mean reprojection error before/after: "
        f"{report['initial_mean_reprojection_error']:.3f} / {report['final_mean_reprojection_error']:.3f}"
    )
    print(f"Report: {report_path}")
    return 0


def backup_reconstruction_state(sparse_dir: Path) -> dict[str, str]:
    backup_paths = {}
    backup_map = {
        "reconstruction_state.json": "reconstruction_state_before_ba.json",
        "reconstruction_points.npz": "reconstruction_points_before_ba.npz",
        "registered_images.npz": "registered_images_before_ba.npz",
    }
    for source_name, backup_name in backup_map.items():
        source = sparse_dir / source_name
        if not source.exists():
            continue
        backup = sparse_dir / backup_name
        shutil.copy2(source, backup)
        backup_paths[source_name] = str(backup)
    return backup_paths


if __name__ == "__main__":
    raise SystemExit(main())
