from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from src.sfm.config import load_scene_config, resolve_project_path
from src.sfm.filtering import filter_reconstruction_by_ba_errors
from src.sfm.reconstruction import load_reconstruction_state, save_reconstruction_state


def main() -> int:
    parser = argparse.ArgumentParser(description="Filter reconstruction observations and points using BA residuals.")
    parser.add_argument("--scene", required=True, help="Path to configs/scenes/<scene>.yaml")
    parser.add_argument("--ba-report", default="", help="Path to bundle_adjustment_report.json")
    parser.add_argument("--max-reprojection-error", type=float, default=8.0)
    parser.add_argument("--max-point-median-error", type=float, default=8.0)
    parser.add_argument("--max-point-max-error", type=float, default=32.0)
    parser.add_argument("--min-track-length", type=int, default=2)
    parser.add_argument("--report-name", default="filtering_report.json")
    parser.add_argument("--dry-run", action="store_true", help="Write report only without changing reconstruction state.")
    args = parser.parse_args()

    config = load_scene_config(args.scene)
    scene = config["scene"]
    sparse_dir = resolve_project_path(scene["sparse_dir"])
    output_dir = resolve_project_path(scene["output_dir"])
    report_dir = output_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    ba_report_path = Path(args.ba_report) if args.ba_report else report_dir / "bundle_adjustment_report.json"
    ba_report = json.loads(ba_report_path.read_text(encoding="utf-8"))
    ba_observations, ba_errors = load_ba_observation_errors(ba_report)

    state = load_reconstruction_state(sparse_dir)
    result = filter_reconstruction_by_ba_errors(
        state=state,
        ba_observations=ba_observations,
        ba_errors=ba_errors,
        max_observation_error=args.max_reprojection_error,
        max_point_median_error=args.max_point_median_error,
        max_point_max_error=args.max_point_max_error,
        min_track_length=args.min_track_length,
    )

    backup_paths = {}
    if not args.dry_run:
        backup_paths = backup_reconstruction_state(sparse_dir)
        state_path, registered_npz_path = save_reconstruction_state(result.filtered_state, sparse_dir)
    else:
        state_path = sparse_dir / "reconstruction_state.json"
        registered_npz_path = sparse_dir / "registered_images.npz"

    report = {
        "scene_name": scene["scene_name"],
        "dry_run": bool(args.dry_run),
        "ba_report_path": str(ba_report_path),
        "state_path": str(state_path),
        "registered_npz_path": str(registered_npz_path),
        "backup_paths": backup_paths,
        **result.report,
    }
    report_path = report_dir / args.report_name
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Scene: {scene['scene_name']}")
    print(f"Observations: {report['observations_before']} -> {report['observations_after']}")
    print(f"Points3D: {report['points_before']} -> {report['points_after']}")
    print(f"Removed observations by reprojection: {report['removed_observations_by_reprojection']}")
    print(f"Removed points total: {report['removed_points_total']}")
    print(f"Report: {report_path}")
    return 0


def load_ba_observation_errors(ba_report: dict) -> tuple[list[dict], np.ndarray]:
    records = ba_report.get("optimized_observation_errors", [])
    if not records:
        raise ValueError("BA report does not contain optimized_observation_errors.")
    observations = [
        {
            "point3D_id": int(record["point3D_id"]),
            "image_name": str(record["image_name"]),
            "keypoint_idx": int(record["keypoint_idx"]),
        }
        for record in records
    ]
    errors = np.asarray([float(record["final_error"]) for record in records], dtype=np.float64)
    return observations, errors


def backup_reconstruction_state(sparse_dir: Path) -> dict[str, str]:
    backup_paths = {}
    backup_map = {
        "reconstruction_state.json": "reconstruction_state_before_filtering.json",
        "reconstruction_points.npz": "reconstruction_points_before_filtering.npz",
        "registered_images.npz": "registered_images_before_filtering.npz",
    }
    for source_name, backup_name in backup_map.items():
        source = sparse_dir / source_name
        if not source.exists():
            continue
        backup = unique_backup_path(sparse_dir / backup_name)
        shutil.copy2(source, backup)
        backup_paths[source_name] = str(backup)
    return backup_paths


def unique_backup_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    for index in range(1, 1000):
        candidate = path.with_name(f"{stem}_{index}{suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not create a unique backup path for {path}")


if __name__ == "__main__":
    raise SystemExit(main())
