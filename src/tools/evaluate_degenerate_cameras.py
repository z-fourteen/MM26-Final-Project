from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.sfm.bundle_adjustment import build_covisibility_graph
from src.sfm.config import load_scene_config, resolve_project_path
from src.sfm.reconstruction import load_reconstruction_state


def main() -> int:
    parser = argparse.ArgumentParser(description="Report registered cameras that look degenerate after BA/filtering.")
    parser.add_argument("--scene", required=True, help="Path to configs/scenes/<scene>.yaml")
    parser.add_argument("--residual-report", default="", help="Residual or BA report with per-observation errors.")
    parser.add_argument("--min-observations", type=int, default=20)
    parser.add_argument("--max-median-error", type=float, default=8.0)
    parser.add_argument("--max-p95-error", type=float, default=16.0)
    parser.add_argument("--max-above-threshold-ratio", type=float, default=0.25)
    parser.add_argument("--error-threshold", type=float, default=8.0)
    parser.add_argument("--min-covisibility-degree", type=int, default=1)
    parser.add_argument("--center-outlier-mad-scale", type=float, default=8.0)
    parser.add_argument("--report-name", default="degenerate_camera_report.json")
    args = parser.parse_args()

    config = load_scene_config(args.scene)
    scene = config["scene"]
    sparse_dir = resolve_project_path(scene["sparse_dir"])
    output_dir = resolve_project_path(scene["output_dir"])
    report_dir = output_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    state = load_reconstruction_state(sparse_dir)
    residual_records = []
    residual_report_path = ""
    if args.residual_report:
        residual_report_path = str(Path(args.residual_report))
        residual_records = load_residual_records(Path(args.residual_report))

    records_by_image: dict[str, list[float]] = {image_name: [] for image_name in state.registered_images}
    for record in residual_records:
        image_name = str(record["image_name"])
        if image_name not in records_by_image:
            continue
        records_by_image[image_name].append(float(record["error"]))

    observation_counts = {image_name: 0 for image_name in state.registered_images}
    for observation in state.observations:
        image_name = str(observation["image_name"])
        if image_name in observation_counts:
            observation_counts[image_name] += 1

    covisibility_degree = {image_name: 0 for image_name in state.registered_images}
    for image_name1, image_name2 in build_covisibility_graph(state):
        covisibility_degree[image_name1] = covisibility_degree.get(image_name1, 0) + 1
        covisibility_degree[image_name2] = covisibility_degree.get(image_name2, 0) + 1

    center_distances = camera_center_distances(state)
    per_camera = []
    for image_name in state.registered_images:
        errors = np.asarray(records_by_image.get(image_name, []), dtype=np.float64)
        reasons = []
        num_observations = int(observation_counts.get(image_name, 0))
        if num_observations < args.min_observations:
            reasons.append("few_observations")
        degree = int(covisibility_degree.get(image_name, 0))
        if degree < args.min_covisibility_degree:
            reasons.append("low_covisibility_degree")
        if errors.size:
            median_error = float(np.median(errors))
            p95_error = float(np.percentile(errors, 95))
            max_error = float(np.max(errors))
            above_ratio = float(np.mean(errors > args.error_threshold))
            if median_error > args.max_median_error:
                reasons.append("high_median_error")
            if p95_error > args.max_p95_error:
                reasons.append("high_p95_error")
            if above_ratio > args.max_above_threshold_ratio:
                reasons.append("many_high_error_observations")
        else:
            median_error = 0.0
            p95_error = 0.0
            max_error = 0.0
            above_ratio = 0.0
            if residual_records:
                reasons.append("no_residual_records")
        center_distance = float(center_distances.get(image_name, 0.0))
        if center_distance > args.center_outlier_mad_scale:
            reasons.append("center_outlier")
        per_camera.append(
            {
                "image_name": image_name,
                "num_observations": num_observations,
                "num_residual_records": int(errors.size),
                "median_error": median_error,
                "p95_error": p95_error,
                "max_error": max_error,
                "above_error_threshold_ratio": above_ratio,
                "covisibility_degree": degree,
                "center_robust_distance": center_distance,
                "is_degenerate_candidate": bool(reasons),
                "reasons": reasons,
            }
        )

    per_camera.sort(key=lambda item: (not item["is_degenerate_candidate"], item["image_name"]))
    candidates = [record for record in per_camera if record["is_degenerate_candidate"]]
    report = {
        "scene_name": scene["scene_name"],
        "residual_report_path": residual_report_path,
        "num_registered_images": len(state.registered_images),
        "num_degenerate_candidates": len(candidates),
        "thresholds": {
            "min_observations": args.min_observations,
            "max_median_error": args.max_median_error,
            "max_p95_error": args.max_p95_error,
            "max_above_threshold_ratio": args.max_above_threshold_ratio,
            "error_threshold": args.error_threshold,
            "min_covisibility_degree": args.min_covisibility_degree,
            "center_outlier_mad_scale": args.center_outlier_mad_scale,
        },
        "per_camera": per_camera,
    }
    report_path = report_dir / args.report_name
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Scene: {scene['scene_name']}")
    print(f"Registered images: {len(state.registered_images)}")
    print(f"Degenerate candidates: {len(candidates)}")
    for record in candidates[:10]:
        print(f"  {record['image_name']}: {', '.join(record['reasons'])}")
    print(f"Report: {report_path}")
    return 0


def load_residual_records(report_path: Path) -> list[dict]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    records = report.get("observation_errors", report.get("optimized_observation_errors", []))
    normalized = []
    for record in records:
        error = float(record["reprojection_error"] if "reprojection_error" in record else record["final_error"])
        normalized.append(
            {
                "image_name": str(record["image_name"]),
                "point3D_id": int(record["point3D_id"]),
                "keypoint_idx": int(record["keypoint_idx"]),
                "error": error,
            }
        )
    return normalized


def camera_center_distances(state) -> dict[str, float]:
    if not state.registered_images:
        return {}
    image_names = list(state.registered_images)
    centers = np.stack([state.registered_images[image_name].center for image_name in image_names]).astype(np.float64)
    median = np.median(centers, axis=0)
    distances = np.linalg.norm(centers - median.reshape(1, 3), axis=1)
    mad = np.median(np.abs(distances - np.median(distances)))
    scale = max(1.4826 * float(mad), 1e-6)
    robust = distances / scale
    return {image_name: float(value) for image_name, value in zip(image_names, robust)}


if __name__ == "__main__":
    raise SystemExit(main())
