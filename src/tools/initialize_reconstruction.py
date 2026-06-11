from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from src.sfm.camera import estimate_simple_pinhole
from src.sfm.config import load_scene_config, resolve_project_path
from src.sfm.features import list_images
from src.sfm.initialization import (
    InitializationResult,
    initialization_succeeds,
    initialize_two_view,
    rank_initial_pair_candidates,
    save_initialization,
)
from src.sfm.matching import draw_match_preview, load_features


def read_rgb(image_path: Path) -> np.ndarray:
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Failed to read image: {image_path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def main() -> int:
    parser = argparse.ArgumentParser(description="Initialize a two-view SfM reconstruction.")
    parser.add_argument("--scene", required=True, help="Path to configs/scenes/<scene>.yaml")
    parser.add_argument("--focal-scale", type=float, default=1.2)
    parser.add_argument("--max-candidate-pairs", type=int, default=20)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()

    config = load_scene_config(args.scene)
    default = config["default"]
    scene = config["scene"]

    image_dir = resolve_project_path(scene["image_dir"])
    feature_dir = resolve_project_path(scene["feature_dir"])
    verified_dir = resolve_project_path(scene["verified_dir"])
    sparse_dir = resolve_project_path(scene["sparse_dir"])
    output_dir = resolve_project_path(scene["output_dir"])
    figure_dir = output_dir / "figures"
    report_dir = output_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    scene_graph_path = verified_dir / "scene_graph.json"
    if not scene_graph_path.exists():
        raise FileNotFoundError(f"Scene graph not found: {scene_graph_path}")

    scene_graph = json.loads(scene_graph_path.read_text(encoding="utf-8"))
    images = list_images(image_dir)
    image_by_name = {image_path.name: image_path for image_path in images}

    candidates = rank_initial_pair_candidates(scene_graph["edges"])
    if not candidates:
        raise ValueError("Scene graph has no Phase 4B initialization candidates.")
    candidates = candidates[: max(args.max_candidate_pairs, 1)]
    candidate_report = [
        summarize_candidate(edge, rank=index + 1)
        for index, edge in enumerate(candidates)
    ]

    min_initial_points = int(default["sfm"].get("min_initial_points", 100))
    min_initial_kept_ratio = float(default["sfm"].get("min_initial_kept_ratio", 0.05))
    max_initial_median_reproj_error_px = float(
        default["sfm"].get("max_initial_median_reproj_error_px", default["sfm"].get("max_reproj_error_px", 8.0))
    )

    selected_edge = None
    selected_rank = 0
    result: InitializationResult | None = None
    attempts = []
    feature_cache = {}
    image_cache = {}
    camera_cache = {}
    for rank, edge in enumerate(candidates, start=1):
        verified_path = Path(edge["verified_path"])
        if not verified_path.is_absolute():
            verified_path = resolve_project_path(verified_path)

        image_path1 = image_by_name[edge["image_name1"]]
        image_path2 = image_by_name[edge["image_name2"]]
        features1 = feature_cache.setdefault(image_path1.name, load_features(feature_dir / f"{image_path1.stem}.npz"))
        features2 = feature_cache.setdefault(image_path2.name, load_features(feature_dir / f"{image_path2.stem}.npz"))
        image1_rgb = image_cache.setdefault(image_path1.name, read_rgb(image_path1))
        image2_rgb = image_cache.setdefault(image_path2.name, read_rgb(image_path2))
        camera1 = camera_cache.setdefault(image_path1.name, camera_from_rgb(image1_rgb, args.focal_scale))
        camera2 = camera_cache.setdefault(image_path2.name, camera_from_rgb(image2_rgb, args.focal_scale))

        candidate_result = initialize_two_view(
            edge=edge,
            verified_path=verified_path,
            keypoints1=features1.keypoints,
            keypoints2=features2.keypoints,
            camera1=camera1,
            camera2=camera2,
            image1_rgb=image1_rgb,
            max_reproj_error_px=float(default["sfm"].get("max_reproj_error_px", 8.0)),
        )
        median_error = (
            float(np.median(candidate_result.reprojection_errors))
            if candidate_result.reprojection_errors.size
            else 0.0
        )
        attempts.append(
            {
                **summarize_candidate(edge, rank=rank),
                "candidate_points": candidate_result.num_candidate_points,
                "kept_points": candidate_result.num_kept_points,
                "kept_ratio": candidate_result.kept_ratio,
                "cheirality_ratio": candidate_result.cheirality_ratio,
                "pose_inliers": candidate_result.pose_inliers,
                "median_reprojection_error": median_error,
                "mean_reprojection_error": (
                    float(np.mean(candidate_result.reprojection_errors))
                    if candidate_result.reprojection_errors.size
                    else 0.0
                ),
                "median_initial_triangulation_angle_deg": candidate_result.median_triangulation_angle_deg,
            }
        )
        if initialization_succeeds(
            candidate_result,
            min_initial_points=min_initial_points,
            min_initial_kept_ratio=min_initial_kept_ratio,
            max_initial_median_reproj_error_px=max_initial_median_reproj_error_px,
        ):
            selected_edge = edge
            selected_rank = rank
            result = candidate_result
            break

    if result is None or selected_edge is None:
        raise ValueError("No initialization candidate passed Phase 4B quality checks.")

    pair_path, points_path = save_initialization(result, sparse_dir)

    median_error = float(np.median(result.reprojection_errors)) if result.num_kept_points else 0.0
    mean_error = float(np.mean(result.reprojection_errors)) if result.num_kept_points else 0.0
    report = {
        "scene_name": scene["scene_name"],
        "selected_pair": [result.image_name1, result.image_name2],
        "verified_path": str(result.verified_path),
        "num_candidate_pairs": len(scene_graph["edges"]),
        "ranked_candidate_pairs": len(candidates),
        "selected_rank": selected_rank,
        "selected_num_inliers": int(selected_edge["num_inliers"]),
        "selected_inlier_ratio": float(selected_edge["inlier_ratio"]),
        "selected_model_type": str(selected_edge["model_type"]),
        "selected_calibration_status": str(selected_edge["calibration_status"]),
        "selected_homography_ratio": float(selected_edge["homography_ratio"]),
        "selected_essential_ratio": float(selected_edge["essential_ratio"]),
        "selected_phase3b_triangulation_angle_deg": float(selected_edge["median_triangulation_angle_deg"]),
        "selected_graph_degree_score": float(selected_edge["graph_degree_score"]),
        "score": result.score,
        "candidate_points": result.num_candidate_points,
        "kept_points": result.num_kept_points,
        "kept_ratio": result.kept_ratio,
        "pose_inliers": result.pose_inliers,
        "cheirality_ratio": result.cheirality_ratio,
        "median_initial_triangulation_angle_deg": result.median_triangulation_angle_deg,
        "median_reprojection_error": median_error,
        "mean_reprojection_error": mean_error,
        "baseline_norm": float(np.linalg.norm(result.t2)),
        "quality_thresholds": {
            "min_initial_points": min_initial_points,
            "min_initial_kept_ratio": min_initial_kept_ratio,
            "max_initial_median_reproj_error_px": max_initial_median_reproj_error_px,
        },
        "candidate_attempts": attempts,
        "pair_path": str(pair_path),
        "points_path": str(points_path),
    }
    report_path = report_dir / "initialization_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    candidates_report_path = report_dir / "initialization_candidates_report.json"
    candidates_report_path.write_text(json.dumps(candidate_report, indent=2), encoding="utf-8")

    preview_path = figure_dir / "initial_pair_matches.jpg"
    image_path1 = image_by_name[result.image_name1]
    image_path2 = image_by_name[result.image_name2]
    features1 = feature_cache[result.image_name1]
    features2 = feature_cache[result.image_name2]
    draw_match_preview(
        image_path1,
        image_path2,
        features1,
        features2,
        result.inlier_matches,
        preview_path,
        limit=120,
    )

    print(f"Scene: {scene['scene_name']}")
    print(f"Selected pair: {result.image_name1} <-> {result.image_name2}")
    print(f"Selected rank: {selected_rank} / {len(candidates)}")
    print(f"Candidate scene graph edges: {len(scene_graph['edges'])}")
    print(f"Points candidate/kept: {result.num_candidate_points} / {result.num_kept_points}")
    print(f"Kept ratio: {result.kept_ratio:.3f}")
    print(f"Reprojection error median/mean: {median_error:.3f} / {mean_error:.3f}")
    print(f"Pair: {pair_path}")
    print(f"Points: {points_path}")
    print(f"Report: {report_path}")
    print(f"Candidates: {candidates_report_path}")
    print(f"Preview: {preview_path}")
    return 0


def camera_from_rgb(image_rgb: np.ndarray, focal_scale: float):
    height, width = image_rgb.shape[:2]
    return estimate_simple_pinhole(width=width, height=height, focal_scale=focal_scale)


def summarize_candidate(edge: dict, rank: int) -> dict:
    return {
        "rank": rank,
        "image_name1": edge["image_name1"],
        "image_name2": edge["image_name2"],
        "score": float(edge["initialization_score"]),
        "status": edge["status"],
        "model_type": edge["model_type"],
        "calibration_status": edge["calibration_status"],
        "num_inliers": int(edge["num_inliers"]),
        "inlier_ratio": float(edge["inlier_ratio"]),
        "homography_ratio": float(edge["homography_ratio"]),
        "essential_ratio": float(edge["essential_ratio"]),
        "median_triangulation_angle_deg": float(edge["median_triangulation_angle_deg"]),
        "graph_degree_score": float(edge["graph_degree_score"]),
    }


if __name__ == "__main__":
    raise SystemExit(main())
