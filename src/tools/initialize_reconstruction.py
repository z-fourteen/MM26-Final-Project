from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from src.sfm.camera import estimate_simple_pinhole
from src.sfm.config import load_scene_config, resolve_project_path
from src.sfm.features import list_images
from src.sfm.initialization import initialize_two_view, save_initialization, select_initial_pair
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
    edge = select_initial_pair(scene_graph["edges"])
    verified_path = Path(edge["verified_path"])
    if not verified_path.is_absolute():
        verified_path = resolve_project_path(verified_path)

    images = list_images(image_dir)
    image_by_name = {image_path.name: image_path for image_path in images}
    image_path1 = image_by_name[edge["image_name1"]]
    image_path2 = image_by_name[edge["image_name2"]]

    features1 = load_features(feature_dir / f"{image_path1.stem}.npz")
    features2 = load_features(feature_dir / f"{image_path2.stem}.npz")

    image1_rgb = read_rgb(image_path1)
    image2_rgb = read_rgb(image_path2)
    height1, width1 = image1_rgb.shape[:2]
    height2, width2 = image2_rgb.shape[:2]
    camera1 = estimate_simple_pinhole(width=width1, height=height1, focal_scale=args.focal_scale)
    camera2 = estimate_simple_pinhole(width=width2, height=height2, focal_scale=args.focal_scale)

    result = initialize_two_view(
        edge=edge,
        verified_path=verified_path,
        keypoints1=features1.keypoints,
        keypoints2=features2.keypoints,
        camera1=camera1,
        camera2=camera2,
        image1_rgb=image1_rgb,
        max_reproj_error_px=float(default["sfm"].get("max_reproj_error_px", 8.0)),
    )
    pair_path, points_path = save_initialization(result, sparse_dir)

    median_error = float(np.median(result.reprojection_errors)) if result.num_kept_points else 0.0
    mean_error = float(np.mean(result.reprojection_errors)) if result.num_kept_points else 0.0
    report = {
        "scene_name": scene["scene_name"],
        "selected_pair": [result.image_name1, result.image_name2],
        "verified_path": str(verified_path),
        "num_candidate_pairs": len(scene_graph["edges"]),
        "selected_num_inliers": int(edge["num_inliers"]),
        "selected_inlier_ratio": float(edge["inlier_ratio"]),
        "score": result.score,
        "candidate_points": result.num_candidate_points,
        "kept_points": result.num_kept_points,
        "median_reprojection_error": median_error,
        "mean_reprojection_error": mean_error,
        "baseline_norm": float(np.linalg.norm(result.t2)),
        "pair_path": str(pair_path),
        "points_path": str(points_path),
    }
    report_path = report_dir / "initialization_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    preview_path = figure_dir / "initial_pair_matches.jpg"
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
    print(f"Candidate verified pairs: {len(scene_graph['edges'])}")
    print(f"Points candidate/kept: {result.num_candidate_points} / {result.num_kept_points}")
    print(f"Reprojection error median/mean: {median_error:.3f} / {mean_error:.3f}")
    print(f"Pair: {pair_path}")
    print(f"Points: {points_path}")
    print(f"Report: {report_path}")
    print(f"Preview: {preview_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
