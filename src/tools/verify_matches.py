from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from src.sfm.config import load_scene_config, resolve_project_path
from src.sfm.features import list_images
from src.sfm.geometry import (
    save_verification_result,
    verified_output_path,
    verify_fundamental_matrix,
)
from src.sfm.matching import ImageFeatures, draw_match_preview, load_features


def load_match_paths(match_dir: Path) -> list[Path]:
    return sorted(path for path in match_dir.glob("*.npz") if path.is_file())


def load_cached_verification(verified_path: Path) -> dict:
    data = np.load(verified_path)
    return {
        "image_name1": str(data["image_name1"]),
        "image_name2": str(data["image_name2"]),
        "verified_path": str(verified_path),
        "num_raw_matches": int(data["num_raw_matches"]),
        "num_inliers": int(data["num_inliers"]),
        "inlier_ratio": float(data["inlier_ratio"]),
        "status": str(data["status"]),
    }


def draw_verified_preview(
    image_dir: Path,
    feature_by_name: dict[str, ImageFeatures],
    verified_path: Path,
    output_path: Path,
) -> None:
    data = np.load(verified_path)
    image_name1 = str(data["image_name1"])
    image_name2 = str(data["image_name2"])
    draw_match_preview(
        image_dir / image_name1,
        image_dir / image_name2,
        feature_by_name[image_name1],
        feature_by_name[image_name2],
        data["inlier_matches"],
        output_path,
        limit=100,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify raw feature matches using fundamental matrix RANSAC.")
    parser.add_argument("--scene", required=True, help="Path to configs/scenes/<scene>.yaml")
    parser.add_argument("--max-pairs", type=int, default=0, help="Optional cap for debugging; 0 means all pairs.")
    parser.add_argument("--preview-count", type=int, default=3)
    parser.add_argument("--force", action="store_true", help="Recompute existing verified files.")
    args = parser.parse_args()

    config = load_scene_config(args.scene)
    default = config["default"]
    scene = config["scene"]

    image_dir = resolve_project_path(scene["image_dir"])
    feature_dir = resolve_project_path(scene["feature_dir"])
    match_dir = resolve_project_path(scene["match_dir"])
    verified_dir = resolve_project_path(scene["verified_dir"])
    output_dir = resolve_project_path(scene["output_dir"])
    figure_dir = output_dir / "figures"
    report_dir = output_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    min_num_matches = int(default["sfm"].get("min_num_matches", 30))
    min_num_inliers = int(default["sfm"].get("min_num_inliers", 30))
    ransac_reproj_threshold_px = float(default["sfm"].get("ransac_reproj_threshold_px", 4.0))
    ransac_confidence = float(default["sfm"].get("ransac_confidence", 0.999))

    images = list_images(image_dir)
    features = [load_features(feature_dir / f"{image_path.stem}.npz") for image_path in images]
    feature_by_name = {feature.image_name: feature for feature in features}

    match_paths = load_match_paths(match_dir)
    if args.max_pairs > 0:
        match_paths = match_paths[: args.max_pairs]
    if not match_paths:
        raise FileNotFoundError(f"No match files found in {match_dir}")

    summaries = []
    for match_path in tqdm(match_paths, desc=f"Verifying {scene['scene_name']}"):
        verified_path = verified_output_path(verified_dir, match_path)
        if verified_path.exists() and not args.force:
            summaries.append(load_cached_verification(verified_path))
            continue

        match_data = np.load(match_path)
        image_name1 = str(match_data["image_name1"])
        image_name2 = str(match_data["image_name2"])
        result = verify_fundamental_matrix(
            match_path=match_path,
            keypoints1=feature_by_name[image_name1].keypoints,
            keypoints2=feature_by_name[image_name2].keypoints,
            min_num_matches=min_num_matches,
            min_num_inliers=min_num_inliers,
            ransac_reproj_threshold_px=ransac_reproj_threshold_px,
            ransac_confidence=ransac_confidence,
        )
        save_verification_result(result, verified_path)
        summaries.append(
            {
                "image_name1": result.image_name1,
                "image_name2": result.image_name2,
                "verified_path": str(verified_path),
                "num_raw_matches": result.num_raw_matches,
                "num_inliers": result.num_inliers,
                "inlier_ratio": result.inlier_ratio,
                "status": result.status,
            }
        )

    verified_edges = [item for item in summaries if item["status"] == "verified"]
    nodes = [{"image_name": image_path.name} for image_path in images]
    scene_graph = {
        "scene_name": scene["scene_name"],
        "nodes": nodes,
        "edges": verified_edges,
    }
    scene_graph_path = verified_dir / "scene_graph.json"
    scene_graph_path.write_text(json.dumps(scene_graph, indent=2), encoding="utf-8")

    inlier_counts = np.array([item["num_inliers"] for item in summaries], dtype=np.int32)
    inlier_ratios = np.array([item["inlier_ratio"] for item in summaries], dtype=np.float32)
    status_counts = {}
    for item in summaries:
        status_counts[item["status"]] = status_counts.get(item["status"], 0) + 1

    report = {
        "scene_name": scene["scene_name"],
        "num_images": len(images),
        "num_pairs": len(match_paths),
        "num_verified_pairs": len(verified_edges),
        "status_counts": status_counts,
        "min_inliers": int(inlier_counts.min()) if len(inlier_counts) else 0,
        "mean_inliers": float(inlier_counts.mean()) if len(inlier_counts) else 0.0,
        "max_inliers": int(inlier_counts.max()) if len(inlier_counts) else 0,
        "min_inlier_ratio": float(inlier_ratios.min()) if len(inlier_ratios) else 0.0,
        "mean_inlier_ratio": float(inlier_ratios.mean()) if len(inlier_ratios) else 0.0,
        "max_inlier_ratio": float(inlier_ratios.max()) if len(inlier_ratios) else 0.0,
        "ransac_reproj_threshold_px": ransac_reproj_threshold_px,
        "ransac_confidence": ransac_confidence,
        "pairs": summaries,
    }
    report_path = report_dir / "geometric_verification_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    preview_candidates = sorted(verified_edges, key=lambda item: item["num_inliers"], reverse=True)[: args.preview_count]
    for item in preview_candidates:
        preview_path = (
            figure_dir
            / f"verified_{Path(item['image_name1']).stem}__{Path(item['image_name2']).stem}.jpg"
        )
        draw_verified_preview(
            image_dir=image_dir,
            feature_by_name=feature_by_name,
            verified_path=Path(item["verified_path"]),
            output_path=preview_path,
        )

    print(f"Scene: {scene['scene_name']}")
    print(f"Pairs: {len(match_paths)}")
    print(f"Verified pairs: {len(verified_edges)}")
    print(f"Status counts: {status_counts}")
    print(f"Inliers min/mean/max: {report['min_inliers']} / {report['mean_inliers']:.1f} / {report['max_inliers']}")
    print(f"Report: {report_path}")
    print(f"Scene graph: {scene_graph_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
