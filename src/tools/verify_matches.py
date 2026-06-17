from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from src.sfm.camera import load_camera_from_feature, load_camera_overrides
from src.sfm.config import load_scene_config, resolve_project_path
from src.sfm.features import list_images
from src.sfm.geometry import (
    save_verification_result,
    verified_output_path,
    verify_geometric_models,
)
from src.sfm.matching import ImageFeatures, draw_match_preview, load_features


def load_match_paths(match_dir: Path) -> list[Path]:
    return sorted(path for path in match_dir.glob("*.npz") if path.is_file())


def load_cached_verification(verified_path: Path) -> dict:
    data = np.load(verified_path)
    num_inliers = int(data["num_inliers"])
    num_h_inliers = int(data["num_h_inliers"]) if "num_h_inliers" in data else 0
    num_e_inliers = int(data["num_e_inliers"]) if "num_e_inliers" in data else 0
    num_s_inliers = int(data["num_s_inliers"]) if "num_s_inliers" in data else 0
    return {
        "image_name1": str(data["image_name1"]),
        "image_name2": str(data["image_name2"]),
        "verified_path": str(verified_path),
        "num_raw_matches": int(data["num_raw_matches"]),
        "num_inliers": num_inliers,
        "num_h_inliers": num_h_inliers,
        "num_e_inliers": num_e_inliers,
        "num_s_inliers": num_s_inliers,
        "inlier_ratio": float(data["inlier_ratio"]),
        "h_inlier_ratio": float(data["h_inlier_ratio"]) if "h_inlier_ratio" in data else 0.0,
        "e_inlier_ratio": float(data["e_inlier_ratio"]) if "e_inlier_ratio" in data else 0.0,
        "s_inlier_ratio": float(data["s_inlier_ratio"]) if "s_inlier_ratio" in data else 0.0,
        "homography_ratio": float(data["homography_ratio"]) if "homography_ratio" in data else 0.0,
        "essential_ratio": float(data["essential_ratio"]) if "essential_ratio" in data else 0.0,
        "similarity_ratio": float(data["similarity_ratio"]) if "similarity_ratio" in data else 0.0,
        "median_triangulation_angle_deg": (
            float(data["median_triangulation_angle_deg"]) if "median_triangulation_angle_deg" in data else 0.0
        ),
        "model_type": str(data["model_type"]) if "model_type" in data else "general",
        "calibration_status": str(data["calibration_status"]) if "calibration_status" in data else "uncertain",
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
    parser.add_argument("--focal-scale", type=float, default=1.2)
    args = parser.parse_args()

    config = load_scene_config(args.scene)
    default = config["default"]
    scene = config["scene"]

    image_dir = resolve_project_path(scene["image_dir"])
    feature_dir = resolve_project_path(scene["feature_dir"])
    match_dir = resolve_project_path(scene["match_dir"])
    verified_dir = resolve_project_path(scene["verified_dir"])
    sparse_dir = resolve_project_path(scene["sparse_dir"])
    output_dir = resolve_project_path(scene["output_dir"])
    figure_dir = output_dir / "figures"
    report_dir = output_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    min_num_matches = int(default["sfm"].get("min_num_matches", 30))
    min_num_inliers = int(default["sfm"].get("min_num_inliers", 30))
    ransac_reproj_threshold_px = float(default["sfm"].get("ransac_reproj_threshold_px", 4.0))
    homography_reproj_threshold_px = float(default["sfm"].get("homography_reproj_threshold_px", ransac_reproj_threshold_px))
    essential_reproj_threshold_px = float(default["sfm"].get("essential_reproj_threshold_px", ransac_reproj_threshold_px))
    ransac_confidence = float(default["sfm"].get("ransac_confidence", 0.999))
    homography_ratio_threshold = float(default["sfm"].get("homography_ratio_threshold", 0.8))
    essential_ratio_threshold = float(default["sfm"].get("essential_ratio_threshold", 0.6))
    panoramic_triangulation_angle_deg = float(default["sfm"].get("panoramic_triangulation_angle_deg", 1.0))
    wtf_border_ratio = float(default["sfm"].get("wtf_border_ratio", 0.1))
    wtf_similarity_ratio_threshold = float(default["sfm"].get("wtf_similarity_ratio_threshold", 0.7))

    images = list_images(image_dir)
    features = [load_features(feature_dir / f"{image_path.stem}.npz") for image_path in images]
    feature_by_name = {feature.image_name: feature for feature in features}
    cameras = {}
    camera_overrides = load_camera_overrides(sparse_dir)
    for image_path in images:
        cameras[image_path.name] = load_camera_from_feature(
            feature_path=feature_dir / f"{image_path.stem}.npz",
            image_path=image_path,
            focal_scale=args.focal_scale,
            prefer_exif=bool(default.get("camera", {}).get("estimate_focal_from_exif", True)),
            override=camera_overrides.get(image_path.name),
        )

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
        result = verify_geometric_models(
            match_path=match_path,
            keypoints1=feature_by_name[image_name1].keypoints,
            keypoints2=feature_by_name[image_name2].keypoints,
            camera1=cameras[image_name1],
            camera2=cameras[image_name2],
            min_num_matches=min_num_matches,
            min_num_inliers=min_num_inliers,
            ransac_reproj_threshold_px=ransac_reproj_threshold_px,
            homography_reproj_threshold_px=homography_reproj_threshold_px,
            essential_reproj_threshold_px=essential_reproj_threshold_px,
            ransac_confidence=ransac_confidence,
            homography_ratio_threshold=homography_ratio_threshold,
            essential_ratio_threshold=essential_ratio_threshold,
            panoramic_triangulation_angle_deg=panoramic_triangulation_angle_deg,
            wtf_border_ratio=wtf_border_ratio,
            wtf_similarity_ratio_threshold=wtf_similarity_ratio_threshold,
        )
        save_verification_result(result, verified_path)
        summaries.append(
            {
                "image_name1": result.image_name1,
                "image_name2": result.image_name2,
                "verified_path": str(verified_path),
                "num_raw_matches": result.num_raw_matches,
                "num_inliers": result.num_inliers,
                "num_h_inliers": result.num_h_inliers,
                "num_e_inliers": result.num_e_inliers,
                "num_s_inliers": result.num_s_inliers,
                "inlier_ratio": result.inlier_ratio,
                "h_inlier_ratio": result.h_inlier_ratio,
                "e_inlier_ratio": result.e_inlier_ratio,
                "s_inlier_ratio": result.s_inlier_ratio,
                "homography_ratio": result.homography_ratio,
                "essential_ratio": result.essential_ratio,
                "similarity_ratio": result.similarity_ratio,
                "median_triangulation_angle_deg": result.median_triangulation_angle_deg,
                "model_type": result.model_type,
                "calibration_status": result.calibration_status,
                "status": result.status,
            }
        )

    scene_graph_statuses = {"verified", "verified_planar", "verified_panoramic"}
    verified_edges = [item for item in summaries if item["status"] in scene_graph_statuses]
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
    homography_ratios = np.array([item["homography_ratio"] for item in summaries], dtype=np.float32)
    essential_ratios = np.array([item["essential_ratio"] for item in summaries], dtype=np.float32)
    triangulation_angles = np.array([item["median_triangulation_angle_deg"] for item in summaries], dtype=np.float32)
    status_counts = {}
    model_type_counts = {}
    calibration_status_counts = {}
    for item in summaries:
        status_counts[item["status"]] = status_counts.get(item["status"], 0) + 1
        model_type_counts[item["model_type"]] = model_type_counts.get(item["model_type"], 0) + 1
        calibration_status_counts[item["calibration_status"]] = calibration_status_counts.get(item["calibration_status"], 0) + 1

    report = {
        "scene_name": scene["scene_name"],
        "num_images": len(images),
        "num_pairs": len(match_paths),
        "num_verified_pairs": len(verified_edges),
        "min_inliers": int(inlier_counts.min()) if len(inlier_counts) else 0,
        "mean_inliers": float(inlier_counts.mean()) if len(inlier_counts) else 0.0,
        "max_inliers": int(inlier_counts.max()) if len(inlier_counts) else 0,
        "min_inlier_ratio": float(inlier_ratios.min()) if len(inlier_ratios) else 0.0,
        "mean_inlier_ratio": float(inlier_ratios.mean()) if len(inlier_ratios) else 0.0,
        "max_inlier_ratio": float(inlier_ratios.max()) if len(inlier_ratios) else 0.0,
        "min_homography_ratio": float(homography_ratios.min()) if len(homography_ratios) else 0.0,
        "mean_homography_ratio": float(homography_ratios.mean()) if len(homography_ratios) else 0.0,
        "max_homography_ratio": float(homography_ratios.max()) if len(homography_ratios) else 0.0,
        "min_essential_ratio": float(essential_ratios.min()) if len(essential_ratios) else 0.0,
        "mean_essential_ratio": float(essential_ratios.mean()) if len(essential_ratios) else 0.0,
        "max_essential_ratio": float(essential_ratios.max()) if len(essential_ratios) else 0.0,
        "median_triangulation_angle_deg": float(np.median(triangulation_angles)) if len(triangulation_angles) else 0.0,
        "status_counts": status_counts,
        "model_type_counts": model_type_counts,
        "calibration_status_counts": calibration_status_counts,
        "ransac_reproj_threshold_px": ransac_reproj_threshold_px,
        "homography_reproj_threshold_px": homography_reproj_threshold_px,
        "essential_reproj_threshold_px": essential_reproj_threshold_px,
        "ransac_confidence": ransac_confidence,
        "homography_ratio_threshold": homography_ratio_threshold,
        "essential_ratio_threshold": essential_ratio_threshold,
        "panoramic_triangulation_angle_deg": panoramic_triangulation_angle_deg,
        "wtf_border_ratio": wtf_border_ratio,
        "wtf_similarity_ratio_threshold": wtf_similarity_ratio_threshold,
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
    print(f"Model type counts: {model_type_counts}")
    print(f"Calibration counts: {calibration_status_counts}")
    print(f"Inliers min/mean/max: {report['min_inliers']} / {report['mean_inliers']:.1f} / {report['max_inliers']}")
    print(
        "Homography ratio min/mean/max: "
        f"{report['min_homography_ratio']:.3f} / {report['mean_homography_ratio']:.3f} / {report['max_homography_ratio']:.3f}"
    )
    print(f"Report: {report_path}")
    print(f"Scene graph: {scene_graph_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
