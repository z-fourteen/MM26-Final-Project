from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.sfm.camera import estimate_simple_pinhole
from src.sfm.config import load_scene_config, resolve_project_path
from src.sfm.features import list_images
from src.sfm.matching import load_features
from src.sfm.reconstruction import (
    Candidate2D3D,
    NextBestViewScore,
    add_registered_image,
    collect_candidate_correspondences,
    load_reconstruction_state,
    register_image_pnp,
    save_reconstruction_state,
    score_next_best_view,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Incrementally register images using PnP over initial 3D points.")
    parser.add_argument("--scene", required=True, help="Path to configs/scenes/<scene>.yaml")
    parser.add_argument("--max-register", type=int, default=10)
    parser.add_argument("--min-2d3d", type=int, default=30)
    parser.add_argument("--min-pnp-inliers", type=int, default=20)
    parser.add_argument("--focal-scale", type=float, default=1.2)
    parser.add_argument("--visibility-levels", type=int, default=3)
    parser.add_argument("--top-candidates", type=int, default=5)
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
    figure_dir.mkdir(parents=True, exist_ok=True)

    state = load_reconstruction_state(sparse_dir)
    images = list_images(image_dir)
    features_by_name = {
        image_path.name: load_features(feature_dir / f"{image_path.stem}.npz")
        for image_path in images
    }
    keypoints_by_name = {
        image_name: features.keypoints
        for image_name, features in features_by_name.items()
    }
    cameras = {}
    for image_path in images:
        # Feature files store image size as [width, height].
        feature_data = np.load(feature_dir / f"{image_path.stem}.npz")
        width, height = [int(value) for value in feature_data["image_size"]]
        cameras[image_path.name] = estimate_simple_pinhole(width, height, focal_scale=args.focal_scale)

    max_reproj_error = float(default["sfm"].get("max_reproj_error_px", 8.0))
    pnp_confidence = float(default["sfm"].get("ransac_confidence", 0.999))
    initial_registered_images = len(state.registered_images)
    initial_points3d = int(state.points3d.shape[0])
    initial_observations = len(state.observations)
    failed_images: set[str] = set()
    registrations = []

    for iteration in range(args.max_register):
        candidates = []
        scored_candidates: list[tuple[Candidate2D3D, NextBestViewScore]] = []
        for image_path in images:
            image_name = image_path.name
            if image_name in state.registered_images or image_name in failed_images:
                continue
            candidate = collect_candidate_correspondences(
                image_name=image_name,
                state=state,
                verified_dir=verified_dir,
                keypoints_by_name=keypoints_by_name,
            )
            candidates.append(candidate)
            score = score_next_best_view(
                candidate=candidate,
                camera=cameras[image_name],
                verified_dir=verified_dir,
                registered_image_names=set(state.registered_images),
                levels=args.visibility_levels,
            )
            scored_candidates.append((candidate, score))

        if not candidates:
            break

        scored_candidates.sort(key=lambda item: (item[1].pyramid_score, item[1].num_2d3d), reverse=True)
        eligible_candidates = [
            (candidate, score)
            for candidate, score in scored_candidates
            if candidate.num_correspondences >= args.min_2d3d
        ]
        if not eligible_candidates:
            break

        best_candidate, best_score = eligible_candidates[0]
        result = register_image_pnp(
            candidate=best_candidate,
            camera=cameras[best_candidate.image_name],
            min_2d3d=args.min_2d3d,
            min_pnp_inliers=args.min_pnp_inliers,
            reproj_error_px=max_reproj_error,
            confidence=pnp_confidence,
        )
        registration_record = {
            "image_name": result.image_name,
            "status": result.status,
            "num_2d3d": result.num_2d3d,
            "pnp_inliers": int(result.pnp_inliers.shape[0]),
            "inlier_ratio": float(result.pnp_inliers.shape[0] / max(result.num_2d3d, 1)),
            "mean_reprojection_error": result.mean_reprojection_error,
            "selection_method": "phase5c_pyramid_visibility",
            "iteration": iteration + 1,
            "pyramid_visibility_score": best_score.pyramid_score,
            "registered_neighbor_count": best_score.registered_neighbor_count,
            "num_general_edges": best_score.num_general_edges,
            "num_planar_edges": best_score.num_planar_edges,
            "mean_homography_ratio": best_score.mean_homography_ratio,
            "eligible_candidates": len(eligible_candidates),
            "total_candidates": len(scored_candidates),
            "top_candidates": [
                summarize_nbv_candidate(candidate, score, min_2d3d=args.min_2d3d)
                for candidate, score in scored_candidates[: max(args.top_candidates, 1)]
            ],
        }
        registrations.append(registration_record)
        if result.success:
            add_registered_image(state, result)
        else:
            failed_images.add(result.image_name)

    state_path, registered_npz_path = save_reconstruction_state(state, sparse_dir)
    plot_camera_centers(state, figure_dir / "registered_camera_centers.png")

    report = {
        "scene_name": scene["scene_name"],
        "initial_registered_images": initial_registered_images,
        "final_registered_images": len(state.registered_images),
        "attempted_images": len(registrations),
        "successful_registrations": sum(1 for item in registrations if item["status"] == "registered"),
        "failed_registrations": sum(1 for item in registrations if item["status"] != "registered"),
        "initial_points3D": initial_points3d,
        "final_points3D": int(state.points3d.shape[0]),
        "initial_observations": initial_observations,
        "final_observations": len(state.observations),
        "selection_method": "phase5c_pyramid_visibility",
        "visibility_levels": args.visibility_levels,
        "state_path": str(state_path),
        "registered_npz_path": str(registered_npz_path),
        "per_image": registrations,
    }
    report_path = report_dir / "registration_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Scene: {scene['scene_name']}")
    print(f"Registered images: {report['final_registered_images']}")
    print(f"Successful new registrations: {report['successful_registrations']}")
    print(f"Failed attempts: {report['failed_registrations']}")
    print(f"State: {state_path}")
    print(f"Registered NPZ: {registered_npz_path}")
    print(f"Report: {report_path}")
    return 0


def summarize_nbv_candidate(candidate, score: NextBestViewScore, min_2d3d: int) -> dict:
    return {
        "image_name": candidate.image_name,
        "num_2d3d": candidate.num_correspondences,
        "eligible": candidate.num_correspondences >= min_2d3d,
        "pyramid_visibility_score": score.pyramid_score,
        "registered_neighbor_count": score.registered_neighbor_count,
        "num_general_edges": score.num_general_edges,
        "num_planar_edges": score.num_planar_edges,
        "mean_homography_ratio": score.mean_homography_ratio,
    }


def plot_camera_centers(state, output_path: Path) -> None:
    centers = {
        image_name: registered.center
        for image_name, registered in state.registered_images.items()
    }
    if not centers:
        return
    names = list(centers)
    values = [centers[name] for name in names]
    xs = [float(center[0]) for center in values]
    zs = [float(center[2]) for center in values]

    plt.figure(figsize=(8, 6))
    plt.scatter(xs, zs, c="tab:blue", s=32)
    for name, x, z in zip(names, xs, zs):
        plt.text(x, z, Path(name).stem, fontsize=7)
    plt.xlabel("Camera center X")
    plt.ylabel("Camera center Z")
    plt.title("Registered camera centers")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


if __name__ == "__main__":
    raise SystemExit(main())
