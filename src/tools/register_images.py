from __future__ import annotations

import argparse
import json
import sys
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
from src.tools import evaluate_degenerate_cameras
from src.tools import evaluate_registered_residuals
from src.tools import filter_reconstruction
from src.tools import run_bundle_adjustment


def main() -> int:
    parser = argparse.ArgumentParser(description="Incrementally register images using PnP over initial 3D points.")
    parser.add_argument("--scene", required=True, help="Path to configs/scenes/<scene>.yaml")
    parser.add_argument("--max-register", type=int, default=10)
    parser.add_argument("--min-2d3d", type=int, default=30)
    parser.add_argument("--min-pnp-inliers", type=int, default=20)
    parser.add_argument("--focal-scale", type=float, default=1.2)
    parser.add_argument("--visibility-levels", type=int, default=3)
    parser.add_argument("--top-candidates", type=int, default=5)
    parser.add_argument("--local-ba-after-registration", action="store_true")
    parser.add_argument("--local-ba-neighbors", type=int, default=6)
    parser.add_argument("--local-ba-max-iterations", type=int, default=20)
    parser.add_argument("--local-ba-max-points", type=int, default=800)
    parser.add_argument("--local-ba-max-observations", type=int, default=4000)
    parser.add_argument("--global-ba-growth-ratio", type=float, default=0.0)
    parser.add_argument("--global-ba-max-iterations", type=int, default=40)
    parser.add_argument("--global-ba-max-points", type=int, default=2000)
    parser.add_argument("--global-ba-max-observations", type=int, default=12000)
    parser.add_argument("--ba-loss", default="cauchy", choices=["linear", "soft_l1", "huber", "cauchy", "arctan"])
    parser.add_argument("--ba-f-scale", type=float, default=4.0)
    parser.add_argument("--filter-after-ba", action="store_true")
    parser.add_argument("--filter-max-reprojection-error", type=float, default=8.0)
    parser.add_argument("--filter-max-point-median-error", type=float, default=8.0)
    parser.add_argument("--filter-max-point-max-error", type=float, default=32.0)
    parser.add_argument("--filter-min-track-length", type=int, default=2)
    parser.add_argument("--diagnose-degenerate-cameras", action="store_true")
    parser.add_argument("--remove-degenerate-cameras", action="store_true")
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
    refinement_reports = []
    last_global_ba_images = len(state.registered_images)
    last_global_ba_points = int(state.points3d.shape[0])

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
            state_path, registered_npz_path = save_reconstruction_state(state, sparse_dir)
            local_reports = []
            if args.local_ba_after_registration:
                local_reports.extend(
                    run_ba_filtering_cycle(
                        args=args,
                        stage_prefix=f"registration_iter{iteration + 1:02d}_{Path(result.image_name).stem}_local",
                        scope="local",
                        target_image=result.image_name,
                        max_iterations=args.local_ba_max_iterations,
                        max_points=args.local_ba_max_points,
                        max_observations=args.local_ba_max_observations,
                        local_neighbors=args.local_ba_neighbors,
                    )
                )
                state = load_reconstruction_state(sparse_dir)
            if should_run_global_ba(
                state=state,
                growth_ratio=args.global_ba_growth_ratio,
                last_global_ba_images=last_global_ba_images,
                last_global_ba_points=last_global_ba_points,
            ):
                local_reports.extend(
                    run_ba_filtering_cycle(
                        args=args,
                        stage_prefix=f"registration_iter{iteration + 1:02d}_global",
                        scope="global",
                        target_image="",
                        max_iterations=args.global_ba_max_iterations,
                        max_points=args.global_ba_max_points,
                        max_observations=args.global_ba_max_observations,
                        local_neighbors=args.local_ba_neighbors,
                    )
                )
                state = load_reconstruction_state(sparse_dir)
                last_global_ba_images = len(state.registered_images)
                last_global_ba_points = int(state.points3d.shape[0])
            registration_record["refinement_reports"] = local_reports
            refinement_reports.extend(local_reports)
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
        "local_ba_after_registration": bool(args.local_ba_after_registration),
        "global_ba_growth_ratio": float(args.global_ba_growth_ratio),
        "filter_after_ba": bool(args.filter_after_ba),
        "diagnose_degenerate_cameras": bool(args.diagnose_degenerate_cameras),
        "remove_degenerate_cameras": bool(args.remove_degenerate_cameras),
        "state_path": str(state_path),
        "registered_npz_path": str(registered_npz_path),
        "refinement_reports": refinement_reports,
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


def should_run_global_ba(
    state,
    growth_ratio: float,
    last_global_ba_images: int,
    last_global_ba_points: int,
) -> bool:
    if growth_ratio <= 0:
        return False
    current_images = len(state.registered_images)
    current_points = int(state.points3d.shape[0])
    image_trigger = current_images >= max(last_global_ba_images + 1, int(np.ceil(last_global_ba_images * growth_ratio)))
    point_trigger = current_points >= max(last_global_ba_points + 1, int(np.ceil(last_global_ba_points * growth_ratio)))
    return image_trigger or point_trigger


def run_ba_filtering_cycle(
    args: argparse.Namespace,
    stage_prefix: str,
    scope: str,
    target_image: str,
    max_iterations: int,
    max_points: int,
    max_observations: int,
    local_neighbors: int,
) -> list[dict]:
    reports = []
    ba_report = f"{stage_prefix}_ba_report.json"
    argv = [
        "run_bundle_adjustment",
        "--scene",
        args.scene,
        "--scope",
        scope,
        "--max-iterations",
        str(max_iterations),
        "--loss",
        args.ba_loss,
        "--f-scale",
        str(args.ba_f_scale),
        "--max-points",
        str(max_points),
        "--max-observations",
        str(max_observations),
        "--min-track-length",
        str(args.filter_min_track_length),
        "--report-name",
        ba_report,
    ]
    if scope == "local":
        argv.extend(["--target-image", target_image, "--local-neighbors", str(local_neighbors)])
    call_tool(run_bundle_adjustment.main, argv)
    reports.append({"stage": stage_prefix, "type": "ba", "report_name": ba_report})

    residual_report = f"{stage_prefix}_registered_residual_report.json"
    call_tool(
        evaluate_registered_residuals.main,
        [
            "evaluate_registered_residuals",
            "--scene",
            args.scene,
            "--report-name",
            residual_report,
        ],
    )
    reports.append({"stage": stage_prefix, "type": "registered_residual", "report_name": residual_report})

    residual_for_degenerate = residual_report
    if args.filter_after_ba:
        filtering_report = f"{stage_prefix}_filtering_report.json"
        call_tool(
            filter_reconstruction.main,
            [
                "filter_reconstruction",
                "--scene",
                args.scene,
                "--ba-report",
                report_path_for_scene(args.scene, residual_report),
                "--max-reprojection-error",
                str(args.filter_max_reprojection_error),
                "--max-point-median-error",
                str(args.filter_max_point_median_error),
                "--max-point-max-error",
                str(args.filter_max_point_max_error),
                "--min-track-length",
                str(args.filter_min_track_length),
                "--report-name",
                filtering_report,
            ],
        )
        reports.append({"stage": stage_prefix, "type": "filtering", "report_name": filtering_report})
        residual_for_degenerate = f"{stage_prefix}_registered_residual_after_filtering_report.json"
        call_tool(
            evaluate_registered_residuals.main,
            [
                "evaluate_registered_residuals",
                "--scene",
                args.scene,
                "--report-name",
                residual_for_degenerate,
            ],
        )
        reports.append(
            {"stage": stage_prefix, "type": "registered_residual_after_filtering", "report_name": residual_for_degenerate}
        )

    if args.diagnose_degenerate_cameras or args.remove_degenerate_cameras:
        degenerate_report = f"{stage_prefix}_degenerate_camera_report.json"
        degenerate_argv = [
            "evaluate_degenerate_cameras",
            "--scene",
            args.scene,
            "--residual-report",
            report_path_for_scene(args.scene, residual_for_degenerate),
            "--report-name",
            degenerate_report,
        ]
        if args.remove_degenerate_cameras:
            degenerate_argv.append("--remove-candidates")
        call_tool(evaluate_degenerate_cameras.main, degenerate_argv)
        reports.append({"stage": stage_prefix, "type": "degenerate_cameras", "report_name": degenerate_report})
    return reports


def report_path_for_scene(scene_path: str, report_name: str) -> str:
    config = load_scene_config(scene_path)
    report_dir = resolve_project_path(config["scene"]["output_dir"]) / "reports"
    return str(report_dir / report_name)


def call_tool(main_func, argv: list[str]) -> None:
    old_argv = sys.argv
    try:
        sys.argv = argv
        exit_code = main_func()
        if exit_code not in (0, None):
            raise RuntimeError(f"Tool failed with exit code {exit_code}: {' '.join(argv)}")
    finally:
        sys.argv = old_argv


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
