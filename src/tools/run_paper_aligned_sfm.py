from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from src.sfm.config import load_scene_config, resolve_project_path
from src.sfm.filtering import filter_reconstruction_by_ba_errors
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
from src.sfm.residuals import (
    evaluate_registered_residuals_in_memory,
    write_residual_npz,
    write_residual_report,
)
from src.sfm.runtime import SfMRuntimeContext, load_runtime_context
from src.tools import evaluate_degenerate_cameras
from src.tools import filter_reconstruction
from src.tools import run_bundle_adjustment
from src.tools import triangulate_registered_tracks


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the paper-aligned incremental SfM controller.")
    parser.add_argument("--scene", required=True, help="Path to configs/scenes/<scene>.yaml")
    parser.add_argument("--max-register", type=int, default=20)
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
    parser.add_argument("--global-ba-growth-ratio", type=float, default=1.1)
    parser.add_argument("--global-ba-min-interval", type=int, default=0)
    parser.add_argument("--global-ba-max-iterations", type=int, default=40)
    parser.add_argument("--global-ba-max-points", type=int, default=2500)
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
    parser.add_argument("--enable-recursive-track-splitting", action="store_true")
    parser.add_argument("--run-final-refinement", action="store_true")
    parser.add_argument("--report-name", default="paper_aligned_sfm_report.json")
    args = parser.parse_args()

    config = load_scene_config(args.scene)
    default = config["default"]
    scene = config["scene"]
    sparse_dir = resolve_project_path(scene["sparse_dir"])
    output_dir = resolve_project_path(scene["output_dir"])
    report_dir = output_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    runtime = load_runtime_context(args.scene, focal_scale=args.focal_scale)

    max_reproj_error = float(default["sfm"].get("max_reproj_error_px", 8.0))
    pnp_confidence = float(default["sfm"].get("ransac_confidence", 0.999))

    state = load_reconstruction_state(sparse_dir)
    initial_registered_images = len(state.registered_images)
    initial_points3d = int(state.points3d.shape[0])
    initial_observations = len(state.observations)
    failed_images: set[str] = set()
    last_global_ba_images = len(state.registered_images)
    last_global_ba_points = int(state.points3d.shape[0])
    last_global_ba_iteration = 0
    iterations = []

    for iteration in range(1, args.max_register + 1):
        state = load_reconstruction_state(sparse_dir)
        selected = select_next_image(
            images=runtime.image_paths,
            state=state,
            verified_dir=runtime.verified_dir,
            keypoints_by_name=runtime.keypoints_by_name,
            cameras=runtime.cameras,
            failed_images=failed_images,
            min_2d3d=args.min_2d3d,
            visibility_levels=args.visibility_levels,
        )
        if selected is None:
            break
        candidate, score = selected
        registration = register_image_pnp(
            candidate=candidate,
            camera=runtime.cameras[candidate.image_name],
            min_2d3d=args.min_2d3d,
            min_pnp_inliers=args.min_pnp_inliers,
            reproj_error_px=max_reproj_error,
            confidence=pnp_confidence,
        )
        record = {
            "iteration": iteration,
            "image_name": registration.image_name,
            "status": registration.status,
            "num_2d3d": registration.num_2d3d,
            "pnp_inliers": int(registration.pnp_inliers.shape[0]),
            "mean_reprojection_error": registration.mean_reprojection_error,
            "pyramid_visibility_score": score.pyramid_score,
            "registered_neighbor_count": score.registered_neighbor_count,
            "reports": [],
        }
        if not registration.success:
            failed_images.add(registration.image_name)
            iterations.append(record)
            continue

        add_registered_image(state, registration)
        save_reconstruction_state(state, sparse_dir)

        rt_reports = run_rt(args, stage_prefix=f"iter{iteration:02d}_{Path(registration.image_name).stem}_registered")
        record["reports"].extend(rt_reports)

        if args.local_ba_after_registration:
            record["reports"].extend(
                run_ba_filtering_cycle(
                    args=args,
                    runtime=runtime,
                    stage_prefix=f"iter{iteration:02d}_{Path(registration.image_name).stem}_local",
                    scope="local",
                    target_image=registration.image_name,
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
            min_interval=args.global_ba_min_interval,
            current_iteration=iteration,
            last_global_ba_iteration=last_global_ba_iteration,
            last_global_ba_images=last_global_ba_images,
            last_global_ba_points=last_global_ba_points,
        ):
            record["reports"].extend(
                run_global_refinement(args, runtime=runtime, stage_prefix=f"iter{iteration:02d}_growth_global")
            )
            state = load_reconstruction_state(sparse_dir)
            last_global_ba_images = len(state.registered_images)
            last_global_ba_points = int(state.points3d.shape[0])
            last_global_ba_iteration = iteration

        iterations.append(record)

    final_reports = []
    if args.run_final_refinement:
        final_reports = run_global_refinement(args, runtime=runtime, stage_prefix="final_global")

    final_state = load_reconstruction_state(sparse_dir)
    final_residual_report = "paper_aligned_final_registered_residual_report.json"
    write_registered_residual_report(args, runtime, final_residual_report)
    report = {
        "scene_name": scene["scene_name"],
        "initial_registered_images": initial_registered_images,
        "final_registered_images": len(final_state.registered_images),
        "initial_points3D": initial_points3d,
        "final_points3D": int(final_state.points3d.shape[0]),
        "initial_observations": initial_observations,
        "final_observations": len(final_state.observations),
        "max_register": args.max_register,
        "local_ba_after_registration": bool(args.local_ba_after_registration),
        "global_ba_growth_ratio": float(args.global_ba_growth_ratio),
        "global_ba_min_interval": int(args.global_ba_min_interval),
        "filter_after_ba": bool(args.filter_after_ba),
        "diagnose_degenerate_cameras": bool(args.diagnose_degenerate_cameras),
        "remove_degenerate_cameras": bool(args.remove_degenerate_cameras),
        "enable_recursive_track_splitting": bool(args.enable_recursive_track_splitting),
        "iterations": iterations,
        "final_refinement_reports": final_reports,
        "final_residual_report": final_residual_report,
    }
    report_path = report_dir / args.report_name
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Scene: {scene['scene_name']}")
    print(f"Registered images: {initial_registered_images} -> {len(final_state.registered_images)}")
    print(f"Points3D: {initial_points3d} -> {int(final_state.points3d.shape[0])}")
    print(f"Observations: {initial_observations} -> {len(final_state.observations)}")
    print(f"Iterations: {len(iterations)}")
    print(f"Report: {report_path}")
    return 0


def select_next_image(
    images: list[Path],
    state,
    verified_dir: Path,
    keypoints_by_name: dict[str, np.ndarray],
    cameras: dict[str, object],
    failed_images: set[str],
    min_2d3d: int,
    visibility_levels: int,
) -> tuple[Candidate2D3D, NextBestViewScore] | None:
    scored = []
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
        if candidate.num_correspondences < min_2d3d:
            continue
        score = score_next_best_view(
            candidate=candidate,
            camera=cameras[image_name],
            verified_dir=verified_dir,
            registered_image_names=set(state.registered_images),
            levels=visibility_levels,
        )
        scored.append((candidate, score))
    if not scored:
        return None
    scored.sort(key=lambda item: (item[1].pyramid_score, item[1].num_2d3d), reverse=True)
    return scored[0]


def run_rt(args: argparse.Namespace, stage_prefix: str, stage: str = "registered_rt") -> list[dict]:
    report_name = f"{stage_prefix}_rt_report.json"
    argv = [
        "triangulate_registered_tracks",
        "--scene",
        args.scene,
        "--stage",
        stage,
        "--report-name",
        report_name,
    ]
    if args.enable_recursive_track_splitting:
        argv.extend(["--enable-recursive-track-splitting", "--min-recursive-consensus-size", "3"])
    call_tool(triangulate_registered_tracks.main, argv)
    return [{"stage": stage_prefix, "type": "rt", "report_name": report_name}]


def run_global_refinement(args: argparse.Namespace, runtime: SfMRuntimeContext, stage_prefix: str) -> list[dict]:
    reports = []
    reports.extend(run_rt(args, stage_prefix=f"{stage_prefix}_pre", stage="pre_ba_rt"))
    reports.extend(
        run_ba_filtering_cycle(
            args=args,
            runtime=runtime,
            stage_prefix=f"{stage_prefix}_ba1",
            scope="global",
            target_image="",
            max_iterations=args.global_ba_max_iterations,
            max_points=args.global_ba_max_points,
            max_observations=args.global_ba_max_observations,
            local_neighbors=args.local_ba_neighbors,
        )
    )
    residual_report = report_path_for_scene(args.scene, f"{stage_prefix}_ba1_registered_residual_after_filtering_report.json")
    if not Path(residual_report).exists():
        residual_report = report_path_for_scene(args.scene, f"{stage_prefix}_ba1_registered_residual_report.json")
    reports.extend(run_rt_with_residual(args, stage_prefix=f"{stage_prefix}_post", residual_report=residual_report))
    reports.extend(
        run_ba_filtering_cycle(
            args=args,
            runtime=runtime,
            stage_prefix=f"{stage_prefix}_ba2",
            scope="global",
            target_image="",
            max_iterations=args.global_ba_max_iterations,
            max_points=args.global_ba_max_points,
            max_observations=args.global_ba_max_observations,
            local_neighbors=args.local_ba_neighbors,
        )
    )
    return reports


def run_rt_with_residual(args: argparse.Namespace, stage_prefix: str, residual_report: str) -> list[dict]:
    report_name = f"{stage_prefix}_rt_report.json"
    argv = [
        "triangulate_registered_tracks",
        "--scene",
        args.scene,
        "--stage",
        "post_ba_rt",
        "--residual-report",
        residual_report,
        "--max-observation-error",
        str(args.filter_max_reprojection_error),
        "--report-name",
        report_name,
    ]
    if args.enable_recursive_track_splitting:
        argv.extend(["--enable-recursive-track-splitting", "--min-recursive-consensus-size", "3"])
    call_tool(triangulate_registered_tracks.main, argv)
    return [{"stage": stage_prefix, "type": "post_ba_rt", "report_name": report_name}]


def run_ba_filtering_cycle(
    args: argparse.Namespace,
    runtime: SfMRuntimeContext,
    stage_prefix: str,
    scope: str,
    target_image: str,
    max_iterations: int,
    max_points: int,
    max_observations: int,
    local_neighbors: int,
) -> list[dict]:
    reports = []
    priority_report = f"{stage_prefix}_priority_residual_report.json"
    priority_report_path = write_registered_residual_report(
        args,
        runtime,
        priority_report,
        compact=True,
        write_npz=True,
    )
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
        "--priority-residual-report",
        priority_report_path,
        "--report-name",
        ba_report,
    ]
    if scope == "local":
        argv.extend(["--target-image", target_image, "--local-neighbors", str(local_neighbors)])
    call_tool(run_bundle_adjustment.main, argv)
    reports.append({"stage": stage_prefix, "type": "ba", "report_name": ba_report})

    residual_report = f"{stage_prefix}_registered_residual_report.json"
    residual_report_path = write_registered_residual_report(
        args,
        runtime,
        residual_report,
        compact=True,
        write_npz=True,
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
                residual_report_path,
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
        write_registered_residual_report(
            args,
            runtime,
            residual_for_degenerate,
            compact=True,
            write_npz=True,
        )
        reports.append(
            {"stage": stage_prefix, "type": "registered_residual_after_filtering", "report_name": residual_for_degenerate}
        )

    if args.diagnose_degenerate_cameras or args.remove_degenerate_cameras:
        degenerate_report = f"{stage_prefix}_degenerate_camera_report.json"
        argv = [
            "evaluate_degenerate_cameras",
            "--scene",
            args.scene,
            "--residual-report",
            report_path_for_scene(args.scene, residual_for_degenerate),
            "--report-name",
            degenerate_report,
        ]
        if args.remove_degenerate_cameras:
            argv.append("--remove-candidates")
        call_tool(evaluate_degenerate_cameras.main, argv)
        reports.append({"stage": stage_prefix, "type": "degenerate_cameras", "report_name": degenerate_report})
    return reports


def should_run_global_ba(
    state,
    growth_ratio: float,
    min_interval: int,
    current_iteration: int,
    last_global_ba_iteration: int,
    last_global_ba_images: int,
    last_global_ba_points: int,
) -> bool:
    if growth_ratio <= 0:
        return False
    if min_interval > 0 and current_iteration - last_global_ba_iteration < min_interval:
        return False
    current_images = len(state.registered_images)
    current_points = int(state.points3d.shape[0])
    image_trigger = current_images >= max(last_global_ba_images + 1, int(np.ceil(last_global_ba_images * growth_ratio)))
    point_trigger = current_points >= max(last_global_ba_points + 1, int(np.ceil(last_global_ba_points * growth_ratio)))
    return image_trigger or point_trigger


def report_path_for_scene(scene_path: str, report_name: str) -> str:
    config = load_scene_config(scene_path)
    report_dir = resolve_project_path(config["scene"]["output_dir"]) / "reports"
    return str(report_dir / report_name)


def write_registered_residual_report(
    args: argparse.Namespace,
    runtime: SfMRuntimeContext,
    report_name: str,
    compact: bool = False,
    write_npz: bool = False,
) -> str:
    state = load_reconstruction_state(runtime.sparse_dir)
    observations, errors, summary = evaluate_registered_residuals_in_memory(
        state=state,
        cameras=runtime.cameras,
        keypoints_by_name=runtime.keypoints_by_name,
    )
    report_path = runtime.report_dir / report_name
    npz_name = None
    if write_npz:
        npz_name = f"{report_path.stem}_errors.npz"
        write_residual_npz(runtime.report_dir / npz_name, observations, errors)
    write_residual_report(
        report_path=report_path,
        scene_name=runtime.config["scene"]["scene_name"],
        observations=observations,
        errors=errors,
        summary=summary,
        compact=compact,
        npz_name=npz_name,
    )
    return str(report_path)


def call_tool(main_func, argv: list[str]) -> None:
    old_argv = sys.argv
    try:
        sys.argv = argv
        exit_code = main_func()
        if exit_code not in (0, None):
            raise RuntimeError(f"Tool failed with exit code {exit_code}: {' '.join(argv)}")
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    raise SystemExit(main())
