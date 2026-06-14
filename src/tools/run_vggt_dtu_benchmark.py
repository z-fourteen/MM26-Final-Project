#!/usr/bin/env python3
"""Run VGGT reconstruction for one DTU scan on a V100-class GPU.

Modes:
  - Default: depth unprojection → point cloud → PLY
  - --use-point-head: VGGT world_points → PLY
  - --use-ba: VGGT → predict_tracks → COLMAP BA → refined point cloud → PLY
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

from src.adapters.vggt_adapter import ensure_vggt_importable

ensure_vggt_importable()

from vggt.models.vggt import VGGT
from vggt.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--confidence-threshold", type=float, default=5.0)
    parser.add_argument("--use-point-head", action="store_true",
                        help="Use VGGT Point Head instead of depth unprojection")
    parser.add_argument("--use-ba", action="store_true",
                        help="Run COLMAP bundle adjustment after VGGT inference")
    parser.add_argument("--ba-max-reproj-error", type=float, default=4.0,
                        help="Max reprojection error for BA track filtering")
    parser.add_argument("--ba-shared-camera", action="store_true", default=True,
                        help="Share camera intrinsics across all images in BA")
    parser.add_argument("--ba-vis-thresh", type=float, default=0.2,
                        help="Visibility threshold for track prediction")
    parser.add_argument("--ba-max-query-pts", type=int, default=4096,
                        help="Max query points for track prediction")
    parser.add_argument("--ba-query-frame-num", type=int, default=8,
                        help="Number of query frames for track prediction")
    parser.add_argument("--ba-fine-tracking", action="store_true", default=False,
                        help="Use fine tracking (more accurate but much more memory)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_rgb_images(image_dir: Path, target_size: int = 518) -> tuple[torch.Tensor, np.ndarray, list[Path]]:
    paths = sorted(
        path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not paths:
        raise ValueError(f"No images found in {image_dir}")

    tensors = []
    valid_masks = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        width, height = image.size
        if width >= height:
            new_width = target_size
            new_height = round(height * target_size / width / 14) * 14
        else:
            new_height = target_size
            new_width = round(width * target_size / height / 14) * 14
        image = image.resize((new_width, new_height), Image.Resampling.BICUBIC)
        tensor = pil_to_tensor(image).float().div_(255.0)
        pad_h = target_size - new_height
        pad_w = target_size - new_width
        top, left = pad_h // 2, pad_w // 2
        tensor = torch.nn.functional.pad(
            tensor,
            (left, pad_w - left, top, pad_h - top),
            mode="constant",
            value=1.0,
        )
        valid = np.zeros((target_size, target_size), dtype=bool)
        valid[top: top + new_height, left: left + new_width] = True
        tensors.append(tensor)
        valid_masks.append(valid)
    return torch.stack(tensors), np.stack(valid_masks), paths


def write_binary_ply(path: Path, points: np.ndarray, colors: np.ndarray | None = None) -> None:
    points = np.asarray(points, dtype="<f4")
    if colors is None:
        colors = np.full((len(points), 3), 200, dtype=np.uint8)
    colors = np.asarray(colors, dtype=np.uint8)
    dtype = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ])
    vertices = np.empty(len(points), dtype=dtype)
    for i, name in enumerate(("x", "y", "z")):
        vertices[name] = points[:, i]
    for i, name in enumerate(("red", "green", "blue")):
        vertices[name] = colors[:, i]
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(vertices)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    with path.open("wb") as handle:
        handle.write(header.encode("ascii"))
        vertices.tofile(handle)


def write_cameras(path: Path, image_paths: list[Path], extrinsic: np.ndarray, intrinsic: np.ndarray) -> None:
    camera_to_world = closed_form_inverse_se3(torch.from_numpy(extrinsic))[:, :3].numpy()
    records = []
    for index, image_path in enumerate(image_paths):
        records.append({
            "index": index,
            "image": image_path.name,
            "camera_center_world": camera_to_world[index, :3, 3].tolist(),
            "world_from_camera_3x4": camera_to_world[index].tolist(),
            "camera_from_world_3x4": extrinsic[index, :3, :4].tolist(),
            "intrinsic_3x3": intrinsic[index].tolist(),
        })
    path.write_text(
        json.dumps({"camera_convention": "OpenCV world-to-camera", "cameras": records}, indent=2),
        encoding="utf-8",
    )


def run_colmap_ba(
    depth_conf: np.ndarray,
    points_3d: np.ndarray,
    extrinsic_np: np.ndarray,
    intrinsic_np: np.ndarray,
    images_track: torch.Tensor,
    valid_mask: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
    output_dir: Path,
    image_paths: list[Path],
) -> tuple[np.ndarray, np.ndarray | None]:
    """Run COLMAP bundle adjustment on VGGT outputs.

    Returns (refined_points_3d, colors) or raises on failure.
    """
    import pycolmap
    from vggt.dependency.track_predict import predict_tracks
    from vggt.dependency.np_to_pycolmap import batch_np_matrix_to_pycolmap, pycolmap_to_batch_np_matrix

    track_res = 518
    images_dev = images_track.to(device)
    image_size = np.array([track_res, track_res])

    print(f"Predicting tracks (query_frame_num={args.ba_query_frame_num}, max_query_pts={args.ba_max_query_pts})...")
    track_started = time.time()
    with torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
        pred_tracks, pred_vis_scores, pred_confs, points_3d_for_ba, points_rgb = predict_tracks(
            images_dev,
            conf=depth_conf,
            points_3d=points_3d,
            masks=None,
            max_query_pts=args.ba_max_query_pts,
            query_frame_num=args.ba_query_frame_num,
            keypoint_extractor="aliked+sp",
            fine_tracking=args.ba_fine_tracking,
        )
    torch.cuda.empty_cache()
    print(f"Tracks predicted in {time.time() - track_started:.1f}s")

    track_mask = pred_vis_scores > args.ba_vis_thresh

    print(f"Building pycolmap reconstruction (max_reproj_error={args.ba_max_reproj_error})...")
    reconstruction, valid_track_mask = batch_np_matrix_to_pycolmap(
        points_3d_for_ba,
        extrinsic_np,
        intrinsic_np,  # use 518 intrinsics directly
        pred_tracks,
        image_size,
        masks=track_mask,
        max_reproj_error=args.ba_max_reproj_error,
        shared_camera=args.ba_shared_camera,
        camera_type="SIMPLE_PINHOLE",
        points_rgb=points_rgb,
    )

    if reconstruction is None:
        raise RuntimeError("BA reconstruction is None — no valid tracks after filtering")

    print(f"Running COLMAP bundle adjustment (Ceres, shared_camera={args.ba_shared_camera})...")
    ba_started = time.time()
    ba_options = pycolmap.BundleAdjustmentOptions()
    ba_options.ceres.loss_function_type = pycolmap.LossFunctionType.CAUCHY
    ba_options.ceres.solver_options.max_num_iterations = 100
    ba_options.ceres.solver_options.num_threads = -1
    ba_options.print_summary = True
    pycolmap.bundle_adjustment(reconstruction, ba_options)
    print(f"BA completed in {time.time() - ba_started:.1f}s")

    # Extract refined points and cameras
    refined_pts, refined_extrinsic, refined_intrinsic, _ = pycolmap_to_batch_np_matrix(
        reconstruction, device="cpu", camera_type="SIMPLE_PINHOLE"
    )

    # Build colors from reconstruction points
    num_pts = len(refined_pts)
    if num_pts > 0:
        ba_colors = np.full((num_pts, 3), 200, dtype=np.uint8)
    else:
        ba_colors = None

    # Save refined cameras
    write_cameras(output_dir / "cameras_ba.json", image_paths, refined_extrinsic, refined_intrinsic)

    # Save BA metadata
    ba_meta = {
        "method": "VGGT + COLMAP BA (Ceres, Cauchy loss)",
        "ba_points": num_pts,
        "ba_shared_camera": args.ba_shared_camera,
        "ba_max_reproj_error": args.ba_max_reproj_error,
        "ba_vis_thresh": args.ba_vis_thresh,
    }
    (output_dir / "ba_meta.json").write_text(json.dumps(ba_meta, indent=2))
    print(json.dumps(ba_meta, indent=2))

    return refined_pts, ba_colors


def main() -> int:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    scan_dir = Path(args.scan_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths_all = sorted(
        path for path in (scan_dir / "images").iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not image_paths_all:
        raise ValueError(f"No images found in {scan_dir / 'images'}")

    # Load images at 518 for VGGT inference
    images_518, valid_mask, image_paths = load_rgb_images(scan_dir / "images", target_size=518)
    device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    images_518 = images_518.to(device)
    print(f"scan={scan_dir.name} images={len(image_paths)} tensor={tuple(images_518.shape)} dtype={dtype}")

    started = time.time()
    enable_track = args.use_ba
    model = VGGT(enable_point=args.use_point_head, enable_track=enable_track)
    state = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    invalid_unexpected = [
        key for key in unexpected
        if not key.startswith(("point_head.", "track_head.")) and key != "point_head"
    ]
    if missing or invalid_unexpected:
        raise RuntimeError(f"Checkpoint mismatch: missing={missing}, unexpected={invalid_unexpected}")
    model.eval().to(device)

    with torch.inference_mode(), torch.amp.autocast(
        device_type=device.type, dtype=dtype, enabled=device.type == "cuda"
    ):
        predictions = model(images_518)
        pose_encoding = predictions["pose_enc"]
        depth = predictions["depth"]
        depth_confidence = predictions["depth_conf"]
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_encoding, images_518.shape[-2:])

    pose_np = pose_encoding.squeeze(0).float().cpu().numpy()
    depth_np = depth.squeeze(0).float().cpu().numpy()
    confidence_np = depth_confidence.squeeze(0).float().cpu().numpy()
    extrinsic_np = extrinsic.squeeze(0).float().cpu().numpy()
    intrinsic_np = intrinsic.squeeze(0).float().cpu().numpy()

    # Save predictions.npz (always)
    np.savez_compressed(
        output_dir / "predictions.npz",
        pose_enc=pose_np,
        depth=depth_np,
        depth_conf=confidence_np,
        extrinsic=extrinsic_np,
        intrinsic=intrinsic_np,
        image_paths=np.asarray([path.name for path in image_paths]),
        valid_mask=valid_mask,
    )

    # --- Choose output path based on mode ---
    if args.use_ba:
        # Free VGGT model to make room for track predictor
        del model
        torch.cuda.empty_cache()
        print(f"GPU memory after VGGT cleanup: {torch.cuda.max_memory_allocated(device) / 2**30:.2f} GiB")

        # Depth-derived points for BA input
        points_for_ba = unproject_depth_map_to_point_map(depth_np, extrinsic_np, intrinsic_np)

        ba_pts, ba_colors = run_colmap_ba(
            depth_conf=confidence_np,
            points_3d=points_for_ba,
            extrinsic_np=extrinsic_np,
            intrinsic_np=intrinsic_np,
            images_track=images_518,  # use 518 images for tracking (V100 memory limit)
            valid_mask=valid_mask,
            args=args, device=device, dtype=dtype,
            output_dir=output_dir, image_paths=image_paths,
        )

        output_ply = output_dir / "points_ba.ply"
        write_binary_ply(output_ply, ba_pts, ba_colors)
        num_points = len(ba_pts)
    elif args.use_point_head:
        world_points = predictions["world_points"].squeeze(0).float().cpu().numpy()
        world_conf = predictions["world_points_conf"].squeeze(0).float().cpu().numpy()
        keep = valid_mask & np.isfinite(world_points).all(axis=-1) & (world_conf >= args.confidence_threshold)
        points = world_points
        output_ply = output_dir / "points_pointhead.ply"
        colors = images_518.float().cpu().numpy().transpose(0, 2, 3, 1)
        colors = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
        write_binary_ply(output_ply, points[keep], colors[keep])
        num_points = int(keep.sum())
    else:
        points = unproject_depth_map_to_point_map(depth_np, extrinsic_np, intrinsic_np)
        keep = valid_mask & np.isfinite(points).all(axis=-1) & (confidence_np >= args.confidence_threshold)
        output_ply = output_dir / "points_depth.ply"
        colors = images_518.float().cpu().numpy().transpose(0, 2, 3, 1)
        colors = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
        write_binary_ply(output_ply, points[keep], colors[keep])
        num_points = int(keep.sum())

    write_cameras(output_dir / "cameras.json", image_paths, extrinsic_np, intrinsic_np)

    metadata = {
        "scan": scan_dir.name,
        "images": len(image_paths),
        "mode": "ba" if args.use_ba else ("point_head" if args.use_point_head else "depth"),
        "confidence_threshold": args.confidence_threshold,
        "points": num_points,
        "seconds": time.time() - started,
        "checkpoint": str(Path(args.ckpt).resolve()),
        "preprocess": "RGB ignoring alpha; aspect-preserving resize and white pad to 518",
    }
    (output_dir / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    if device.type == "cuda":
        print(f"cuda_max_allocated_gib={torch.cuda.max_memory_allocated(device) / 2**30:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
