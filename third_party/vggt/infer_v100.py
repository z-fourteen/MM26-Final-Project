#!/usr/bin/env python3
"""Minimal VGGT inference runner for V100-class GPUs.

This avoids demo-only dependencies such as gradio, viser, pycolmap, trimesh,
and LightGlue. It writes camera matrices, dense predictions, and a simple PLY
point cloud from depth unprojection.
"""

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download

from vggt.models.vggt import VGGT
from vggt.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


def parse_args():
    parser = argparse.ArgumentParser(description="Run VGGT inference without demo dependencies.")
    parser.add_argument("--image_folder", required=True, help="Folder containing input images.")
    parser.add_argument("--output_dir", default="outputs/vggt_infer", help="Directory for predictions.")
    parser.add_argument("--ckpt", default=None, help="Path to model.pt. If omitted, downloads facebook/VGGT-1B.")
    parser.add_argument("--mode", choices=["pad", "crop"], default="pad", help="Image preprocessing mode.")
    parser.add_argument("--max_images", type=int, default=None, help="Optionally limit the number of images.")
    parser.add_argument("--conf_percentile", type=float, default=70.0, help="Keep points above this confidence percentile.")
    parser.add_argument("--device", default="cuda", help="Use cuda for V100 inference.")
    return parser.parse_args()


def list_images(image_folder, max_images=None):
    exts = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp")
    image_paths = []
    for ext in exts:
        image_paths.extend(glob.glob(os.path.join(image_folder, ext)))
    image_paths = sorted(image_paths)
    if max_images is not None:
        image_paths = image_paths[:max_images]
    if not image_paths:
        raise ValueError(f"No images found in {image_folder}")
    return image_paths


def load_model(ckpt, device):
    if ckpt is None:
        ckpt = hf_hub_download(repo_id="facebook/VGGT-1B", filename="model.pt")

    model = VGGT(enable_track=False)
    state = torch.load(ckpt, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded checkpoint: {ckpt}")
    print(f"Missing keys: {len(missing)}; unexpected keys: {len(unexpected)}")
    model.eval().to(device)
    return model


def write_ply(path, points, colors):
    points = points.reshape(-1, 3).astype(np.float32)
    colors = colors.reshape(-1, 3).astype(np.uint8)
    with open(path, "w", encoding="ascii") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(points, colors):
            f.write(f"{p[0]} {p[1]} {p[2]} {c[0]} {c[1]} {c[2]}\n")


def write_camera_json(path, image_paths, extrinsic, intrinsic):
    cam_to_world = closed_form_inverse_se3(extrinsic)[:, :3, :]
    cameras = []
    for idx, image_path in enumerate(image_paths):
        cameras.append(
            {
                "index": idx,
                "image": os.path.basename(image_path),
                "camera_center_world": cam_to_world[idx, :3, 3].astype(float).tolist(),
                "world_from_camera_3x4": cam_to_world[idx].astype(float).tolist(),
                "camera_from_world_3x4": extrinsic[idx].astype(float).tolist(),
                "intrinsic_3x3": intrinsic[idx].astype(float).tolist(),
            }
        )
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"camera_convention": "OpenCV; camera_from_world extrinsic", "cameras": cameras}, f, indent=2)


def write_camera_csv(path, image_paths, extrinsic):
    cam_to_world = closed_form_inverse_se3(extrinsic)[:, :3, :]
    with open(path, "w", encoding="ascii") as f:
        f.write("index,image,camera_center_world_x,camera_center_world_y,camera_center_world_z\n")
        for idx, image_path in enumerate(image_paths):
            c = cam_to_world[idx, :3, 3]
            f.write(f"{idx},{os.path.basename(image_path)},{c[0]},{c[1]},{c[2]}\n")


def write_pose_encoding_csv(path, image_paths, pose_enc, intrinsic):
    with open(path, "w", encoding="ascii") as f:
        f.write(
            "index,image,tx,ty,tz,qx,qy,qz,qw,fov_h_rad,fov_w_rad,fx,fy,cx,cy\n"
        )
        for idx, image_path in enumerate(image_paths):
            pose = pose_enc[idx]
            k = intrinsic[idx]
            f.write(
                f"{idx},{os.path.basename(image_path)},"
                f"{pose[0]},{pose[1]},{pose[2]},"
                f"{pose[3]},{pose[4]},{pose[5]},{pose[6]},"
                f"{pose[7]},{pose[8]},"
                f"{k[0, 0]},{k[1, 1]},{k[0, 2]},{k[1, 2]}\n"
            )


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    dtype = torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
    print(f"Device: {torch.cuda.get_device_name(0)}; dtype: {dtype}")

    image_paths = list_images(args.image_folder, args.max_images)
    print(f"Found {len(image_paths)} images")
    images = load_and_preprocess_images(image_paths, mode=args.mode).to(device)
    print(f"Input tensor: {tuple(images.shape)}")

    model = load_model(args.ckpt, device)
    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=dtype):
            predictions = model(images)

    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])

    pred_np = {}
    for key in ("pose_enc", "depth", "depth_conf", "world_points", "world_points_conf"):
        pred_np[key] = predictions[key].detach().cpu().numpy().squeeze(0)
    pred_np["extrinsic"] = extrinsic.detach().cpu().numpy().squeeze(0)
    pred_np["intrinsic"] = intrinsic.detach().cpu().numpy().squeeze(0)

    points_from_depth = unproject_depth_map_to_point_map(
        pred_np["depth"], pred_np["extrinsic"], pred_np["intrinsic"]
    )
    pred_np["world_points_from_depth"] = points_from_depth
    pred_np["image_paths"] = np.array(image_paths)

    np.savez_compressed(output_dir / "predictions.npz", **pred_np)

    colors = images.detach().cpu().numpy().transpose(0, 2, 3, 1)
    colors = (colors * 255).clip(0, 255).astype(np.uint8)
    conf = pred_np["depth_conf"]
    threshold = np.percentile(conf.reshape(-1), args.conf_percentile)
    mask = conf >= threshold
    filtered_points = points_from_depth[mask]
    filtered_colors = colors[mask]
    filtered_conf = conf[mask]
    filtered_frame_idx = np.broadcast_to(
        np.arange(points_from_depth.shape[0])[:, None, None], points_from_depth.shape[:3]
    )[mask]

    write_ply(output_dir / "points_depth.ply", filtered_points, filtered_colors)
    np.savez_compressed(
        output_dir / "points_depth.npz",
        xyz=filtered_points.astype(np.float32),
        rgb=filtered_colors.astype(np.uint8),
        confidence=filtered_conf.astype(np.float32),
        frame_idx=filtered_frame_idx.astype(np.int32),
    )
    write_camera_json(output_dir / "cameras.json", image_paths, pred_np["extrinsic"], pred_np["intrinsic"])
    write_camera_csv(output_dir / "camera_centers.csv", image_paths, pred_np["extrinsic"])
    write_pose_encoding_csv(output_dir / "camera_pose_paper_format.csv", image_paths, pred_np["pose_enc"], pred_np["intrinsic"])

    print(f"Saved: {output_dir / 'predictions.npz'}")
    print(f"Saved: {output_dir / 'points_depth.ply'}")
    print(f"Saved: {output_dir / 'points_depth.npz'}")
    print(f"Saved: {output_dir / 'cameras.json'}")
    print(f"Saved: {output_dir / 'camera_centers.csv'}")
    print(f"Saved: {output_dir / 'camera_pose_paper_format.csv'}")
    print(f"CUDA max allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")


if __name__ == "__main__":
    main()
