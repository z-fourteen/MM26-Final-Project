#!/usr/bin/env python3
"""Prepare a COLMAP-text style dataset from VGGT inference outputs.

The resulting folder can be used by 3D Gaussian Splatting implementations that
read COLMAP cameras/images/points plus an image folder. We export preprocessed
518x518 images, because VGGT's predicted intrinsics are in that image space.
"""

import argparse
import glob
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from vggt.utils.load_fn import load_and_preprocess_images


def parse_args():
    parser = argparse.ArgumentParser(description="Convert VGGT outputs to a COLMAP-text 3DGS dataset.")
    parser.add_argument("--image_folder", required=True, help="Original input image folder used for VGGT.")
    parser.add_argument("--predictions", required=True, help="VGGT predictions.npz.")
    parser.add_argument("--points", required=True, help="VGGT points_depth.npz.")
    parser.add_argument("--output_dir", required=True, help="Output dataset directory.")
    parser.add_argument("--mode", choices=["pad", "crop"], default="pad", help="Same preprocessing mode as inference.")
    parser.add_argument("--max_points", type=int, default=200000, help="Limit exported sparse points.")
    parser.add_argument("--copy_originals", action="store_true", help="Also copy original images to original_images/.")
    return parser.parse_args()


def list_images(image_folder):
    exts = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp")
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(image_folder, ext)))
    paths = sorted(paths)
    if not paths:
        raise ValueError(f"No images found in {image_folder}")
    return paths


def rotmat_to_colmap_qvec(rot):
    """Convert world-to-camera rotation matrix to COLMAP qvec [qw, qx, qy, qz]."""
    m = np.asarray(rot, dtype=np.float64)
    trace = np.trace(m)
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        qw = (m[2, 1] - m[1, 2]) / s
        qx = 0.25 * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        qw = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qy = 0.25 * s
        qz = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        qw = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
        qz = 0.25 * s
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    return q / np.linalg.norm(q)


def save_preprocessed_images(image_paths, output_dir, mode):
    images = load_and_preprocess_images(image_paths, mode=mode)
    output_dir.mkdir(parents=True, exist_ok=True)
    names = []
    for idx, image_tensor in enumerate(images):
        name = f"{idx:06d}.png"
        array = image_tensor.permute(1, 2, 0).numpy()
        array = (array * 255.0).clip(0, 255).astype(np.uint8)
        Image.fromarray(array).save(output_dir / name)
        names.append(name)
    return names, images.shape[-2], images.shape[-1]


def write_cameras_txt(path, intrinsics, width, height):
    with open(path, "w", encoding="ascii") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write("# Number of cameras: {}\n".format(len(intrinsics)))
        for idx, k in enumerate(intrinsics, start=1):
            fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
            f.write(f"{idx} PINHOLE {width} {height} {fx} {fy} {cx} {cy}\n")


def write_images_txt(path, extrinsics, image_names):
    with open(path, "w", encoding="ascii") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write("# Number of images: {}, mean observations per image: 0\n".format(len(image_names)))
        for idx, (ext, name) in enumerate(zip(extrinsics, image_names), start=1):
            q = rotmat_to_colmap_qvec(ext[:3, :3])
            t = ext[:3, 3]
            f.write(
                f"{idx} {q[0]} {q[1]} {q[2]} {q[3]} "
                f"{t[0]} {t[1]} {t[2]} {idx} {name}\n\n"
            )


def write_points3d_txt(path, points_npz, max_points):
    xyz = points_npz["xyz"]
    rgb = points_npz["rgb"]
    conf = points_npz["confidence"] if "confidence" in points_npz else np.ones(len(xyz), dtype=np.float32)

    if max_points is not None and len(xyz) > max_points:
        # Keep the highest-confidence points for a cleaner 3DGS initialization.
        keep = np.argsort(conf)[-max_points:]
        xyz = xyz[keep]
        rgb = rgb[keep]

    with open(path, "w", encoding="ascii") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write("# Number of points: {}, mean track length: 0\n".format(len(xyz)))
        for idx, (p, c) in enumerate(zip(xyz, rgb), start=1):
            f.write(f"{idx} {p[0]} {p[1]} {p[2]} {int(c[0])} {int(c[1])} {int(c[2])} 1.0\n")


def main():
    args = parse_args()
    image_paths = list_images(args.image_folder)
    predictions = np.load(args.predictions)
    points = np.load(args.points)

    extrinsics = predictions["extrinsic"]
    intrinsics = predictions["intrinsic"]
    if len(image_paths) != len(extrinsics):
        raise ValueError(f"Image count {len(image_paths)} != camera count {len(extrinsics)}")

    output_dir = Path(args.output_dir)
    images_dir = output_dir / "images"
    sparse_dir = output_dir / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    image_names, height, width = save_preprocessed_images(image_paths, images_dir, args.mode)
    write_cameras_txt(sparse_dir / "cameras.txt", intrinsics, width, height)
    write_images_txt(sparse_dir / "images.txt", extrinsics, image_names)
    write_points3d_txt(sparse_dir / "points3D.txt", points, args.max_points)

    if args.copy_originals:
        original_dir = output_dir / "original_images"
        original_dir.mkdir(parents=True, exist_ok=True)
        for image_path in image_paths:
            shutil.copy2(image_path, original_dir / os.path.basename(image_path))

    print(f"Saved 3DGS/COLMAP-text dataset to: {output_dir}")
    print(f"Images: {images_dir}")
    print(f"COLMAP text model: {sparse_dir}")


if __name__ == "__main__":
    main()
