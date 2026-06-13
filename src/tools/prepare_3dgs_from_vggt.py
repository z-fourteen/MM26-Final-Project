from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from src.adapters.vggt_adapter import ensure_vggt_importable
from src.sfm.config import load_scene_config, resolve_project_path
from src.tools import check_3dgs_scene


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare a 3DGS source directory from VGGT predictions.")
    parser.add_argument("--scene", required=True, help="Path to configs/scenes/<scene>.yaml")
    parser.add_argument("--predictions", required=True, help="VGGT predictions.npz with extrinsic/intrinsic arrays")
    parser.add_argument("--points", required=True, help="VGGT points npz with xyz/rgb[/confidence] arrays")
    parser.add_argument("--output-name", default="", help="Defaults to <scene_name>_vggt")
    parser.add_argument("--output-root", default="data/3dgs_inputs")
    parser.add_argument("--mode", choices=["pad", "crop"], default="pad", help="VGGT preprocessing mode")
    parser.add_argument("--max-points", type=int, default=200000)
    parser.add_argument("--copy-originals", action="store_true")
    parser.add_argument("--skip-check", action="store_true")
    args = parser.parse_args()

    config = load_scene_config(args.scene)
    scene = config["scene"]
    scene_name = str(scene["scene_name"])
    output_name = args.output_name or f"{scene_name}_vggt"
    output_dir = resolve_project_path(args.output_root) / output_name
    images_dir = output_dir / "images"
    sparse_dir = output_dir / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    predictions = np.load(resolve_project_path(args.predictions))
    points = np.load(resolve_project_path(args.points))
    source_image_dir = resolve_project_path(scene["image_dir"])
    image_paths = list_prediction_images(predictions, source_image_dir)
    extrinsics = np.asarray(predictions["extrinsic"], dtype=np.float64)
    intrinsics = np.asarray(predictions["intrinsic"], dtype=np.float64)
    if len(image_paths) != len(extrinsics):
        raise ValueError(f"Image count {len(image_paths)} != VGGT camera count {len(extrinsics)}")
    if len(intrinsics) != len(extrinsics):
        raise ValueError(f"Intrinsic count {len(intrinsics)} != extrinsic count {len(extrinsics)}")

    image_names, height, width = save_preprocessed_images(image_paths, images_dir, args.mode)
    write_cameras_txt(sparse_dir / "cameras.txt", intrinsics, width, height)
    write_images_txt(sparse_dir / "images.txt", extrinsics, image_names)
    write_points3d_txt(sparse_dir / "points3D.txt", points, args.max_points)

    if args.copy_originals:
        original_dir = output_dir / "original_images"
        original_dir.mkdir(parents=True, exist_ok=True)
        for image_path in image_paths:
            shutil.copy2(image_path, original_dir / image_path.name)

    report_path = resolve_project_path(scene["output_dir"]) / "reports" / f"{output_name}_3dgs_scene_check.json"
    if not args.skip_check:
        call_tool(
            check_3dgs_scene.main,
            [
                "check_3dgs_scene",
                "--source-path",
                str(output_dir),
                "--write-ply",
                "--report-name",
                str(report_path),
            ],
        )

    print(f"3DGS VGGT source: {output_dir}")
    print(
        "Train command: python third_party/gaussian-splatting/train.py "
        f"-s {output_dir.as_posix()} -m outputs/3dgs/{output_name}"
    )
    return 0


def list_images(image_dir: Path) -> list[Path]:
    paths = sorted(path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    if not paths:
        raise ValueError(f"No images found in {image_dir}")
    return paths


def list_prediction_images(predictions, fallback_image_dir: Path) -> list[Path]:
    if "image_paths" not in predictions:
        return list_images(fallback_image_dir)
    paths = [Path(str(path)) for path in predictions["image_paths"].tolist()]
    resolved = []
    for path in paths:
        if path.is_absolute() and path.exists():
            resolved.append(path)
            continue
        fallback = fallback_image_dir / path.name
        if fallback.exists():
            resolved.append(fallback)
            continue
        candidate = resolve_project_path(path)
        if candidate.exists():
            resolved.append(candidate)
            continue
        raise FileNotFoundError(f"VGGT prediction image does not exist: {path}")
    if not resolved:
        raise ValueError("VGGT predictions contain no image paths")
    return resolved


def save_preprocessed_images(image_paths: list[Path], output_dir: Path, mode: str) -> tuple[list[str], int, int]:
    ensure_vggt_importable()
    try:
        from vggt.utils.load_fn import load_and_preprocess_images
    except Exception as exc:
        raise RuntimeError(
            "VGGT is not importable; run this tool in an environment with "
            "VGGT dependencies installed and third_party/vggt present"
        ) from exc

    images = load_and_preprocess_images([str(path) for path in image_paths], mode=mode)
    output_dir.mkdir(parents=True, exist_ok=True)
    names = []
    for idx, image_tensor in enumerate(images):
        name = f"{idx:06d}.png"
        array = image_tensor.permute(1, 2, 0).numpy()
        array = (array * 255.0).clip(0, 255).astype(np.uint8)
        Image.fromarray(array).save(output_dir / name)
        names.append(name)
    return names, int(images.shape[-2]), int(images.shape[-1])


def write_cameras_txt(path: Path, intrinsics: np.ndarray, width: int, height: int) -> None:
    lines = [
        "# Camera list with one line of data per camera:",
        "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]",
        f"# Number of cameras: {len(intrinsics)}",
    ]
    for idx, k in enumerate(intrinsics, start=1):
        fx, fy, cx, cy = float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2])
        lines.append(f"{idx} PINHOLE {width} {height} {fx:.12g} {fy:.12g} {cx:.12g} {cy:.12g}")
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def write_images_txt(path: Path, extrinsics: np.ndarray, image_names: list[str]) -> None:
    lines = [
        "# Image list with two lines of data per image:",
        "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
        "#   POINTS2D[] as (X, Y, POINT3D_ID)",
        f"# Number of images: {len(image_names)}, mean observations per image: 0",
    ]
    for idx, (ext, name) in enumerate(zip(extrinsics, image_names), start=1):
        q = rotmat_to_colmap_qvec(ext[:3, :3])
        t = ext[:3, 3]
        lines.append(
            f"{idx} {q[0]:.17g} {q[1]:.17g} {q[2]:.17g} {q[3]:.17g} "
            f"{t[0]:.17g} {t[1]:.17g} {t[2]:.17g} {idx} {name}"
        )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def write_points3d_txt(path: Path, points_npz, max_points: int) -> None:
    xyz = np.asarray(points_npz["xyz"], dtype=np.float64)
    rgb = np.asarray(points_npz["rgb"], dtype=np.uint8)
    conf = (
        np.asarray(points_npz["confidence"], dtype=np.float64)
        if "confidence" in points_npz
        else np.ones((len(xyz),), dtype=np.float64)
    )
    if max_points > 0 and len(xyz) > max_points:
        keep = np.argsort(conf)[-max_points:]
        xyz = xyz[keep]
        rgb = rgb[keep]

    lines = [
        "# 3D point list with one line of data per point:",
        "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)",
        f"# Number of points: {len(xyz)}, mean track length: 0",
    ]
    for idx, (point, color) in enumerate(zip(xyz, rgb), start=1):
        lines.append(
            f"{idx} {point[0]:.17g} {point[1]:.17g} {point[2]:.17g} "
            f"{int(color[0])} {int(color[1])} {int(color[2])} 1.0"
        )
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def rotmat_to_colmap_qvec(rot: np.ndarray) -> np.ndarray:
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
    norm = np.linalg.norm(q)
    if norm > 0:
        q /= norm
    return q


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
