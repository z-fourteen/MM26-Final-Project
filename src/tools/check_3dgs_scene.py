from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


SUPPORTED_CAMERA_MODELS = {"PINHOLE"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class CameraRecord:
    camera_id: int
    model: str
    width: int
    height: int
    params: tuple[float, ...]


@dataclass(frozen=True)
class ImageRecord:
    image_id: int
    camera_id: int
    image_name: str
    num_points2d: int
    num_linked_points3d: int


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate an SfM scene layout for 3DGS COLMAP loading.")
    parser.add_argument("--source-path", required=True, help="Scene root passed to 3DGS train.py -s")
    parser.add_argument("--images", default="images", help="Image subdirectory passed to 3DGS --images")
    parser.add_argument("--write-ply", action="store_true", help="Write sparse/0/points3D.ply if it is missing")
    parser.add_argument("--report-name", default="", help="Optional JSON report path")
    args = parser.parse_args()

    source_path = Path(args.source_path)
    sparse_dir = source_path / "sparse" / "0"
    images_dir = source_path / args.images
    errors: list[str] = []
    warnings: list[str] = []

    if not source_path.exists():
        errors.append(f"source path does not exist: {source_path}")
    if not images_dir.exists():
        errors.append(f"images directory does not exist: {images_dir}")
    if not sparse_dir.exists():
        errors.append(f"sparse/0 directory does not exist: {sparse_dir}")

    cameras_path = sparse_dir / "cameras.txt"
    images_path = sparse_dir / "images.txt"
    points_path = sparse_dir / "points3D.txt"
    for path in [cameras_path, images_path, points_path]:
        if not path.exists():
            errors.append(f"required COLMAP text file is missing: {path}")

    cameras = read_cameras_text(cameras_path) if cameras_path.exists() else {}
    image_records = read_images_text(images_path) if images_path.exists() else {}
    point_count, point_track_refs = read_points3d_text(points_path) if points_path.exists() else (0, 0)

    for camera in cameras.values():
        if camera.model not in SUPPORTED_CAMERA_MODELS:
            errors.append(
                f"camera {camera.camera_id} uses {camera.model}; this 3DGS loader requires PINHOLE text cameras"
            )
        if camera.width <= 0 or camera.height <= 0:
            errors.append(f"camera {camera.camera_id} has invalid size {camera.width}x{camera.height}")
        if camera.model == "PINHOLE" and len(camera.params) != 4:
            errors.append(f"camera {camera.camera_id} PINHOLE expects 4 params, got {len(camera.params)}")

    missing_images = []
    size_mismatches = []
    unknown_camera_ids = []
    for record in image_records.values():
        camera = cameras.get(record.camera_id)
        if camera is None:
            unknown_camera_ids.append((record.image_name, record.camera_id))
            continue
        image_path = images_dir / record.image_name
        if not image_path.exists():
            missing_images.append(record.image_name)
            continue
        try:
            with Image.open(image_path) as img:
                if img.size != (camera.width, camera.height):
                    size_mismatches.append(
                        {
                            "image_name": record.image_name,
                            "image_size": list(img.size),
                            "camera_size": [camera.width, camera.height],
                        }
                    )
        except Exception as exc:  # pragma: no cover - defensive user-data check.
            errors.append(f"failed to open image {image_path}: {exc}")

    if missing_images:
        errors.append(f"{len(missing_images)} registered images are missing under {images_dir}")
    if unknown_camera_ids:
        errors.append(f"{len(unknown_camera_ids)} images reference missing camera ids")
    if size_mismatches:
        errors.append(f"{len(size_mismatches)} image sizes differ from cameras.txt")
    if not cameras:
        errors.append("no cameras parsed")
    if not image_records:
        errors.append("no registered images parsed")
    if point_count <= 0:
        errors.append("no points parsed from points3D.txt")

    ply_path = sparse_dir / "points3D.ply"
    if args.write_ply and points_path.exists() and not ply_path.exists():
        write_points3d_ply(points_path, ply_path)
    elif not ply_path.exists():
        warnings.append("points3D.ply is missing; 3DGS can create it from points3D.txt on first load")

    num_image_files = count_image_files(images_dir) if images_dir.exists() else 0
    report = {
        "source_path": str(source_path),
        "images_dir": str(images_dir),
        "sparse_dir": str(sparse_dir),
        "num_image_files": num_image_files,
        "num_cameras": len(cameras),
        "num_registered_images": len(image_records),
        "num_points3D": point_count,
        "num_point_track_refs": point_track_refs,
        "missing_images": missing_images[:50],
        "size_mismatches": size_mismatches[:50],
        "unknown_camera_ids": unknown_camera_ids[:50],
        "warnings": warnings,
        "errors": errors,
        "valid": not errors,
    }

    if args.report_name:
        report_path = Path(args.report_name)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Source: {source_path}")
    print(f"Images dir: {images_dir}")
    print(f"Cameras: {len(cameras)}")
    print(f"Registered images: {len(image_records)}")
    print(f"Image files: {num_image_files}")
    print(f"Points3D: {point_count}")
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    if errors:
        print("Errors:")
        for error in errors:
            print(f"  - {error}")
        return 1
    print("3DGS scene check: OK")
    return 0


def read_cameras_text(path: Path) -> dict[int, CameraRecord]:
    cameras = {}
    for line in read_data_lines(path):
        elems = line.split()
        camera_id = int(elems[0])
        cameras[camera_id] = CameraRecord(
            camera_id=camera_id,
            model=elems[1],
            width=int(elems[2]),
            height=int(elems[3]),
            params=tuple(float(value) for value in elems[4:]),
        )
    return cameras


def read_images_text(path: Path) -> dict[int, ImageRecord]:
    records = {}
    lines = list(read_data_lines(path))
    if len(lines) % 2 != 0:
        raise ValueError(f"images.txt has an odd number of non-comment lines: {path}")
    for idx in range(0, len(lines), 2):
        header = lines[idx].split()
        points_line = lines[idx + 1].split()
        image_id = int(header[0])
        point3d_ids = [int(value) for value in points_line[2::3]]
        records[image_id] = ImageRecord(
            image_id=image_id,
            camera_id=int(header[8]),
            image_name=header[9],
            num_points2d=len(point3d_ids),
            num_linked_points3d=sum(1 for value in point3d_ids if value != -1),
        )
    return records


def read_points3d_text(path: Path) -> tuple[int, int]:
    count = 0
    track_refs = 0
    for line in read_data_lines(path):
        elems = line.split()
        count += 1
        track_refs += max(0, (len(elems) - 8) // 2)
    return count, track_refs


def read_data_lines(path: Path):
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            yield line


def count_image_files(images_dir: Path) -> int:
    return sum(1 for path in images_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def write_points3d_ply(points_path: Path, ply_path: Path) -> None:
    rows = []
    for line in read_data_lines(points_path):
        elems = line.split()
        rows.append(
            (
                float(elems[1]),
                float(elems[2]),
                float(elems[3]),
                int(elems[4]),
                int(elems[5]),
                int(elems[6]),
            )
        )
    with ply_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(rows)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property float nx\n")
        f.write("property float ny\n")
        f.write("property float nz\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for x, y, z, r, g, b in rows:
            f.write(f"{x} {y} {z} 0 0 0 {r} {g} {b}\n")


if __name__ == "__main__":
    raise SystemExit(main())
