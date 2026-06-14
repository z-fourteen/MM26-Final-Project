from __future__ import annotations

import argparse
import csv
import json
import struct
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from src.sfm.config import load_scene_config, project_root, resolve_project_path


SUPPORTED_POINT_ARRAY_KEYS = ("points", "points3D", "xyz", "vertices")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate reconstructed point clouds against GT with Accuracy, Completeness, and Overall."
    )
    parser.add_argument("--scene", default="", help="Path to configs/scenes/<scene>.yaml")
    parser.add_argument("--pred", default="", help="Predicted point cloud path, e.g. points3D.ply or points3D.txt")
    parser.add_argument("--gt", default="", help="GT point cloud or mesh path. DTU scenes default to raw DTU points.ply")
    parser.add_argument(
        "--source-path",
        default="",
        help="3DGS-style source root. Uses <source-path>/sparse/0/points3D.{ply,txt} when --pred is omitted.",
    )
    parser.add_argument(
        "--pred-name",
        default="",
        help="Name under data/3dgs_inputs, e.g. dtu_scan37_sfm_clean. Used when --pred is omitted.",
    )
    parser.add_argument("--output-json", default="", help="JSON report path. Defaults to outputs/<scene>/reports/...")
    parser.add_argument("--output-csv", default="", help="CSV report path. Defaults next to the JSON report.")
    parser.add_argument(
        "--alignment",
        choices=("icp", "center-scale", "none"),
        default="icp",
        help="Similarity alignment before distance evaluation. Use none only when clouds already share coordinates.",
    )
    parser.add_argument("--icp-threshold", type=float, default=0.0, help="Absolute ICP correspondence threshold.")
    parser.add_argument(
        "--icp-threshold-ratio",
        type=float,
        default=0.05,
        help="ICP threshold as a ratio of the GT bounding-box diagonal when --icp-threshold is not set.",
    )
    parser.add_argument("--icp-max-iterations", type=int, default=100)
    parser.add_argument("--icp-samples", type=int, default=50000, help="Max points per cloud for ICP fitting.")
    parser.add_argument("--num-samples", type=int, default=0, help="Set both --pred-samples and --gt-samples.")
    parser.add_argument("--pred-samples", type=int, default=0, help="Max predicted points for final metrics. 0 means all.")
    parser.add_argument("--gt-samples", type=int, default=200000, help="Max GT points for final metrics.")
    parser.add_argument("--voxel-size", type=float, default=0.0, help="Optional voxel downsample size before metrics.")
    parser.add_argument(
        "--max-distance",
        type=float,
        default=0.0,
        help="Clip distances above this value before averaging. 0 disables clipping.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    scene_config = load_scene_config(args.scene)["scene"] if args.scene else {}
    scene_name = str(scene_config.get("scene_name") or infer_scene_name(args))
    pred_path = resolve_prediction_path(args, scene_config)
    gt_path = resolve_gt_path(args, scene_config)
    report_stem = infer_report_stem(args, pred_path, scene_name)
    report_dir = infer_report_dir(args, pred_path, scene_name)
    output_json = resolve_output_json(args.output_json, report_dir, report_stem)
    output_csv = resolve_output_csv(args.output_csv, output_json)

    pred_samples = args.num_samples if args.num_samples > 0 else args.pred_samples
    gt_samples = args.num_samples if args.num_samples > 0 else args.gt_samples

    pred_points, pred_meta = load_geometry_points(
        pred_path,
        sample_count=pred_samples,
        voxel_size=args.voxel_size,
        seed=args.seed,
        role="prediction",
    )
    gt_points, gt_meta = load_geometry_points(
        gt_path,
        sample_count=gt_samples,
        voxel_size=args.voxel_size,
        seed=args.seed + 1,
        role="gt",
    )
    if len(pred_points) == 0:
        raise ValueError(f"prediction has no valid points: {pred_path}")
    if len(gt_points) == 0:
        raise ValueError(f"GT has no valid points: {gt_path}")

    transform_report = estimate_alignment(
        pred_points,
        gt_points,
        method=args.alignment,
        icp_samples=args.icp_samples,
        icp_threshold=args.icp_threshold,
        icp_threshold_ratio=args.icp_threshold_ratio,
        icp_max_iterations=args.icp_max_iterations,
        seed=args.seed + 2,
    )
    aligned_pred_points = apply_transform(pred_points, np.asarray(transform_report["matrix"], dtype=np.float64))

    accuracy_distances = nearest_distances(aligned_pred_points, gt_points)
    completeness_distances = nearest_distances(gt_points, aligned_pred_points)
    accuracy = distance_stats(accuracy_distances, args.max_distance)
    completeness = distance_stats(completeness_distances, args.max_distance)
    overall = 0.5 * (accuracy["mean"] + completeness["mean"])

    report: dict[str, Any] = {
        "scene": scene_name,
        "prediction_path": str(pred_path),
        "gt_path": str(gt_path),
        "alignment": transform_report,
        "parameters": {
            "num_samples": int(args.num_samples),
            "pred_samples": int(pred_samples),
            "gt_samples": int(gt_samples),
            "voxel_size": float(args.voxel_size),
            "max_distance": float(args.max_distance),
            "seed": int(args.seed),
        },
        "prediction": pred_meta | {"evaluated_points": int(len(pred_points))},
        "gt": gt_meta | {"evaluated_points": int(len(gt_points))},
        "metrics": {
            "accuracy": accuracy,
            "completeness": completeness,
            "overall": float(overall),
        },
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_csv(output_csv, report)
    print_report(report, output_json, output_csv)
    return 0


def infer_scene_name(args: argparse.Namespace) -> str:
    for value in (args.pred_name, args.source_path, args.pred):
        if value:
            path = Path(value)
            if path.name:
                return path.stem
    return "pointcloud_eval"


def infer_report_stem(args: argparse.Namespace, pred_path: Path, scene_name: str) -> str:
    if args.pred_name:
        return f"{Path(args.pred_name).name}_pointcloud_eval"
    source_path = resolve_project_path(args.source_path) if args.source_path else infer_3dgs_source_root(pred_path)
    if source_path is not None and source_path.name:
        return f"{source_path.name}_pointcloud_eval"
    return f"{scene_name}_pointcloud_eval"


def infer_report_dir(args: argparse.Namespace, pred_path: Path, scene_name: str) -> Path:
    source_path = resolve_project_path(args.source_path) if args.source_path else infer_3dgs_source_root(pred_path)
    if source_path is not None:
        return source_path / "reports"
    return project_root() / "outputs" / scene_name / "reports"


def infer_3dgs_source_root(pred_path: Path) -> Path | None:
    parts = pred_path.parts
    if len(parts) >= 4 and parts[-3:] in [
        ("sparse", "0", "points3D.ply"),
        ("sparse", "0", "points3D.txt"),
    ]:
        return pred_path.parents[2]
    if len(parts) >= 4 and pred_path.name in {"points3D.ply", "points3D.txt"}:
        sparse_zero = pred_path.parent
        if sparse_zero.name == "0" and sparse_zero.parent.name == "sparse":
            return sparse_zero.parent.parent
    return None


def resolve_prediction_path(args: argparse.Namespace, scene_config: dict[str, Any]) -> Path:
    if args.pred:
        return require_file(resolve_project_path(args.pred), "--pred")
    if args.source_path:
        return require_first_existing(
            [resolve_project_path(args.source_path) / "sparse" / "0" / name for name in ("points3D.ply", "points3D.txt")],
            "--source-path",
        )
    if args.pred_name:
        root = project_root() / "data" / "3dgs_inputs" / args.pred_name / "sparse" / "0"
        return require_first_existing([root / "points3D.ply", root / "points3D.txt"], "--pred-name")

    scene_name = str(scene_config.get("scene_name", ""))
    if not scene_name:
        raise ValueError("Provide --pred, --source-path, --pred-name, or --scene with scene_name.")
    candidates = [
        project_root() / "data" / "3dgs_inputs" / f"{scene_name}_sfm" / "sparse" / "0" / "points3D.ply",
        project_root() / "data" / "3dgs_inputs" / f"{scene_name}_sfm" / "sparse" / "0" / "points3D.txt",
        resolve_project_path(scene_config.get("sparse_dir", "")) / "points3D.ply",
        resolve_project_path(scene_config.get("sparse_dir", "")) / "points3D.txt",
    ]
    existing = [path for path in candidates if path.exists()]
    if existing:
        return existing[0]

    globbed = sorted((project_root() / "data" / "3dgs_inputs").glob(f"{scene_name}_*/sparse/0/points3D.ply"))
    if len(globbed) == 1:
        return globbed[0]
    if len(globbed) > 1:
        options = "\n".join(f"  - {path}" for path in globbed[:20])
        raise ValueError(f"Multiple prediction clouds match {scene_name}. Pass --pred or --pred-name explicitly:\n{options}")
    raise FileNotFoundError(f"No prediction point cloud found for scene {scene_name}. Pass --pred explicitly.")


def resolve_gt_path(args: argparse.Namespace, scene_config: dict[str, Any]) -> Path:
    if args.gt:
        return require_file(resolve_project_path(args.gt), "--gt")

    dataset_name = str(scene_config.get("dataset_name", ""))
    if dataset_name.lower().startswith("dtu/"):
        scan_name = dataset_name.split("/", 1)[1]
        return require_file(project_root() / "data" / "raw" / "DTU" / scan_name / "points.ply", "DTU GT")

    scene_name = str(scene_config.get("scene_name", ""))
    if scene_name.startswith("dtu_scan"):
        scan_name = "scan" + scene_name.removeprefix("dtu_scan")
        return require_file(project_root() / "data" / "raw" / "DTU" / scan_name / "points.ply", "DTU GT")
    raise ValueError("Could not infer GT path. Pass --gt explicitly.")


def resolve_output_json(output_json_arg: str, report_dir: Path, report_stem: str) -> Path:
    if output_json_arg:
        return resolve_project_path(output_json_arg)
    return report_dir / f"{report_stem}.json"


def resolve_output_csv(output_csv_arg: str, output_json: Path) -> Path:
    if output_csv_arg:
        return resolve_project_path(output_csv_arg)
    return output_json.with_suffix(".csv")


def require_file(path: Path, label: str) -> Path:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"{label} file does not exist: {path}")
    return path


def require_first_existing(paths: list[Path], label: str) -> Path:
    for path in paths:
        if path.exists() and path.is_file():
            return path
    options = ", ".join(str(path) for path in paths)
    raise FileNotFoundError(f"{label} did not resolve to an existing point cloud. Tried: {options}")


def load_geometry_points(
    path: Path,
    *,
    sample_count: int,
    voxel_size: float,
    seed: int,
    role: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".txt":
        points = read_colmap_points3d_text(path)
        kind = "colmap_points3D_txt"
    elif suffix == ".npy":
        points = np.asarray(np.load(path), dtype=np.float64)
        kind = "npy"
    elif suffix == ".npz":
        points, key = read_npz_points(path)
        kind = f"npz:{key}"
    elif suffix == ".ply":
        points = read_ply_vertices(path)
        kind = "ply_vertices"
    else:
        points, kind = read_open3d_geometry(path, sample_count=sample_count, role=role)

    points = clean_points(points)
    input_count = int(len(points))
    if voxel_size > 0.0 and len(points):
        points = voxel_downsample(points, voxel_size)
    points = random_sample(points, sample_count, seed)
    return points, {
        "path": str(path),
        "kind": kind,
        "input_points": input_count,
        "bbox_min": points.min(axis=0).tolist() if len(points) else [],
        "bbox_max": points.max(axis=0).tolist() if len(points) else [],
    }


def read_open3d_geometry(path: Path, *, sample_count: int, role: str) -> tuple[np.ndarray, str]:
    try:
        import open3d as o3d
    except ImportError as exc:  # pragma: no cover - depends on local environment.
        raise ImportError("open3d is required to read PLY/OBJ/STL/OFF geometry. Install requirements.txt.") from exc

    mesh = o3d.io.read_triangle_mesh(str(path), enable_post_processing=False)
    if mesh.has_triangles():
        target_samples = sample_count if sample_count > 0 else 200000
        sampled = mesh.sample_points_uniformly(number_of_points=int(target_samples))
        return np.asarray(sampled.points, dtype=np.float64), "triangle_mesh_sampled"

    cloud = o3d.io.read_point_cloud(str(path))
    if cloud.has_points():
        return np.asarray(cloud.points, dtype=np.float64), "point_cloud"

    if len(mesh.vertices):
        return np.asarray(mesh.vertices, dtype=np.float64), "mesh_vertices"
    raise ValueError(f"Could not read points from {role} geometry: {path}")


def read_ply_vertices(path: Path) -> np.ndarray:
    with path.open("rb") as file:
        header_lines: list[str] = []
        while True:
            line = file.readline()
            if not line:
                raise ValueError(f"PLY header is missing end_header: {path}")
            text = line.decode("ascii", errors="replace").strip()
            header_lines.append(text)
            if text == "end_header":
                break
        vertex_count, properties, fmt = parse_ply_header(header_lines, path)
        if fmt == "ascii":
            return read_ascii_ply_vertices(file, vertex_count, properties)
        if fmt in {"binary_little_endian", "binary_big_endian"}:
            return read_binary_ply_vertices(file, vertex_count, properties, fmt)
    raise ValueError(f"Unsupported PLY format in {path}: {fmt}")


def parse_ply_header(header_lines: list[str], path: Path) -> tuple[int, list[tuple[str, str]], str]:
    if not header_lines or header_lines[0] != "ply":
        raise ValueError(f"Not a PLY file: {path}")
    fmt = ""
    vertex_count = 0
    properties: list[tuple[str, str]] = []
    in_vertex = False
    for line in header_lines[1:]:
        elems = line.split()
        if not elems:
            continue
        if elems[0] == "format" and len(elems) >= 2:
            fmt = elems[1]
        elif elems[0] == "element":
            in_vertex = len(elems) >= 3 and elems[1] == "vertex"
            if in_vertex:
                vertex_count = int(elems[2])
        elif elems[0] == "property" and in_vertex:
            if len(elems) >= 3 and elems[1] != "list":
                properties.append((elems[1], elems[2]))
            else:
                raise ValueError(f"List properties are not supported for PLY vertices: {path}")
    if fmt not in {"ascii", "binary_little_endian", "binary_big_endian"}:
        raise ValueError(f"Unsupported PLY format in {path}: {fmt}")
    prop_names = [name for _, name in properties]
    for coord in ("x", "y", "z"):
        if coord not in prop_names:
            raise ValueError(f"PLY vertex property {coord!r} is missing: {path}")
    return vertex_count, properties, fmt


def read_ascii_ply_vertices(file, vertex_count: int, properties: list[tuple[str, str]]) -> np.ndarray:
    prop_names = [name for _, name in properties]
    xyz_indices = [prop_names.index(coord) for coord in ("x", "y", "z")]
    rows = np.empty((vertex_count, 3), dtype=np.float64)
    for idx in range(vertex_count):
        line = file.readline()
        if not line:
            raise ValueError(f"PLY ended before reading {vertex_count} vertices")
        elems = line.decode("ascii", errors="replace").split()
        rows[idx] = [float(elems[col]) for col in xyz_indices]
    return rows


def read_binary_ply_vertices(
    file,
    vertex_count: int,
    properties: list[tuple[str, str]],
    fmt: str,
) -> np.ndarray:
    endian = "<" if fmt == "binary_little_endian" else ">"
    struct_codes = [ply_struct_code(prop_type) for prop_type, _ in properties]
    vertex_struct = struct.Struct(endian + "".join(struct_codes))
    prop_names = [name for _, name in properties]
    xyz_indices = [prop_names.index(coord) for coord in ("x", "y", "z")]
    rows = np.empty((vertex_count, 3), dtype=np.float64)
    for idx in range(vertex_count):
        payload = file.read(vertex_struct.size)
        if len(payload) != vertex_struct.size:
            raise ValueError(f"PLY ended before reading {vertex_count} binary vertices")
        values = vertex_struct.unpack(payload)
        rows[idx] = [float(values[col]) for col in xyz_indices]
    return rows


def ply_struct_code(prop_type: str) -> str:
    codes = {
        "char": "b",
        "int8": "b",
        "uchar": "B",
        "uint8": "B",
        "short": "h",
        "int16": "h",
        "ushort": "H",
        "uint16": "H",
        "int": "i",
        "int32": "i",
        "uint": "I",
        "uint32": "I",
        "float": "f",
        "float32": "f",
        "double": "d",
        "float64": "d",
    }
    try:
        return codes[prop_type]
    except KeyError as exc:
        raise ValueError(f"Unsupported PLY property type: {prop_type}") from exc


def read_colmap_points3d_text(path: Path) -> np.ndarray:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        elems = line.split()
        if len(elems) < 4:
            continue
        rows.append((float(elems[1]), float(elems[2]), float(elems[3])))
    return np.asarray(rows, dtype=np.float64)


def read_npz_points(path: Path) -> tuple[np.ndarray, str]:
    data = np.load(path)
    for key in SUPPORTED_POINT_ARRAY_KEYS:
        if key in data:
            arr = np.asarray(data[key], dtype=np.float64)
            if arr.ndim == 2 and arr.shape[1] >= 3:
                return arr[:, :3], key
    for key in data.files:
        arr = np.asarray(data[key])
        if arr.ndim == 2 and arr.shape[1] >= 3:
            return arr[:, :3].astype(np.float64), key
    raise ValueError(f"No Nx3 point array found in {path}")


def clean_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"Expected Nx3 points, got shape {points.shape}")
    points = points[:, :3]
    return points[np.isfinite(points).all(axis=1)]


def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    voxel_indices = np.floor(points / float(voxel_size)).astype(np.int64)
    _, inverse = np.unique(voxel_indices, axis=0, return_inverse=True)
    counts = np.bincount(inverse).astype(np.float64)
    sums = np.zeros((len(counts), 3), dtype=np.float64)
    np.add.at(sums, inverse, points)
    return sums / counts.reshape(-1, 1)


def random_sample(points: np.ndarray, sample_count: int, seed: int) -> np.ndarray:
    if sample_count <= 0 or len(points) <= sample_count:
        return points
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(points), size=int(sample_count), replace=False)
    return points[np.sort(indices)]


def estimate_alignment(
    pred_points: np.ndarray,
    gt_points: np.ndarray,
    *,
    method: str,
    icp_samples: int,
    icp_threshold: float,
    icp_threshold_ratio: float,
    icp_max_iterations: int,
    seed: int,
) -> dict[str, Any]:
    center_scale = initial_center_scale_transform(pred_points, gt_points)
    if method == "none":
        matrix = np.eye(4, dtype=np.float64)
        return {"method": method, "matrix": matrix.tolist()}
    if method == "center-scale":
        return {"method": method, "matrix": center_scale.tolist()}
    if method != "icp":
        raise ValueError(f"Unsupported alignment method: {method}")

    source = random_sample(pred_points, icp_samples, seed)
    target = random_sample(gt_points, icp_samples, seed + 1)
    threshold = float(icp_threshold)
    if threshold <= 0.0:
        threshold = float(icp_threshold_ratio) * bbox_diagonal(target)
    matrix, fitness, rmse, iterations = run_similarity_icp(
        source,
        target,
        init_matrix=center_scale,
        threshold=threshold,
        max_iterations=icp_max_iterations,
    )
    return {
        "method": method,
        "matrix": matrix.tolist(),
        "initial_matrix": center_scale.tolist(),
        "fitness": float(fitness),
        "inlier_rmse": float(rmse),
        "threshold": float(threshold),
        "source_points": int(len(source)),
        "target_points": int(len(target)),
        "max_iterations": int(icp_max_iterations),
        "iterations": int(iterations),
    }


def run_similarity_icp(
    source: np.ndarray,
    target: np.ndarray,
    *,
    init_matrix: np.ndarray,
    threshold: float,
    max_iterations: int,
) -> tuple[np.ndarray, float, float, int]:
    matrix = np.asarray(init_matrix, dtype=np.float64).copy()
    tree = cKDTree(target)
    previous_rmse = np.inf
    fitness = 0.0
    rmse = np.inf
    iterations = 0
    for iteration in range(max(1, int(max_iterations))):
        transformed = apply_transform(source, matrix)
        distances, indices = tree.query(transformed, k=1)
        mask = np.isfinite(distances)
        if threshold > 0.0:
            mask &= distances <= threshold
        if int(mask.sum()) < 3:
            break
        matched_target = target[indices[mask]]
        delta = estimate_similarity_transform(transformed[mask], matched_target)
        matrix = delta @ matrix
        inlier_distances = distances[mask]
        rmse = float(np.sqrt(np.mean(inlier_distances**2)))
        fitness = float(mask.mean())
        iterations = iteration + 1
        if abs(previous_rmse - rmse) <= max(1e-12, previous_rmse * 1e-6):
            break
        previous_rmse = rmse
    if not np.isfinite(rmse):
        transformed = apply_transform(source, matrix)
        distances, _ = tree.query(transformed, k=1)
        mask = distances <= threshold if threshold > 0.0 else np.ones(len(distances), dtype=bool)
        fitness = float(mask.mean()) if len(mask) else 0.0
        rmse = float(np.sqrt(np.mean(distances[mask] ** 2))) if np.any(mask) else float(np.sqrt(np.mean(distances**2)))
    return matrix, fitness, rmse, iterations


def estimate_similarity_transform(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    source_zero = source - source_center.reshape(1, 3)
    target_zero = target - target_center.reshape(1, 3)
    covariance = (target_zero.T @ source_zero) / float(len(source))
    u, singular_values, vt = np.linalg.svd(covariance)
    correction = np.eye(3, dtype=np.float64)
    if np.linalg.det(u @ vt) < 0.0:
        correction[-1, -1] = -1.0
    rotation = u @ correction @ vt
    variance = float(np.mean(np.sum(source_zero**2, axis=1)))
    scale = 1.0 if variance <= 0.0 else float(np.sum(singular_values * np.diag(correction)) / variance)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = scale * rotation
    matrix[:3, 3] = target_center - scale * rotation @ source_center
    return matrix


def initial_center_scale_transform(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    source_radius = rms_radius(source, source_center)
    target_radius = rms_radius(target, target_center)
    scale = 1.0 if source_radius <= 0.0 else target_radius / source_radius
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] *= scale
    matrix[:3, 3] = target_center - scale * source_center
    return matrix


def rms_radius(points: np.ndarray, center: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((points - center.reshape(1, 3)) ** 2, axis=1))))


def bbox_diagonal(points: np.ndarray) -> float:
    if len(points) == 0:
        return 1.0
    diagonal = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    return max(diagonal, 1e-12)


def apply_transform(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    hom = np.ones((len(points), 4), dtype=np.float64)
    hom[:, :3] = points
    return (matrix @ hom.T).T[:, :3]


def nearest_distances(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    tree = cKDTree(target)
    try:
        distances, _ = tree.query(source, k=1, workers=-1)
    except TypeError:  # pragma: no cover - older scipy.
        distances, _ = tree.query(source, k=1)
    return np.asarray(distances, dtype=np.float64)


def distance_stats(distances: np.ndarray, max_distance: float) -> dict[str, Any]:
    raw = np.asarray(distances, dtype=np.float64)
    values = raw
    clipped = False
    if max_distance > 0.0:
        values = np.minimum(raw, float(max_distance))
        clipped = True
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "rmse": float(np.sqrt(np.mean(values**2))),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
        "raw_mean": float(np.mean(raw)),
        "count": int(len(values)),
        "clipped": clipped,
    }


def write_csv(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics = report["metrics"]
    alignment = report["alignment"]
    row = {
        "scene": report["scene"],
        "prediction_path": report["prediction_path"],
        "gt_path": report["gt_path"],
        "alignment": alignment["method"],
        "alignment_fitness": alignment.get("fitness", ""),
        "alignment_inlier_rmse": alignment.get("inlier_rmse", ""),
        "accuracy": metrics["accuracy"]["mean"],
        "completeness": metrics["completeness"]["mean"],
        "overall": metrics["overall"],
        "accuracy_median": metrics["accuracy"]["median"],
        "completeness_median": metrics["completeness"]["median"],
        "pred_points": report["prediction"]["evaluated_points"],
        "gt_points": report["gt"]["evaluated_points"],
    }
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def print_report(report: dict[str, Any], output_json: Path, output_csv: Path) -> None:
    metrics = report["metrics"]
    print(f"Scene: {report['scene']}")
    print(f"Prediction: {report['prediction_path']}")
    print(f"GT: {report['gt_path']}")
    print(f"Alignment: {report['alignment']['method']}")
    if report["alignment"]["method"] == "icp":
        print(
            "ICP: "
            f"fitness={report['alignment']['fitness']:.6g}, "
            f"inlier_rmse={report['alignment']['inlier_rmse']:.6g}, "
            f"threshold={report['alignment']['threshold']:.6g}"
        )
    print("")
    print("Metric          Mean")
    print(f"Accuracy     {metrics['accuracy']['mean']:.9g}")
    print(f"Completeness {metrics['completeness']['mean']:.9g}")
    print(f"Overall      {metrics['overall']:.9g}")
    print("")
    print(f"JSON: {output_json}")
    print(f"CSV: {output_csv}")


if __name__ == "__main__":
    raise SystemExit(main())
