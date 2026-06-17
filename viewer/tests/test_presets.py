from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from viewer.camera_io import load_cameras, read_ply_summary
from viewer.example_catalog import load_example_specs


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_ROOT = PROJECT_ROOT / "viewer/examples"


class VggtBaPresetTest(unittest.TestCase):
    def test_vggt_ba_7k_files_and_metadata(self) -> None:
        specs = {spec.example_id: spec for spec in load_example_specs(EXAMPLES_ROOT, PROJECT_ROOT)}
        spec = specs["vggt_ba_truck_7k"]
        camera_path = spec.camera_path
        ply_path = spec.ply_path

        self.assertIsNotNone(camera_path)
        assert camera_path is not None
        self.assertTrue(camera_path.is_file())
        self.assertTrue(ply_path.is_file())

        cameras = load_cameras(camera_path)
        summary = read_ply_summary(ply_path)

        self.assertEqual(len(cameras), 32)
        self.assertTrue(all(camera["rotation"] is not None for camera in cameras))
        self.assertTrue(all(camera["fx"] is not None and camera["fy"] is not None for camera in cameras))
        self.assertEqual(summary["vertex_count"], 404_547)
        self.assertEqual(summary["format"], "binary_little_endian")
        self.assertTrue(summary["is_gaussian_splat"])
        self.assertEqual(spec.expected_ply_bytes, summary["file_size"])

        rotations = np.asarray([camera["rotation"] for camera in cameras])
        forward_directions = rotations @ np.array([0.0, 0.0, 1.0])
        self.assertGreater(float(np.linalg.norm(np.std(forward_directions, axis=0))), 0.5)

    def test_camera_selection_immediately_updates_render_view(self) -> None:
        viewer_html = (PROJECT_ROOT / "viewer/static/viewer.html").read_text(encoding="utf-8")
        self.assertIn("if (navigateToView && camera.rotation) enterSelectedCamera();", viewer_html)
        self.assertIn("updateSelectedCamera(0, true);", viewer_html)

    def test_example_catalog_contains_both_methods(self) -> None:
        specs = {spec.example_id: spec for spec in load_example_specs(EXAMPLES_ROOT, PROJECT_ROOT)}
        self.assertEqual(set(specs), {"colmap_truck_30k", "vggt_ba_truck_7k"})
        self.assertEqual(specs["colmap_truck_30k"].method, "COLMAP")
        self.assertEqual(specs["vggt_ba_truck_7k"].method, "VGGT")


if __name__ == "__main__":
    unittest.main()
