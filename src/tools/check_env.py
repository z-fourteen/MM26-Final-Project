from __future__ import annotations

import importlib.util
from pathlib import Path


REQUIRED_MODULES = [
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("cv2", "opencv-contrib-python"),
    ("pycolmap", "pycolmap"),
    ("open3d", "open3d"),
    ("networkx", "networkx"),
    ("matplotlib", "matplotlib"),
    ("PIL", "Pillow"),
    ("yaml", "PyYAML"),
    ("tqdm", "tqdm"),
]

REQUIRED_DIRS = [
    "configs",
    "configs/scenes",
    "data/scenes/south_building_small/images",
    "data/scenes/south_building_small/features",
    "data/scenes/south_building_small/matches",
    "data/scenes/south_building_small/verified",
    "data/scenes/south_building_small/tracks",
    "data/scenes/south_building_small/sparse/0",
    "outputs/south_building_small/logs",
    "outputs/south_building_small/figures",
    "outputs/south_building_small/reports",
]


def module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    missing_modules = []
    missing_dirs = []

    print("Dependency check")
    for module_name, package_name in REQUIRED_MODULES:
        ok = module_available(module_name)
        status = "OK" if ok else "MISSING"
        print(f"  {package_name:<24} {status}")
        if not ok:
            missing_modules.append(package_name)

    print("\nDirectory check")
    for rel_path in REQUIRED_DIRS:
        ok = (root / rel_path).exists()
        status = "OK" if ok else "MISSING"
        print(f"  {rel_path:<50} {status}")
        if not ok:
            missing_dirs.append(rel_path)

    if missing_modules or missing_dirs:
        print("\nEnvironment is not ready.")
        if missing_modules:
            print("Missing packages: " + ", ".join(missing_modules))
        if missing_dirs:
            print("Missing directories: " + ", ".join(missing_dirs))
        return 1

    print("\nEnvironment is ready for Phase 1.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
