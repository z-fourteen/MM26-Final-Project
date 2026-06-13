from __future__ import annotations

import sys
from pathlib import Path


def ensure_vggt_importable(project_root: Path | None = None) -> Path:
    """Add the vendored VGGT repository to sys.path and return its root."""
    root = project_root or Path(__file__).resolve().parents[2]
    candidates = [
        root / "third_party" / "vggt",
        root / "vggt",
    ]
    for candidate in candidates:
        if (candidate / "vggt").is_dir():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            return candidate
    raise RuntimeError(
        "VGGT repository was not found. Expected third_party/vggt with the "
        "original vggt Python package inside it."
    )
