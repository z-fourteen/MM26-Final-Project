# Viewer Examples

Each example directory contains a tracked `scene.json` manifest and a small
tracked camera file. Large `point_cloud.ply` files are intentionally ignored.

Current PLY sizes:

| Example | Bytes | Approximate size |
|---|---:|---:|
| `colmap_truck_30k` | 513,109,068 | 489 MiB |
| `vggt_ba_truck_7k` | 100,329,187 | 95.7 MiB |

For direct collaboration, send the two PLY files separately and place them at
these exact paths after cloning the repository:

```text
viewer/examples/colmap_truck_30k/point_cloud.ply
viewer/examples/vggt_ba_truck_7k/point_cloud.ply
```

The received files may have any original filename, but they must be renamed to
`point_cloud.ply` at the destination. It is valid to provide only one example;
the viewer lists only examples whose local PLY and camera files are available.

Check exact sizes on Linux:

```bash
stat -c '%s %n' viewer/examples/*/point_cloud.ply
```

Or copy received files into place with validation:

```bash
python -m viewer.prepare_examples --mode copy \
  --source colmap_truck_30k=/path/to/received_colmap.ply \
  --source vggt_ba_truck_7k=/path/to/received_vggt.ply
```

Prepare symbolic links from the original local experiment paths when those
directories are still available:

```bash
python -m viewer.prepare_examples --mode symlink
```
