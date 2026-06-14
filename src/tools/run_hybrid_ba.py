#!/usr/bin/env python3
"""Hybrid BA: VGGT cameras + SfM tracks → COLMAP text → BA.

Writes COLMAP text format from VGGT cameras + triangulated SfM tracks,
then uses pycolmap's built-in read/write for BA.
"""

from __future__ import annotations

import argparse, json, struct, time
from pathlib import Path

import numpy as np
import pycolmap
from scipy.spatial.transform import Rotation

from src.adapters.vggt_adapter import ensure_vggt_importable
ensure_vggt_importable()

from src.sfm.config import load_scene_config, resolve_project_path
from src.sfm.triangulation import triangulate_point_dlt


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions-npz", required=True)
    p.add_argument("--scene", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--min-track-length", type=int, default=2)
    p.add_argument("--max-reproj-error", type=float, default=4.0)
    p.add_argument("--ba-max-iter", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def write_colmap_text(out_dir, image_names, extrinsic, intrinsic, points3d, tracks_obs):
    """Write cameras.txt, images.txt, points3D.txt in COLMAP format."""
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    N = len(image_names)
    h, w = 518, 518

    # cameras.txt: CAMERA_ID, SIMPLE_PINHOLE, W, H, f, cx, cy
    with (out / "cameras.txt").open("w") as f:
        for i, K in enumerate(intrinsic):
            f.write(f"{i+1} SIMPLE_PINHOLE {w} {h} {K[0,0]:.6f} {K[0,2]:.6f} {K[1,2]:.6f}\n")

    # images.txt with observations
    # Build image→point observations mapping
    img_obs = {i: [] for i in range(N)}  # img_idx → [(x,y, point3d_id), ...]
    for pt_id, track in enumerate(tracks_obs):
        for img_idx, xy in track:
            if img_idx < N:
                img_obs[img_idx].append((float(xy[0]), float(xy[1]), pt_id + 1))

    with (out / "images.txt").open("w") as f:
        for i in range(N):
            name = image_names[i]
            ext = extrinsic[i]
            R = ext[:3, :3]
            t = -R.T @ ext[:3, 3]  # camera center in world (COLMAP convention)
            quat = Rotation.from_matrix(R).as_quat()  # [x,y,z,w]
            # QW QX QY QZ TX TY TZ IMAGE_ID IMAGE_NAME
            f.write(f"{quat[3]:.9f} {quat[0]:.9f} {quat[1]:.9f} {quat[2]:.9f} ")
            f.write(f"{t[0]:.9f} {t[1]:.9f} {t[2]:.9f} {i+1} {name}\n")
            # Point2D observations
            obs = img_obs[i]
            if obs:
                f.write(" ".join(f"{x:.6f} {y:.6f} {pid}" for x, y, pid in obs) + "\n")
            else:
                f.write("\n")

    # Build reverse mapping: (img_idx, point3d_id) → point2D_idx in that image's obs list
    pt2d_idx_lookup = {}
    for i in range(N):
        for obs_idx, (x, y, pid) in enumerate(img_obs[i]):
            pt2d_idx_lookup[(i, pid)] = obs_idx

    # points3D.txt
    with (out / "points3D.txt").open("w") as f:
        for pt_id, (track, xyz) in enumerate(zip(tracks_obs, points3d)):
            pid = pt_id + 1
            track_parts = []
            for img_idx, _ in track:
                obs_idx = pt2d_idx_lookup.get((img_idx, pid))
                if obs_idx is not None:
                    track_parts.append(f"{img_idx+1} {obs_idx}")
            if track_parts:
                track_str = " ".join(track_parts)
                f.write(f"{xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f} 128 128 128 0.0 {track_str}\n")


def load_keypoint_coordinates(feature_dir):
    kps = {}
    for fp in sorted(Path(feature_dir).glob("*.npz")):
        d = np.load(fp)
        kps[str(d["image_name"])] = d["keypoints"][:, :2].astype(np.float64)
    return kps


def main():
    args = parse_args()
    np.random.seed(args.seed)

    # Load VGGT
    with np.load(args.predictions_npz) as pred:
        img_names = [str(n) for n in pred["image_paths"].tolist()]
        ext = pred["extrinsic"]
        K = pred["intrinsic"]
    name_to_idx = {n: i for i, n in enumerate(img_names)}

    # Load scene
    cfg = load_scene_config(args.scene)["scene"]
    feat_dir = resolve_project_path(cfg["feature_dir"])
    verif_dir = resolve_project_path(cfg["verified_dir"])
    out_dir = Path(args.output_dir)
    colmap_dir = out_dir / "colmap_hybrid"
    colmap_dir.mkdir(parents=True, exist_ok=True)

    kp_coords = load_keypoint_coordinates(feat_dir)

    # Build tracks
    obs_map = {}
    kp2g = {}
    nid = 0
    for vp in sorted(verif_dir.glob("*.npz")):
        v = np.load(vp, allow_pickle=True)
        if str(v.get("status","")) not in {"verified","verified_planar"}: continue
        if str(v.get("model_type","")) in {"panoramic","rejected_wtf"}: continue
        n1, n2 = str(v["image_name1"]), str(v["image_name2"])
        if n1 not in name_to_idx or n2 not in name_to_idx: continue
        i1, i2 = name_to_idx[n1], name_to_idx[n2]
        k1a, k2a = kp_coords.get(n1), kp_coords.get(n2)
        if k1a is None or k2a is None: continue
        for kp1, kp2 in v["inlier_matches"]:
            kp1, kp2 = int(kp1), int(kp2)
            if kp1<0 or kp2<0 or kp1>=len(k1a) or kp2>=len(k2a): continue
            g1, g2 = kp2g.get((i1,kp1)), kp2g.get((i2,kp2))
            if g1 is not None and g2 is not None:
                if g1!=g2:
                    obs_map[g1].extend(obs_map.pop(g2,[]))
                    for k,vk in list(kp2g.items()):
                        if vk==g2: kp2g[k]=g1
            elif g1 is not None:
                obs_map[g1].append((i2, k2a[kp2]))
                kp2g[(i2,kp2)]=g1
            elif g2 is not None:
                obs_map[g2].append((i1, k1a[kp1]))
                kp2g[(i1,kp1)]=g2
            else:
                obs_map[nid]=[(i1,k1a[kp1]),(i2,k2a[kp2])]
                kp2g[(i1,kp1)]=nid; kp2g[(i2,kp2)]=nid
                nid+=1

    # Filter + deduplicate tracks
    good = []
    for gid, obs in obs_map.items():
        seen = set(); dedup=[]
        for img_idx, xy in obs:
            if img_idx not in seen:
                seen.add(img_idx); dedup.append((img_idx, xy))
        if len(dedup)>=args.min_track_length:
            good.append(dedup)
    print(f"Tracks: {len(good)}")

    # Triangulate
    tri_pts, tri_tracks = [], []
    for track in good:
        idxs = [t[0] for t in track]; pts2d = [t[1] for t in track]
        # Precompute projection matrices
        Ps=[]
        for img_idx in idxs:
            R=ext[img_idx,:3,:3].astype(np.float64); t=ext[img_idx,:3,3].astype(np.float64)
            Ki=np.array([[K[img_idx,0,0],0,K[img_idx,0,2]],[0,K[img_idx,1,1],K[img_idx,1,2]],[0,0,1]],dtype=np.float64)
            Ps.append(Ki@np.hstack([R,t.reshape(3,1)]))

        best_pt, best_n, best_inl = None, 0, []
        for a in range(min(8,len(track))):
            for b in range(a+1, min(8,len(track))):
                pt = triangulate_point_dlt(Ps[a], Ps[b], pts2d[a], pts2d[b])
                if pt is None or not np.isfinite(pt).all(): continue
                n=0; inl=[]
                for j,(_,xy) in enumerate(track):
                    if j>=len(Ps): break
                    cam_pt = Ps[j]@np.append(pt,1.0)
                    if cam_pt[2]<=0: continue
                    proj=cam_pt[:2]/cam_pt[2]
                    if np.linalg.norm(proj-xy)<args.max_reproj_error:
                        n+=1; inl.append(j)
                if n>best_n: best_n=n; best_pt=pt; best_inl=inl

        if best_pt is not None and best_n>=2:
            tri_pts.append(best_pt)
            tri_tracks.append([(track[j][0], track[j][1]) for j in best_inl])
    print(f"Triangulated: {len(tri_pts)} points")

    if len(tri_pts)<10:
        print("Too few points")
        return 1

    # Write COLMAP text
    print("Writing COLMAP text...")
    write_colmap_text(colmap_dir, img_names, ext, K, np.array(tri_pts), tri_tracks)

    # Read into pycolmap
    print("Reading into pycolmap...")
    recon = pycolmap.Reconstruction()
    recon.read_text(str(colmap_dir))

    # BA
    print("Running BA...")
    t0=time.time()
    opts = pycolmap.BundleAdjustmentOptions()
    opts.ceres.loss_function_type = pycolmap.LossFunctionType.CAUCHY
    opts.ceres.solver_options.max_num_iterations = args.ba_max_iter
    opts.ceres.solver_options.num_threads = -1
    opts.print_summary = True
    pycolmap.bundle_adjustment(recon, opts)
    print(f"BA done in {time.time()-t0:.1f}s")

    # Save
    out_colmap = colmap_dir / "ba_output"
    out_colmap.mkdir(exist_ok=True)
    recon.write_text(str(out_colmap))

    # Extract refined points
    refined=[]
    with (out_colmap / "points3D.txt").open() as f:
        for line in f:
            if line.startswith("#"): continue
            parts=line.strip().split()
            if len(parts)>=3:
                refined.append([float(parts[0]),float(parts[1]),float(parts[2])])
    refined=np.array(refined,dtype=np.float32)
    print(f"BA refined points: {len(refined)}")

    # Write PLY
    pts=refined if len(refined) else np.array(tri_pts,dtype=np.float32)
    from src.tools.run_vggt_dtu_benchmark import write_binary_ply
    write_binary_ply(out_dir/"points_hybrid_ba.ply", pts)

    meta={"method":"VGGT+SfM tracks+COLMAP BA","vggt_cameras":len(img_names),
          "sfm_tracks":len(good),"triangulated":len(tri_pts),"ba_refined":len(refined)}
    (out_dir/"hybrid_ba_meta.json").write_text(json.dumps(meta,indent=2))
    print(json.dumps(meta,indent=2))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
