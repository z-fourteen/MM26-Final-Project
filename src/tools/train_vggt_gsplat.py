#!/usr/bin/env python3
"""Minimal 3D Gaussian Splatting training from VGGT outputs.

This script intentionally avoids the full gsplat example stack. It reads VGGT's
camera matrices and point cloud, optimizes Gaussian parameters against the input
views, and writes rendered training views plus a Gaussian PLY.
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from gsplat import export_splats
from gsplat.rendering import rasterization


SH_C0 = 0.28209479177387814


def parse_args():
    parser = argparse.ArgumentParser(description="Train a small 3DGS model initialized from VGGT.")
    parser.add_argument("--data_dir", default="outputs/kitchen_3dgs", help="Prepared 3DGS data dir with images/.")
    parser.add_argument("--predictions", default="outputs/kitchen_full/predictions.npz", help="VGGT predictions.npz.")
    parser.add_argument("--points", default="outputs/kitchen_full/points_depth.npz", help="VGGT points_depth.npz.")
    parser.add_argument("--output_dir", default="outputs/kitchen_gsplat_minimal", help="Training output dir.")
    parser.add_argument("--max_points", type=int, default=100_000, help="Number of initial Gaussians.")
    parser.add_argument("--steps", type=int, default=3000, help="Optimization steps.")
    parser.add_argument("--save_every", type=int, default=500, help="Save checkpoint every N steps.")
    parser.add_argument("--render_every", type=int, default=500, help="Render preview images every N steps.")
    parser.add_argument("--device", default="cuda", help="Training device.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--init_scale", type=float, default=0.006, help="Initial Gaussian scale in scene units.")
    parser.add_argument("--background", choices=["white", "black"], default="white")
    parser.add_argument("--lr_means", type=float, default=1.6e-4)
    parser.add_argument("--lr_scales", type=float, default=5e-3)
    parser.add_argument("--lr_quats", type=float, default=1e-3)
    parser.add_argument("--lr_opacities", type=float, default=5e-2)
    parser.add_argument("--lr_colors", type=float, default=2.5e-3)
    parser.add_argument("--scale_reg", type=float, default=1e-4)
    parser.add_argument("--opacity_reg", type=float, default=1e-4)
    return parser.parse_args()


def load_images(image_dir):
    paths = sorted(
        [p for p in Path(image_dir).iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    )
    if not paths:
        raise ValueError(f"No images found in {image_dir}")
    arrays = []
    for path in paths:
        img = Image.open(path).convert("RGB")
        arrays.append(np.asarray(img, dtype=np.float32) / 255.0)
    images = torch.from_numpy(np.stack(arrays, axis=0))
    return images, [p.name for p in paths]


def make_viewmats(extrinsics):
    viewmats = torch.eye(4, dtype=torch.float32).repeat(len(extrinsics), 1, 1)
    viewmats[:, :3, :4] = torch.from_numpy(extrinsics).float()
    return viewmats


def logit(x):
    x = x.clamp(1e-4, 1.0 - 1e-4)
    return torch.log(x / (1.0 - x))


def init_gaussians(points_npz, max_points, init_scale, device):
    xyz = points_npz["xyz"].astype(np.float32)
    rgb = points_npz["rgb"].astype(np.float32) / 255.0
    conf = points_npz["confidence"].astype(np.float32)

    if len(xyz) > max_points:
        keep = np.argsort(conf)[-max_points:]
        xyz = xyz[keep]
        rgb = rgb[keep]
        conf = conf[keep]

    means = torch.nn.Parameter(torch.from_numpy(xyz).to(device))
    log_scales = torch.nn.Parameter(
        torch.full((len(xyz), 3), float(np.log(init_scale)), dtype=torch.float32, device=device)
    )
    quats = torch.nn.Parameter(
        torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=device).repeat(len(xyz), 1)
    )
    opacities = torch.nn.Parameter(logit(torch.full((len(xyz),), 0.1, dtype=torch.float32, device=device)))
    color_logits = torch.nn.Parameter(logit(torch.from_numpy(rgb).to(device)))
    return means, log_scales, quats, opacities, color_logits, conf


def render(means, log_scales, quats, opacities, color_logits, viewmat, k, width, height, background):
    colors = torch.sigmoid(color_logits)
    scales = torch.exp(log_scales).clamp(1e-5, 0.2)
    quats_norm = F.normalize(quats, dim=-1)
    render_colors, render_alphas, _ = rasterization(
        means=means,
        quats=quats_norm,
        scales=scales,
        opacities=torch.sigmoid(opacities),
        colors=colors,
        viewmats=viewmat[None],
        Ks=k[None],
        width=width,
        height=height,
        packed=True,
        backgrounds=background,
    )
    return render_colors[0], render_alphas[0]


def save_png(path, tensor):
    array = tensor.detach().cpu().clamp(0, 1).numpy()
    Image.fromarray((array * 255).astype(np.uint8)).save(path)


def save_checkpoint(path, step, params, optimizer):
    means, log_scales, quats, opacities, color_logits = params
    torch.save(
        {
            "step": step,
            "means": means.detach().cpu(),
            "log_scales": log_scales.detach().cpu(),
            "quats": F.normalize(quats.detach().cpu(), dim=-1),
            "opacities": opacities.detach().cpu(),
            "color_logits": color_logits.detach().cpu(),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )


def export_gaussian_ply(path, params):
    means, log_scales, quats, opacities, color_logits = params
    rgb = torch.sigmoid(color_logits.detach())
    sh0 = ((rgb - 0.5) / SH_C0).unsqueeze(1)
    shN = torch.empty((rgb.shape[0], 0, 3), dtype=rgb.dtype, device=rgb.device)
    export_splats(
        means=means.detach(),
        scales=log_scales.detach(),
        quats=F.normalize(quats.detach(), dim=-1),
        opacities=opacities.detach(),
        sh0=sh0,
        shN=shN,
        format="ply",
        save_to=str(path),
    )


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    render_dir = output_dir / "renders"
    ckpt_dir = output_dir / "checkpoints"
    render_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    images, image_names = load_images(Path(args.data_dir) / "images")
    height, width = images.shape[1:3]
    predictions = np.load(args.predictions)
    points_npz = np.load(args.points)

    images = images.to(device)
    viewmats = make_viewmats(predictions["extrinsic"]).to(device)
    ks = torch.from_numpy(predictions["intrinsic"]).float().to(device)
    bg_value = 1.0 if args.background == "white" else 0.0
    background = torch.full((3,), bg_value, dtype=torch.float32, device=device)

    params = init_gaussians(points_npz, args.max_points, args.init_scale, device)[:5]
    means, log_scales, quats, opacities, color_logits = params

    optimizer = torch.optim.Adam(
        [
            {"params": [means], "lr": args.lr_means},
            {"params": [log_scales], "lr": args.lr_scales},
            {"params": [quats], "lr": args.lr_quats},
            {"params": [opacities], "lr": args.lr_opacities},
            {"params": [color_logits], "lr": args.lr_colors},
        ]
    )

    stats = {
        "data_dir": args.data_dir,
        "num_images": len(images),
        "height": height,
        "width": width,
        "num_gaussians": int(means.shape[0]),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "loss": [],
    }
    with open(output_dir / "train_config.json", "w", encoding="utf-8") as f:
        json.dump({**vars(args), **stats}, f, indent=2)

    start = time.time()
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, len(images), (1,), device=device).item()
        pred, alpha = render(
            means,
            log_scales,
            quats,
            opacities,
            color_logits,
            viewmats[idx],
            ks[idx],
            width,
            height,
            background,
        )
        target = images[idx]
        loss_rgb = F.l1_loss(pred, target)
        loss = (
            loss_rgb
            + args.scale_reg * torch.exp(log_scales).mean()
            + args.opacity_reg * torch.sigmoid(opacities).mean()
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step == 1 or step % 20 == 0:
            elapsed = time.time() - start
            msg = (
                f"step {step:05d}/{args.steps} "
                f"loss={loss.item():.5f} rgb={loss_rgb.item():.5f} "
                f"alpha_mean={alpha.mean().item():.3f} "
                f"elapsed={elapsed/60:.1f}m"
            )
            print(msg, flush=True)
            stats["loss"].append({"step": step, "loss": float(loss.item()), "rgb": float(loss_rgb.item())})

        if step == 1 or step % args.render_every == 0 or step == args.steps:
            with torch.no_grad():
                for ridx in sorted(set([0, len(images) // 2, len(images) - 1])):
                    preview, _ = render(
                        means,
                        log_scales,
                        quats,
                        opacities,
                        color_logits,
                        viewmats[ridx],
                        ks[ridx],
                        width,
                        height,
                        background,
                    )
                    save_png(render_dir / f"step_{step:05d}_{image_names[ridx]}", preview)

        if step % args.save_every == 0 or step == args.steps:
            save_checkpoint(ckpt_dir / f"step_{step:05d}.pt", step, params, optimizer)
            export_gaussian_ply(output_dir / f"gaussians_step_{step:05d}.ply", params)

    stats["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    stats["elapsed_sec"] = time.time() - start
    with open(output_dir / "train_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"Done. Outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
