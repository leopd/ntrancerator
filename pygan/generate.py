#!/usr/bin/env python3
"""Generate images using a pretrained StyleGAN2-ADA model."""

import argparse
import os
import sys
import time

import numpy as np
import torch
from PIL import Image
from picklescan.scanner import scan_file_path

# Add the vendored stylegan2-ada-pytorch to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "stylegan2"))

import dnnlib
from legacy import load_network_pkl


def scan_model(pkl_path: str) -> None:
    """Scan a pickle file for malicious content. Raises on infection."""
    result = scan_file_path(pkl_path)
    if result.scan_err:
        raise RuntimeError(f"picklescan failed to scan {pkl_path}")
    if result.infected_files > 0:
        dangerous = [g for g in result.globals if g.safety.value == "dangerous"]
        raise RuntimeError(
            f"picklescan detected {result.issues_count} issue(s) in {pkl_path}: "
            + ", ".join(f"{g.module}.{g.name}" for g in dangerous)
        )


def load_generator(pkl_path: str, device: torch.device) -> torch.nn.Module:
    scan_model(pkl_path)
    with dnnlib.util.open_url(pkl_path) as f:
        data = load_network_pkl(f)
    G = data["G_ema"].to(device)
    G.eval()
    return G


def generate_images(
    G: torch.nn.Module,
    seeds: list[int],
    truncation_psi: float,
    device: torch.device,
    noise_mode: str = "const",
) -> list[np.ndarray]:
    images = []
    label = torch.zeros([1, G.c_dim], device=device)

    for seed in seeds:
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
        with torch.no_grad():
            img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
        # Convert from NCHW [-1,1] float to HWC [0,255] uint8
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        images.append(img[0].cpu().numpy())

    return images


def main():
    parser = argparse.ArgumentParser(description="StyleGAN2-ADA image generation")
    parser.add_argument(
        "--network",
        default=os.path.join(os.path.dirname(__file__), "models", "metfaces.pkl"),
        help="Path to .pkl model file",
    )
    parser.add_argument(
        "--seeds",
        default="0,1,2,3",
        help="Comma-separated list of random seeds",
    )
    parser.add_argument(
        "--trunc",
        type=float,
        default=0.7,
        help="Truncation psi (0=mean face, 1=full variety)",
    )
    parser.add_argument(
        "--outdir",
        default=os.path.join(os.path.dirname(__file__), "out"),
        help="Output directory",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on",
    )
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    device = torch.device(args.device)
    os.makedirs(args.outdir, exist_ok=True)

    print(f"Loading model from {args.network}...")
    t0 = time.time()
    G = load_generator(args.network, device)
    print(f"  Loaded in {time.time() - t0:.1f}s")
    print(f"  Resolution: {G.img_resolution}x{G.img_resolution}")
    print(f"  Latent dim: {G.z_dim}, style layers: {G.num_ws}")

    print(f"Generating {len(seeds)} images (trunc={args.trunc})...")
    t0 = time.time()
    images = generate_images(G, seeds, args.trunc, device)
    elapsed = time.time() - t0
    print(f"  Generated in {elapsed:.2f}s ({elapsed / len(seeds):.2f}s per image)")

    for seed, img in zip(seeds, images):
        path = os.path.join(args.outdir, f"seed{seed:04d}.png")
        Image.fromarray(img, "RGB").save(path)
        print(f"  Saved {path}")


if __name__ == "__main__":
    main()
