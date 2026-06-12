#!/usr/bin/env python3
"""StyleGAN2 inference server — reads z-vectors from stdin, writes images to stdout.

Protocol (binary, little-endian):
  Startup:  server writes a 12-byte header:
            [z_dim: u32] [img_size: u32] [img_channels: u32]

  Request:  client writes z_dim × f32  (z-vector, 4 × z_dim bytes)
  Response: server writes img_size × img_size × img_channels × u8  (RGB pixels, row-major)

  EOF on stdin → server exits cleanly.

Usage:
    python gan_server.py --network models/metfaces.pkl [--trunc 0.7] [--device cuda]
"""

import argparse
import os
import struct
import sys
import warnings

import numpy as np
import torch
from picklescan.scanner import scan_file_path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "stylegan2"))

import dnnlib
from legacy import load_network_pkl


def scan_model(pkl_path: str) -> None:
    result = scan_file_path(pkl_path)
    if result.scan_err:
        raise RuntimeError(f"picklescan failed to scan {pkl_path}")
    if result.infected_files > 0:
        raise RuntimeError(f"picklescan detected issues in {pkl_path}")


def main():
    parser = argparse.ArgumentParser(description="StyleGAN2 pipe inference server")
    parser.add_argument(
        "--network",
        default=os.path.join(os.path.dirname(__file__), "models", "metfaces.pkl"),
    )
    parser.add_argument("--trunc", type=float, default=0.7)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    device = torch.device(args.device)

    # Log to stderr so stdout stays clean for the binary protocol.
    def log(msg):
        print(msg, file=sys.stderr, flush=True)

    log(f"Scanning {args.network}...")
    scan_model(args.network)

    log(f"Loading model from {args.network}...")
    with dnnlib.util.open_url(args.network) as f:
        data = load_network_pkl(f)
    G = data["G_ema"].to(device)
    G.eval()

    z_dim = G.z_dim
    img_size = G.img_resolution
    img_channels = G.img_channels
    label = torch.zeros([1, G.c_dim], device=device)

    log(f"Ready: z_dim={z_dim}, img={img_size}x{img_size}x{img_channels}")

    # Write header to stdout.
    out = sys.stdout.buffer
    out.write(struct.pack("<III", z_dim, img_size, img_channels))
    out.flush()

    # Read z-vectors from stdin, generate images, write to stdout.
    inp = sys.stdin.buffer
    z_bytes = z_dim * 4  # f32

    while True:
        raw = inp.read(z_bytes)
        if len(raw) == 0:
            break  # EOF
        if len(raw) < z_bytes:
            log(f"Warning: incomplete z-vector ({len(raw)}/{z_bytes} bytes), exiting")
            break

        z = torch.frombuffer(bytearray(raw), dtype=torch.float32).reshape(1, z_dim).to(device)

        with torch.no_grad():
            img = G(z, label, truncation_psi=args.trunc, noise_mode="const")

        # NCHW [-1,1] float → HWC [0,255] uint8
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        pixels = img[0].cpu().numpy().tobytes()

        out.write(pixels)
        out.flush()


if __name__ == "__main__":
    main()
