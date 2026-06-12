#!/usr/bin/env python3
"""Export a StyleGAN2-ADA generator to ONNX for TensorRT conversion.

The exported model takes a z-vector (latent) as input and produces an
RGB image tensor (NCHW, float32, range [-1, 1]).

Truncation is baked into the export via --trunc (default 0.7).

Usage:
    uv run export_onnx.py --network models/metfaces.pkl --output models/metfaces.onnx
"""

import argparse
import os
import sys
import time
import warnings

import torch
from picklescan.scanner import scan_file_path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "stylegan2"))

# Force reference (pure-PyTorch) implementations of custom ops so that
# torch.jit.trace can capture the entire graph without custom CUDA kernels.
import torch_utils.custom_ops as _custom_ops

_custom_ops.get_plugin = lambda *a, **kw: (_ for _ in ()).throw(
    RuntimeError("disabled for ONNX export")
)

import dnnlib
from legacy import load_network_pkl

# Monkey-patch modulated_conv2d to always use the unfused path.
# The fused path uses grouped convolution with groups=batch_size, which
# creates data-dependent shapes that ONNX cannot represent. The unfused
# path is functionally identical but uses standard ops.
import training.networks as _networks

_orig_modulated_conv2d = _networks.modulated_conv2d


def _modulated_conv2d_unfused(*args, **kwargs):
    kwargs["fused_modconv"] = False
    return _orig_modulated_conv2d(*args, **kwargs)


_networks.modulated_conv2d = _modulated_conv2d_unfused


class StyleGAN2Wrapper(torch.nn.Module):
    """Wraps the StyleGAN2 generator for clean ONNX export.

    Folds mapping + truncation + synthesis into a single forward(z) call.
    Forces fp32 and const noise to avoid custom CUDA op dependencies.
    """

    def __init__(self, G, truncation_psi: float = 0.7):
        super().__init__()
        self.mapping = G.mapping
        self.synthesis = G.synthesis
        self.z_dim = G.z_dim
        self.c_dim = G.c_dim
        self.truncation_psi = truncation_psi
        self.register_buffer(
            "w_avg", G.mapping.w_avg.unsqueeze(0).unsqueeze(0)
        )
        self.num_ws = G.mapping.num_ws

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        label = torch.zeros(z.shape[0], self.c_dim, device=z.device)
        ws = self.mapping(z, label)
        ws = self.w_avg + self.truncation_psi * (ws - self.w_avg)
        img = self.synthesis(
            ws, noise_mode="const", force_fp32=True, fused_modconv=False
        )
        return img


def main():
    parser = argparse.ArgumentParser(description="Export StyleGAN2-ADA to ONNX")
    parser.add_argument(
        "--network",
        default=os.path.join(os.path.dirname(__file__), "models", "metfaces.pkl"),
        help="Path to .pkl model file",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output .onnx path (default: same name as network with .onnx extension)",
    )
    parser.add_argument(
        "--trunc",
        type=float,
        default=0.7,
        help="Truncation psi baked into the export",
    )
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.splitext(args.network)[0] + ".onnx"

    # Safety scan
    print(f"Scanning {args.network} for malicious content...")
    result = scan_file_path(args.network)
    if result.infected_files > 0:
        raise RuntimeError(f"picklescan detected issues in {args.network}")
    print("  Clean.")

    warnings.filterwarnings("ignore")

    print(f"Loading model from {args.network}...")
    with dnnlib.util.open_url(args.network) as f:
        data = load_network_pkl(f)
    G = data["G_ema"].cuda()
    G.eval()
    print(f"  Resolution: {G.img_resolution}x{G.img_resolution}")
    print(f"  z_dim: {G.z_dim}, c_dim: {G.c_dim}, num_ws: {G.num_ws}")

    wrapper = StyleGAN2Wrapper(G, truncation_psi=args.trunc).cuda()
    wrapper.eval()

    z = torch.randn(1, G.z_dim, device="cuda")

    # Verify the wrapper produces a valid image before export.
    with torch.no_grad():
        test_img = wrapper(z)
    print(f"  Test forward pass: {test_img.shape}, range [{test_img.min():.2f}, {test_img.max():.2f}]")

    print("Exporting to ONNX (JIT trace, batch=1, unfused modconv)...")
    t0 = time.time()
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (z,),
            args.output,
            input_names=["z"],
            output_names=["image"],
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )
    print(f"  Exported in {time.time() - t0:.1f}s")
    print(f"  Saved to {args.output}")
    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"  Size: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
