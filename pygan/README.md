# pygan — StyleGAN2-ADA Inference & Model Conversion

Python subproject for GAN model management: inference, safety scanning, and
serving models to the Rust `gan-slider` binary via a pipe protocol.

## Setup

```bash
cd pygan

# Clone the vendored StyleGAN2-ADA-PyTorch code
git clone --depth 1 https://github.com/NVlabs/stylegan2-ada-pytorch.git stylegan2

# Install dependencies (requires uv)
uv sync

# Download a pretrained model (MetFaces 1024x1024)
mkdir -p models
curl -L -o models/metfaces.pkl \
  https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl
```

### Jetson / GB10 Prerequisites

The `pyproject.toml` is configured to pull PyTorch from NVIDIA's Jetson index
(`pypi.jetson-ai-lab.io/sbsa/cu130`).  You also need:

```bash
sudo apt-get install -y libcudnn9-cuda-13 libcudss0-cuda-13
echo '/usr/lib/aarch64-linux-gnu/libcudss/13' | sudo tee /etc/ld.so.conf.d/cudss.conf
sudo ldconfig
```

## Tools

### generate.py — Image Generation

Generate images from a pretrained model.  Runs picklescan before loading.

```bash
uv run generate.py --seeds 42,137,256 --trunc 0.7
# Output: out/seed0042.png, out/seed0137.png, out/seed0256.png
```

### gan_server.py — Pipe Inference Server

Binary protocol server for the Rust `gan-slider` binary.  Reads z-vectors from
stdin, writes RGB pixels to stdout.  Not intended to be run directly by users.

Protocol:
- **Startup**: server writes `[z_dim: u32, img_size: u32, img_channels: u32]` (12 bytes, LE)
- **Request**: client writes `z_dim * 4` bytes (f32 z-vector)
- **Response**: server writes `img_size * img_size * img_channels` bytes (u8 RGB pixels)

### export_onnx.py — ONNX Export (Experimental)

Attempts ONNX export for TensorRT conversion.  Currently blocked by StyleGAN2's
data-dependent reshape operations.  See `specs/gan-slider-spec.md` §A1 for
details.

## Model Conversion Workflow

1. **Download** a `.pkl` model (e.g. from NVIDIA's pretrained collection)
2. **Scan** with picklescan: `uv run -c "from picklescan.scanner import scan_file_path; print(scan_file_path('models/yourmodel.pkl'))"`
3. **Test** with generate.py: `uv run generate.py --network models/yourmodel.pkl --seeds 0`
4. **Use** with gan-slider: `cargo run --release --bin gan-slider -- --model models/yourmodel.pkl`

Models must be in NVIDIA's `stylegan2-ada-pytorch` pickle format (containing
`G_ema` key).  TF-format pickles can potentially be converted using
`stylegan2/legacy.py`, but custom forks may have incompatible architectures.
