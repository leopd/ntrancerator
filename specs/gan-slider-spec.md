# GAN Slider — Spec & Assumptions

## Overview

`gan-slider` is a real-time interactive tool that connects an Akai APC mini mk2
MIDI controller to a StyleGAN2-ADA generator.  Eight physical sliders control a
random linear projection into the GAN's 512-dimensional latent space; the
generated images are displayed in a wgpu/Vulkan window with live FPS reporting.

## Architecture

```
APC mini mk2 (MIDI)
   │ CC 48..55 (sliders)
   │ Note 64..71 (buttons)
   ▼
┌──────────────────────┐
│  gan-slider (Rust)   │
│                      │
│  MIDI → MidiState    │
│  SliderProjection    │   stdin (z-vector, 2048 bytes)
│  GanClient ─────────────►┌──────────────────────┐
│                      │   │ gan_server.py (Python)│
│  wgpu display  ◄─────────┤ StyleGAN2-ADA + CUDA │
│  (Vulkan)        RGB │   └──────────────────────┘
│                 pixels    stdout (3 MB per frame)
└──────────────────────┘
```

## Assumptions Worth Revisiting

### A1: Pipe-based IPC instead of TensorRT

We use a Python subprocess (`gan_server.py`) communicating over stdin/stdout
pipes instead of TensorRT.  The ONNX export of StyleGAN2-ADA failed due to:

- **Modulated convolution** uses `groups=batch_size` in a grouped conv2d, which
  creates data-dependent tensor shapes that ONNX cannot represent.
- **upfirdn2d reference impl** uses `x.shape[N]` in slice indices, also
  data-dependent.
- The unfused modconv path avoids the grouped conv but still hits upfirdn2d
  issues.

A TensorRT engine would give ~2-3x speedup and avoid the ~3 MB/frame pipe
overhead.  If revisited:
- Use NVIDIA's official `stylegan3` repo which has better ONNX export support.
- Or use `torch-tensorrt` (Torch-TensorRT) for direct TorchScript→TRT
  compilation, bypassing ONNX entirely.
- Or patch upfirdn2d to use fixed-size ops (replace `x.shape[N]` slicing with
  `F.pad` + `narrow`).

### A2: MetFaces model instead of WikiArt

The user requested WikiArt.  The only available WikiArt model was trained with a
custom TF StyleGAN2 fork that has non-standard architecture parameters
(`res_log2` kwarg) incompatible with the standard NVIDIA legacy converter.  We
use MetFaces (1024x1024 portrait paintings from the Met) instead, which loads
cleanly from NVIDIA's official PyTorch pretrained collection.

To get WikiArt working, one would need to:
- Find or train a WikiArt model using the standard `stylegan2-ada-pytorch` code.
- Or port the custom TF model by patching the legacy converter to handle the
  extra kwargs.

### A3: Button-to-note mapping (APC mini mk2)

We assume the 8 round buttons directly above sliders 1-8 send Note-On messages
for MIDI notes 64..71.  This is the default factory mapping for the APC mini mk2
in "mode 0".  If the controller is in a different mode, the note numbers may
differ.  The `--list` flag can help debug MIDI routing.

### A4: Random projection scaling

The slider-to-z projection uses a simple random Gaussian matrix with no
normalization.  Each slider axis contributes `t * direction[i]` where `t ∈
[-1, 1]` and `direction[i] ~ N(0, 1)`.  With 8 sliders at extremes, the
displacement magnitude is roughly `sqrt(8) * sqrt(512) ≈ 64`, which may push
z-vectors far from the training distribution.

If generated images look bad at slider extremes, consider:
- Scaling the projection matrix by `1 / sqrt(z_dim)` or similar.
- Using truncation-aware projection (project in W-space instead of Z-space).
- Adding a norm constraint to keep `||z||` within a reasonable range.

### A5: Presentation mode (Mailbox vs Fifo)

The gan-slider uses `PresentMode::Mailbox` (triple-buffered, no vsync cap) to
maximize throughput and show the true GAN inference rate.  The spectrogram viewer
uses `Fifo` (vsync-capped).  If tearing is visible, switch to `Fifo`.

### A6: Image transfer overhead

Each frame transfers ~3 MB of RGB pixels through a Unix pipe (stdin→stdout).
On this hardware (GB10), this costs ~1-2ms per frame.  For higher resolution
models or faster GPUs, shared memory (e.g. `/dev/shm` mmap) would reduce this
to near-zero.

### A7: cuDNN compatibility

The Jetson GB10 with CUDA 13.0 requires `libcudnn9-cuda-13` specifically.
Installing `libcudnn9-cuda-12` causes a silent crash (`Cannot load symbol
cublasLtGetVersion`) on any `conv2d` operation.  The system-level ldconfig was
also patched to include `/usr/lib/aarch64-linux-gnu/libcudss/13`.

### A8: Python venv path

The Rust `GanClient` hardcodes the venv at `{pygan_dir}/.venv/bin/python`.  If
the user uses a different Python environment, they'll need to adjust the path or
create a symlink.
