# GPU Setup Guide

Cull supports GPU acceleration via PyTorch on NVIDIA (CUDA) and Apple Silicon (MPS).
Without PyTorch the pipeline runs entirely on CPU — no change in behaviour or results.

---

## Quick Start

### Windows — NVIDIA GPU (CUDA)

```bash
# Activate your virtual environment first
.venv\Scripts\activate

# Install PyTorch with CUDA 12.1 support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

For other CUDA versions (11.8, 12.4) see https://pytorch.org/get-started/locally/

### macOS — Apple Silicon (M1 / M2 / M3)

```bash
# Activate your virtual environment first
source .venv/bin/activate

# MPS support is included in the standard PyTorch build
pip install torch torchvision
```

Requires macOS 12.3 or later.

### CPU-only (no GPU)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Use this if you want torch installed but don't have a GPU, or want a smaller download.

---

## Verify Installation

```bash
# Check which device Cull will use
python -c "from pipeline.utils import DeviceManager; print(DeviceManager().detect())"
# Expected: "cuda", "mps", or "cpu"

# Check CUDA directly
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# Check MPS directly
python -c "import torch; print('MPS:', torch.backends.mps.is_available())"
```

---

## Configuration

Edit `config.yaml` to control device selection:

```yaml
# Device selection
device: auto          # auto | cpu | cuda | mps

# GPU batch size — images processed simultaneously on GPU
# Increase for better GPU utilisation (needs more VRAM)
# Decrease (e.g. 4) if you hit out-of-memory errors
gpu_batch_size: 8
```

| `device` value | Behaviour |
|----------------|-----------|
| `auto` (default) | Prefer CUDA > MPS > CPU |
| `cpu` | Force CPU + multiprocessing (ignores GPU) |
| `cuda` | Use CUDA if available, else auto-detect |
| `mps` | Use MPS if available, else auto-detect |

---

## GPU vs CPU Mode

When a GPU is detected the pipeline switches to **GPU-batch mode**:

- Images are loaded one-by-one (I/O bound, no benefit from parallelism)
- A batch of `gpu_batch_size` images is accumulated, then all GPU ops run together
- Multiprocessing (`num_workers`) is **disabled** — CUDA contexts cannot be safely shared across spawned processes
- Every GPU stage has a CPU fallback if torch raises (e.g. OOM)

**Stages accelerated on GPU:**

| Stage | GPU op | Expected speedup |
|-------|--------|-----------------|
| Sharpness scoring | `torch.fft` Laplacian conv2d | 3–8× CUDA, 2–5× MPS |
| Motion blur detection | `torch.fft.fft2` (FFT) | 10–50× CUDA, 5–10× MPS |
| Saliency detection | Sobel conv2d | 2–5× |
| Aesthetic scoring | RGB→Gray/HSV tensor math | 2× |

**Stages that remain on CPU:**
- Image loading (I/O bound)
- Face detection (MediaPipe manages its own XNNPACK/Metal acceleration)
- Exposure + white balance scoring (sub-millisecond, not worth GPU overhead)
- Composition tagging (operates on 256×256 crops, negligible)
- Deduplication + ranking (sequential by design)

---

## Tuning `gpu_batch_size`

| VRAM | Recommended `gpu_batch_size` |
|------|------------------------------|
| 4 GB | 4 |
| 8 GB | 8 (default) |
| 12 GB | 16 |
| 24 GB+ | 32 |

If you see `RuntimeError: CUDA out of memory`, reduce `gpu_batch_size` in `config.yaml`.

---

## Troubleshooting

**"cuda" detected but pipeline still runs slowly**
- Confirm `device: auto` in config.yaml
- Run the verify command above — should print `cuda`
- Check that `gpu_batch_size` is at least 4; single-image GPU ops have high overhead

**MPS errors on Apple Silicon**
- Requires macOS 12.3+. Run `sw_vers` to check.
- Some torch ops have limited MPS support in older torch versions. Upgrade:
  `pip install --upgrade torch torchvision`
- The CPU fallback triggers automatically for any unsupported op.

**`ModuleNotFoundError: No module named 'torch'`**
- PyTorch is optional. Install it using one of the commands above.
- Without torch, the pipeline runs on CPU silently.

**Force CPU even with GPU available**
```yaml
device: cpu
```
This restores multiprocessing mode (`num_workers`) for CPU-only benchmarking.
