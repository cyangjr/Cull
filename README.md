## Cull
AI-powered photo selection tool that automatically screens and ranks photography sessions using computer vision, saliency detection, and composition analysis.

### Quickstart
**Install:**

```bash
# macOS / Linux
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Windows
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**Run:**

```bash
streamlit run app.py
```

### GPU Acceleration

The pipeline auto-detects the best available device at startup and logs it to the terminal (e.g. `[Cull] Device: MPS (Apple Silicon)`). The selected device is also shown in the Streamlit sidebar.

| Hardware | Behavior |
|---|---|
| NVIDIA (CUDA) | GPU batch mode — sharpness, motion blur, saliency, aesthetic computed on GPU |
| Apple Silicon (MPS) | GPU batch mode — same as CUDA via Metal Performance Shaders |
| AMD / Radeon | CPU mode — multiprocessing used instead (ROCm not supported) |
| CPU only | CPU multiprocessing mode |

**Install PyTorch for your platform:**

- Apple Silicon: `pip install torch torchvision` (MPS is included in standard builds)
- NVIDIA CUDA: follow the [PyTorch install selector](https://pytorch.org/get-started/locally/) to get the right CUDA build
- CPU only: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`

GPU acceleration is optional — the pipeline falls back to CPU automatically if PyTorch is not installed or no supported GPU is found.

### Configuration

Key settings in `config.yaml`:

| Setting | Default | Description |
|---|---|---|
| `device` | `auto` | `auto` / `cpu` / `cuda` / `mps` |
| `gpu_batch_size` | `8` | Images per GPU batch (reduce if OOM) |
| `num_workers` | `null` | CPU worker count (`null` = auto, `0` = sequential) |
| `sharpness_gate_threshold` | `0.3` | Images below this score are culled |
| `batch_size` | `16` | Images per CPU batch |

The sharpness threshold can also be tuned live in the Streamlit sidebar without editing `config.yaml`.
