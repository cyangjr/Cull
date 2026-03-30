## Cull
AI-powered photo selection tool that automatically screens and ranks photography sessions using computer vision, saliency detection, and composition analysis.

### Quickstart (Phase A MVP)
- **Install**:

```bash
py -3.11 -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

- **Run**:

```bash
streamlit run app.py
```

### Notes on CUDA
This Phase A MVP runs on CPU. Later phases can prefer CUDA if you install a CUDA-enabled PyTorch build (recommended to follow the official PyTorch install selector for your CUDA version).
