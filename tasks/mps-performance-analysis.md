# MPS Performance Analysis
_Date: 2026-04-16 | Context: 84 photos taking >200s (~2.4s/image)_

---

## Is MPS Being Utilized?

**Yes, but poorly.** `DeviceManager.detect()` correctly identifies MPS and routes to `_run_gpu_batch()`. However, the GPU utilization is largely wasted due to several structural problems below.

---

## Bottleneck Analysis

### 1. Fake GPU batching (biggest issue)
Every "GPU batch" method loops over images one at a time (`scorer.py:99`, `detector.py:111`, `scorer.py:226`):

```python
# SharpnessScorer.score_batch_gpu
for region in regions:
    t = torch.from_numpy(gray).to(dev).unsqueeze(0).unsqueeze(0)
    lap = F.conv2d(t, lap_kernel, padding=1)
    v = float(lap.var().item())   # ← blocks until GPU finishes, then readback
```

Each iteration triggers:
- numpy → CPU tensor → `.to(mps)` (Metal buffer copy)
- submit a tiny 3×3 conv kernel
- `.item()` (synchronize + readback)

MPS has significant per-dispatch overhead. Sending 8 images one-by-one means 8× the dispatch overhead, not 1×. This completely negates the GPU benefit for small kernels like Laplacian.

Same problem in `MotionBlurDetector.detect_batch_gpu` and `SaliencyDetector.detect_batch_gpu`.

### 2. Full-resolution images passed to MediaPipe
Face detection (`FaceDetector.detect`) runs on every image at native resolution (likely 20–50 MP). MediaPipe BlazeFace on a 24MP image at CPU speed is ~0.2–0.5s/image. For 84 photos: **17–42 seconds just for face detection**, with no GPU involvement.

### 3. Full-resolution images throughout the entire pipeline
The pipeline never downsamples to a working resolution. Sharpness, exposure, WB, aesthetic, saliency — all run on the full image. A 6000×4000 image has 24M pixels; a 1500×1000 thumbnail has 1.5M. Most scoring is invariant to resolution above a few hundred pixels.

### 4. Sequential image loading with no prefetching
Loading is strictly sequential — disk I/O blocks the GPU from doing anything. While the GPU processes batch N, the CPU should be loading batch N+1.

### 5. Aesthetic scorer is NOT using the GPU
`orchestrator.py:395-400` calls `self.aesthetic_scorer.score(r)` in a CPU loop even in `_run_gpu_batch`. The `score_batch_gpu` method exists on `AestheticScorer` but is never called — dead code.

---

## Optimization Opportunities (Ranked by Impact)

### High Impact

**A. Downsample images to a working resolution early** (est. 3–5× speedup)

After loading, resize everything to `max_side=1500px` before any scoring. Face detection, sharpness, saliency — none need 24MP to be accurate. This is the single biggest lever.

```python
# In ImageLoader.load(), after convert("RGB"):
MAX_SIDE = 1500
h, w = image_np.shape[:2]
if max(h, w) > MAX_SIDE:
    scale = MAX_SIDE / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    image_np = cv2.resize(image_np, (new_w, new_h), interpolation=cv2.INTER_AREA)
```

**B. True GPU batching — pad regions to same size, run one conv2d** (est. 2× speedup)

Stack all regions into a single `(N, 1, H, W)` tensor, run one `F.conv2d`, compute variance over each sample. Eliminates N−1 Metal dispatch roundtrips per batch.

**C. Prefetch images in a background thread** (hides I/O latency)

Use `threading.Thread` + a queue. While GPU processes batch K, a thread loads batch K+1. Safe on MPS (unlike multiprocessing which can't share CUDA/MPS contexts).

### Medium Impact

**D. Wire up `aesthetic_scorer.score_batch_gpu`** (already written, not called)

Replace the CPU loop in `_run_gpu_batch` step 9 with the existing `score_batch_gpu` method on `AestheticScorer`.

**E. Resize images before MediaPipe**

BlazeFace works well at 320×320. Pass a thumbnail to the detector, scale bbox coordinates back up. Could cut face detection time by 5–10×.

### Lower Impact

**F. Increase `gpu_batch_size`**

Current default is 8. Apple Silicon has unified memory (no separate VRAM cap), so pushing to 32–64 is safe and amortizes MPS dispatch overhead across more images.

**G. Saliency already skipped on portrait shots** ✓

If `has_faces == True`, saliency is skipped (`scene_type == "scene"` branch). Already correct.

---

## Quick Wins (Zero or Minimal Code)

| Win | Change | Est. Speedup |
|-----|--------|-------------|
| Larger GPU batch | `gpu_batch_size: 32` in config.yaml | ~1.3× |
| Wire aesthetic GPU | Call `score_batch_gpu` in orchestrator.py step 9 | ~1.1× |
| Working resolution cap | ~5 lines in ImageLoader.load() | 3–5× |

---

## Suggested Implementation Order

1. **Resolution cap in ImageLoader** — biggest bang, minimal risk
2. **True GPU batching in SharpnessScorer** — restructure `score_batch_gpu` to use padded batch tensor
3. **Wire aesthetic GPU** — already implemented, just needs to be called
4. **Prefetch thread** — thread-safe on MPS, pairs well with larger batches
5. **MediaPipe resize** — needs bbox coordinate scaling

---

## Files to Modify

| File | Change |
|------|--------|
| `pipeline/utils.py` | Add resolution cap in `ImageLoader.load()` |
| `pipeline/scorer.py` | True batch tensor in `SharpnessScorer.score_batch_gpu` |
| `pipeline/orchestrator.py` | Call `aesthetic_scorer.score_batch_gpu` in step 9 |
| `pipeline/detector.py` | Resize before MediaPipe, scale bbox back |
| `config.yaml` | Increase `gpu_batch_size` to 32 |
