# CPU Performance Optimization Analysis

**Analysis Date:** 2026-04-06
**Target:** Phase A MVP - CPU-only performance improvements
**Priority:** HIGH (Before GPU work)

---

## Executive Summary

There are **significant CPU optimization opportunities** that can deliver **2-5x speedup** without adding GPU complexity. These optimizations are simpler to implement than GPU support and will benefit all users, not just those with high-end hardware.

**Recommended approach:** Implement CPU optimizations FIRST, then add GPU as optional acceleration later.

---

## Current Bottleneck Analysis

### 1. Sequential Processing (CRITICAL BOTTLENECK) 🔴

**Location:** `orchestrator.py:56-108`

**Current Pattern:**
```python
for start in range(0, len(paths), batch_size):
    batch_paths = paths[start : start + batch_size]
    batch: list[ImageRecord] = []
    for p in batch_paths:
        batch.append(self.image_loader.load(str(p)))  # Sequential load

    for i, r in enumerate(batch, start=start):  # Sequential processing
        # All stages run sequentially on single core
        self.face_detector.detect(r)
        self.sharpness_scorer.score(r)
        # ... etc
```

**Problem:**
- Only uses 1 CPU core
- Modern CPUs have 8-16+ cores sitting idle
- I/O bound (image loading) mixed with CPU bound (scoring) in same loop

**Impact:** **Estimated 80% of total pipeline time**

---

### 2. Redundant Image Conversions (MEDIUM BOTTLENECK) 🟡

**Location:** Multiple stages convert RGB→Gray repeatedly

**Examples:**
- `scorer.py:46` - SharpnessScorer: `cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)`
- `scorer.py:61` - ExposureScorer: `cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)`
- `scorer.py:124` - MotionBlurDetector: `cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)`
- `scorer.py:143` - AestheticScorer: `cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)`
- `scorer.py:207` - CompositionTagger (symmetry): `cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)`
- `scorer.py:216` - CompositionTagger (negative space): `cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)`

**Problem:**
- Same grayscale conversion computed 4-6 times per image
- Each conversion processes every pixel (expensive for 24MP images)

**Impact:** **Estimated 5-10% of scoring time**

---

### 3. Expensive FFT Operations (MEDIUM BOTTLENECK) 🟡

**Location:** `scorer.py:123-140` - MotionBlurDetector

**Current Pattern:**
```python
gray = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)  # Good: downsample first
f = np.fft.fft2(gray)  # Expensive: 2D FFT
fshift = np.fft.fftshift(f)
mag = np.log(np.abs(fshift) + 1e-6)
```

**Problem:**
- FFT is computationally expensive even at 50% resolution
- Runs on ALL images, even those that will fail gate
- Only provides metadata (doesn't affect gate decision)

**Impact:** **Estimated 10-15% of pre-gate processing time**

---

### 4. Thumbnail Caching (LOW IMPACT) 🟢

**Location:** `app.py:16-30` - load_thumbnail

**Current State:**
- Already using `@st.cache_data` (good!)
- Opens file, applies EXIF rotation, generates thumbnail

**Opportunity:**
- Could pre-generate thumbnails during pipeline run
- Would improve UI responsiveness

**Impact:** **Estimated <5% (UI only, not pipeline)**

---

## Optimization Recommendations

### Priority 1: Multiprocessing for Image Loading & Scoring ⚡

**Estimated Speedup:** 3-6x (on 8-core CPU)

**Strategy: Pipeline Parallelism**

Split work into 3 concurrent pipelines:

1. **Loader Pool** (2-4 workers)
   - Parallel image loading with PIL/Pillow
   - EXIF extraction
   - Feed into processing queue

2. **Pre-gate Processor Pool** (4-8 workers)
   - Face detection
   - Sharpness scoring
   - Exposure, white balance
   - Gate check
   - Immediately discard gate-failed images

3. **Post-gate Processor Pool** (2-4 workers)
   - Aesthetic scoring
   - Composition tagging
   - Perceptual hashing

**Implementation:**
```python
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import Manager, Queue

def worker_load_image(path: str) -> ImageRecord:
    """Worker function for parallel image loading"""
    loader = ImageLoader()
    return loader.load(path)

def worker_pregate_process(record: ImageRecord, config: PipelineConfig) -> ImageRecord:
    """Worker function for pre-gate processing"""
    # Initialize scorers (once per worker)
    sharpness = SharpnessScorer()
    exposure = ExposureScorer()
    # ... etc

    sharpness.score(record)
    exposure.score(record)
    # Run gate check
    record.passed_gate = (record.sharpness_score or 0.0) >= config.sharpness_gate_threshold

    if not record.passed_gate:
        record.image = None  # Release immediately

    return record

# In CullPipeline.run():
with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
    # Stage 1: Parallel loading
    future_to_path = {executor.submit(worker_load_image, str(p)): p for p in paths}

    # Stage 2: Parallel pre-gate processing
    for future in as_completed(future_to_path):
        record = future.result()
        # Submit to pre-gate pool
        ...
```

**Pros:**
- Massive speedup on multi-core CPUs
- Relatively simple to implement
- Works on all platforms

**Cons:**
- Increased complexity (need proper queue management)
- Memory usage spike (multiple images in flight)
- May need batch size tuning

**Recommendation:** Implement with configurable `num_workers` in config.yaml

---

### Priority 2: Cache Grayscale Conversion 🎯

**Estimated Speedup:** 1.1-1.15x (10-15% faster scoring)

**Strategy:** Convert once, reuse everywhere

**Implementation:**
```python
# In ImageRecord dataclass (utils.py:39)
@dataclass(slots=True)
class ImageRecord:
    image: np.ndarray | None = None
    gray_cache: np.ndarray | None = None  # <-- Add cached grayscale
    # ... rest of fields

# Helper method
def get_gray(self) -> np.ndarray | None:
    if self.image is None:
        return None
    if self.gray_cache is None:
        self.gray_cache = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
    return self.gray_cache
```

**Update all scorers to use:**
```python
# Before:
gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)

# After:
gray = record.get_gray()
if gray is None:
    return
# Use region extraction if needed
```

**Pros:**
- Simple to implement
- No downsides
- Cumulative benefit across multiple stages

**Cons:**
- Slightly increased memory (1 channel vs 3)
- Need to clear cache when releasing pixel data

**Memory Impact:** +33% per image (1 gray channel vs 3 RGB channels), but released with `release_pixel_data`

---

### Priority 3: Defer Expensive Operations Until After Gate ⏱️

**Estimated Speedup:** 1.2-1.4x (20-40% faster for typical 70% gate-fail rate)

**Strategy:** Move expensive non-gate operations after gate check

**Current Order:**
```python
# Pre-gate:
face_detector.detect(r)        # Expensive (MediaPipe)
saliency_detector.detect(r)    # Expensive (gradients + blur)
sharpness_scorer.score(r)      # Medium (Laplacian)
motion_blur_detector.detect(r) # VERY EXPENSIVE (FFT) ← PROBLEM
exposure_scorer.score(r)       # Cheap (histogram)
white_balance_scorer.score(r)  # Cheap (mean RGB)
_run_gate(r)                   # Gate decision

# Post-gate (only 30% of images):
aesthetic_scorer.score(r)
composition_tagger.tag(r)
```

**Problem:** Motion blur detector runs FFT on 100% of images, but:
- Only provides metadata (not used in gate)
- Could run after gate on 30% of images instead

**Proposed Order:**
```python
# Minimal pre-gate (fast):
sharpness_scorer.score(r)      # Required for gate
exposure_scorer.score(r)       # Cheap
white_balance_scorer.score(r)  # Cheap
_run_gate(r)                   # EARLY EXIT for 70% of images

if not r.passed_gate:
    continue  # Skip expensive operations

# Post-gate (expensive operations on 30% only):
face_detector.detect(r)        # Now only 30% of images
saliency_detector.detect(r)    # Now only 30% of images
motion_blur_detector.detect(r) # Now only 30% of images  ← BIG WIN
aesthetic_scorer.score(r)
composition_tagger.tag(r)
```

**Trade-off:** Routing (face detection → scene classification) would happen after gate. Solutions:
1. Use lightweight face detection (haar cascades instead of MediaPipe) for routing only
2. Accept that scene routing only works on gate-passed images
3. Make routing optional via config toggle

**Pros:**
- Huge speedup for typical culling workflows (70%+ rejection rate)
- No code changes to scorers themselves
- Just reorder orchestrator.py

**Cons:**
- Scene routing affected (may need separate lightweight detector)
- Composition tags won't have face/saliency data for early-failed images

**Recommendation:** Make this configurable: `gate_first_mode: bool` in config

---

### Priority 4: Optimize Motion Blur Detection 🔧

**Estimated Speedup:** 1.05-1.1x (5-10% faster if keeping motion blur pre-gate)

**Current Approach:**
```python
gray = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)  # 50% downscale
f = np.fft.fft2(gray)  # 2D FFT
```

**Alternative Approaches:**

**Option A: Variance of Laplacian (Faster)**
```python
def _laplacian_variance_check(self, image: np.ndarray) -> bool:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (0, 0), fx=0.25, fy=0.25)  # Even more aggressive downscale
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < MOTION_BLUR_THRESHOLD  # Low variance = blur
```
- **Pros:** Much faster than FFT, already used for sharpness
- **Cons:** Less specific (can't distinguish motion blur from defocus blur)

**Option B: Directional Gradient Analysis (Fast)**
```python
def _directional_gradient_check(self, image: np.ndarray) -> bool:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (0, 0), fx=0.25, fy=0.25)

    # Sobel in X and Y
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    gx_mag = np.abs(gx).mean()
    gy_mag = np.abs(gy).mean()

    # If one direction dominates, likely motion blur
    ratio = max(gx_mag, gy_mag) / (min(gx_mag, gy_mag) + 1e-6)
    return ratio > 1.5
```
- **Pros:** Faster than FFT, detects directionality
- **Cons:** Less robust than frequency-domain analysis

**Option C: Just Disable It (Simplest)**
- Add config toggle: `enable_motion_blur: false` (already exists!)
- Let users decide if they need this metadata

**Recommendation:**
1. Move motion blur to post-gate (Priority 3)
2. Add alternative fast method as config option
3. Document that FFT is most accurate but slowest

---

### Priority 5: Use Pillow-SIMD (Drop-in Replacement) 🚀

**Estimated Speedup:** 1.2-1.4x for image loading

**What is Pillow-SIMD?**
- Drop-in replacement for Pillow
- Uses SIMD instructions (AVX2, SSE4)
- 2-4x faster for common operations (resize, convert, transpose)

**Installation:**
```bash
pip uninstall pillow
pip install pillow-simd
```

**Changes Required:** **NONE** - it's a drop-in replacement!

**Impact Areas:**
- `utils.py:145-147` - Image loading and EXIF rotation (BIG IMPACT)
- `app.py:22-23` - Thumbnail generation (MEDIUM IMPACT)
- `orchestrator.py:130` - Perceptual hashing (SMALL IMPACT)

**Pros:**
- Zero code changes
- Works on all platforms (x86/x64 with AVX2)
- Free performance boost

**Cons:**
- Not available on some platforms (ARM without special build)
- Slightly more complex install (not on PyPI by default)

**Recommendation:**
- Add as optional optimization in README
- Include in requirements-fast.txt
- Test on Windows/Mac/Linux

---

### Priority 6: Lazy EXIF Extraction 📄

**Estimated Speedup:** 1.05-1.1x for image loading

**Current Approach:** `utils.py:158-204`
```python
def load(self, path: str) -> ImageRecord:
    img = Image.open(p)
    img = ImageOps.exif_transpose(img).convert("RGB")
    image_np = np.array(img)

    exif = self._extract_exif(path)  # Always extracts all EXIF ← SLOWISH
    return ImageRecord(...)
```

**Problem:**
- EXIF extraction opens file twice (once for pixels, once for metadata)
- Extracts ALL fields even if only timestamp is needed
- exifread library is relatively slow

**Optimization:**
```python
def load(self, path: str, extract_exif: bool = True) -> ImageRecord:
    img = Image.open(p)
    img = ImageOps.exif_transpose(img).convert("RGB")
    image_np = np.array(img)

    exif = {}
    if extract_exif:
        exif = self._extract_exif_fast(path)  # Only essential fields

    return ImageRecord(...)

def _extract_exif_fast(self, path: str) -> dict[str, Any]:
    """Extract only timestamp for deduplication"""
    try:
        img = Image.open(path)
        exif_raw = getattr(img, "getexif", lambda: None)()
        if exif_raw:
            return {
                "timestamp": exif_raw.get(36867) or exif_raw.get(306),
                # Skip other fields unless needed
            }
    except Exception:
        pass
    return {}
```

**Alternative:** Extract EXIF only for gate-passed images (since it's only used for dedup timestamp clustering)

**Pros:**
- Faster loading
- Simpler code
- EXIF only extracted when needed

**Cons:**
- Timestamp may be missing for some dedup scenarios
- Less metadata in export

**Recommendation:** Make full EXIF extraction optional via config: `extract_full_exif: bool`

---

## Implementation Roadmap

### Phase 1: Quick Wins (1-2 hours)
**Target: 1.3-1.5x speedup**

1. ✅ Install Pillow-SIMD
2. ✅ Cache grayscale conversions (add `gray_cache` to ImageRecord)
3. ✅ Add config toggle: `gate_first_mode: bool`
4. ✅ Update orchestrator to defer expensive ops when `gate_first_mode=true`

**Deliverables:**
- Updated `requirements-fast.txt` with pillow-simd
- Updated `utils.py` with gray_cache
- Updated `config.yaml` with gate_first_mode
- Updated `orchestrator.py` with conditional ordering

**Risk:** LOW - All changes are backwards compatible

---

### Phase 2: Multiprocessing (3-5 hours)
**Target: 3-6x speedup**

1. ✅ Add `num_workers: int` to config (default: cpu_count() - 1)
2. ✅ Implement ProcessPoolExecutor for parallel image loading
3. ✅ Implement parallel pre-gate processing
4. ✅ Add progress tracking across workers
5. ✅ Test memory usage with different worker counts
6. ✅ Document worker tuning in CLAUDE.md

**Deliverables:**
- Updated `orchestrator.py` with multiprocessing
- New `pipeline/parallel.py` module for worker functions
- Updated `config.yaml` with num_workers setting
- Performance benchmarking script

**Risk:** MEDIUM - Multiprocessing adds complexity, memory usage needs tuning

---

### Phase 3: Advanced Optimizations (2-3 hours)
**Target: Additional 1.1-1.2x speedup**

1. ✅ Implement faster motion blur detection (gradient-based)
2. ✅ Add config option: `motion_blur_method: "fft" | "gradient" | "laplacian"`
3. ✅ Optimize lazy EXIF extraction
4. ✅ Profile and tune batch sizes for different CPU counts

**Deliverables:**
- Updated `scorer.py` with alternative motion blur detectors
- Benchmark comparison of methods
- Updated config with method selection

**Risk:** LOW - All optimizations are opt-in via config

---

## Expected Results

### Before Optimization (Baseline)
- 1000 images @ 24MP on 8-core CPU
- Estimated time: **~15-20 minutes**
- CPU utilization: **12-15%** (single-threaded)

### After Phase 1 (Quick Wins)
- Estimated time: **~10-13 minutes**
- CPU utilization: **12-15%** (still single-threaded)
- Speedup: **1.3-1.5x**

### After Phase 2 (Multiprocessing)
- Estimated time: **~3-5 minutes**
- CPU utilization: **70-90%** (multi-threaded)
- Speedup: **4-6x** vs baseline

### After Phase 3 (Advanced)
- Estimated time: **~2.5-4 minutes**
- CPU utilization: **70-90%**
- Speedup: **5-8x** vs baseline

---

## Comparison: CPU Optimization vs GPU

### CPU Optimization (This Proposal)
**Pros:**
- Works on ALL machines (no special hardware)
- Simpler to implement and maintain
- Benefits 100% of users
- No driver compatibility issues
- Lower memory usage

**Cons:**
- Limited by CPU core count
- Not as fast as GPU for ML models (YOLO, NIMA)

**Implementation Time:** 6-10 hours total

**Speedup:** 5-8x for Phase A heuristics

---

### GPU Support (CUDA/MPS)
**Pros:**
- Much faster for ML models (10-50x)
- Essential for Phase C (YOLO, NIMA, deep saliency)

**Cons:**
- Only benefits users with GPUs (~30-40% of market)
- Driver compatibility headaches
- Higher memory usage
- More complex debugging
- PyTorch dependency (~2GB installed)

**Implementation Time:** 3-5 hours for setup, ongoing maintenance burden

**Speedup:** Minimal for Phase A (no ML models yet), huge for Phase C

---

## Final Recommendation

### Optimal Strategy: CPU First, GPU Later

**Week 1-2: Implement CPU Optimizations (This Document)**
1. Phase 1: Quick wins (1-2 hours) → 1.5x speedup
2. Phase 2: Multiprocessing (3-5 hours) → 5x speedup
3. Phase 3: Advanced (2-3 hours) → 6x speedup
4. **Total time: ~8 hours**
5. **Total speedup: ~6x**
6. **Benefits: 100% of users**

**Week 3-4: Add GPU Support**
- Install PyTorch with CUDA/MPS
- Update DeviceManager to utilize detected device
- Add GPU memory management
- **Benefits: 30-40% of users (those with GPUs)**
- **Speedup: Minimal for Phase A, essential for Phase C**

**Week 5+: Upgrade to ML Models (Phase C)**
- YOLO, NIMA, deep saliency now benefit from GPU
- CPU multiprocessing remains beneficial for non-ML stages
- Best of both worlds: parallel CPU + GPU acceleration

---

## Appendix: Profiling Script

To measure actual bottlenecks, add this profiling script:

```python
# scripts/profile_pipeline.py
import time
from pathlib import Path
from contextlib import contextmanager

@contextmanager
def timer(name: str):
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    print(f"{name}: {elapsed:.3f}s")

def profile_pipeline(folder: str):
    from pipeline import CullPipeline, PipelineConfig

    config = PipelineConfig.load()
    pipeline = CullPipeline(config)

    with timer("Total pipeline"):
        with timer("  Image loading"):
            paths = list(Path(folder).iterdir())
            records = [pipeline.image_loader.load(str(p)) for p in paths[:100]]

        with timer("  Pre-gate processing"):
            for r in records:
                with timer("    Face detection"):
                    pipeline.face_detector.detect(r)
                with timer("    Sharpness"):
                    pipeline.sharpness_scorer.score(r)
                with timer("    Motion blur"):
                    pipeline.motion_blur_detector.detect(r)
                # ... etc

if __name__ == "__main__":
    profile_pipeline("test_photos")
```

---

**Next Steps:**
1. Review this analysis
2. Approve Phase 1 quick wins
3. I'll implement multiprocessing (Phase 2)
4. Benchmark results
5. Then tackle GPU if still needed
