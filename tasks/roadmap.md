# Cull Development Roadmap

## Phase B: GPU Acceleration & Performance (HIGH PRIORITY)

### 1. CUDA & Apple Silicon Support ⚡ [PRIORITY 1]
**Goal:** Enable GPU acceleration for future ML models (YOLO, NIMA, deep saliency)

**Current State:**
- DeviceManager.detect() already exists (utils.py:19-30)
- CUDA/MPS detection implemented but not utilized
- Phase A MVP runs CPU-only

**Tasks:**
- [ ] Install and test PyTorch with CUDA support on Windows
- [ ] Install and test PyTorch with MPS support on Apple Silicon
- [ ] Create device benchmarking script to measure speedup
- [ ] Document PyTorch installation instructions in README for both platforms
- [ ] Add device info logging to pipeline startup
- [ ] Test memory usage patterns on GPU vs CPU

**Acceptance Criteria:**
- Pipeline correctly detects and uses CUDA when available
- Pipeline correctly detects and uses MPS when available
- CPU fallback works seamlessly
- README has clear platform-specific installation instructions
- Performance benchmarks documented (CPU vs CUDA vs MPS)

**Estimated Effort:** 2-3 hours
**Dependencies:** None
**Risks:** CUDA version compatibility, MPS driver issues

---

### 2. Performance Profiling & Optimization 📊 [PRIORITY 2]
**Goal:** Identify and optimize bottlenecks before adding heavier models

**Tasks:**
- [ ] Add timing instrumentation to each pipeline stage
- [ ] Create performance profiling script with sample data (100+ images)
- [ ] Profile memory usage per stage
- [ ] Identify top 3 bottlenecks
- [ ] Optimize batch processing strategy
- [ ] Add progress reporting with ETA calculation
- [ ] Create performance regression test

**Deliverables:**
- `pipeline/profiler.py` - Timing and memory profiling
- `docs/performance.md` - Benchmark results and optimization notes
- ETA display in Streamlit UI

**Estimated Effort:** 3-4 hours
**Dependencies:** None

---

## Phase C: ML Model Upgrades (MEDIUM PRIORITY)

### 3. YOLO Object Detection 🎯 [PRIORITY 3]
**Goal:** Replace heuristic ObjectDetector with real object detection

**Current State:**
- ObjectDetector expands face bbox as proxy (detector.py:13-30)
- Ready to swap with minimal interface changes

**Tasks:**
- [ ] Research YOLO model selection (YOLOv8-nano vs v8-small for speed/accuracy)
- [ ] Integrate ultralytics YOLO library
- [ ] Update ObjectDetector.detect() to use YOLO
- [ ] Add YOLO confidence threshold to config
- [ ] Update ModelRegistry to manage YOLO lifecycle
- [ ] Test subject bbox accuracy vs heuristic
- [ ] Add GPU memory management for YOLO model
- [ ] Update CLAUDE.md with YOLO integration details

**Acceptance Criteria:**
- YOLO detects people, pets, objects better than face-bbox heuristic
- Configurable confidence threshold
- GPU acceleration working
- Memory footprint acceptable (batch_size tuning)
- Backwards compatible with Phase A data

**Estimated Effort:** 4-6 hours
**Dependencies:** GPU acceleration (Task 1)

---

### 4. Deep Learning Saliency Detection 👁️ [PRIORITY 4]
**Goal:** Replace gradient-based saliency with learned model

**Current State:**
- SaliencyDetector uses Sobel gradient magnitude (detector.py:33-74)
- Produces heatmap and peak bbox

**Tasks:**
- [ ] Research saliency models (U2-Net, BASNet, F3Net)
- [ ] Select lightweight model for real-time inference
- [ ] Integrate saliency model
- [ ] Update SaliencyDetector.detect() interface
- [ ] Compare saliency quality vs gradient baseline
- [ ] Add model to ModelRegistry
- [ ] Benchmark GPU vs CPU inference time
- [ ] Add toggle in config: "saliency_method: gradient | model"

**Acceptance Criteria:**
- Model-based saliency outperforms gradient on portrait/landscape test set
- <2s inference time per image on GPU
- Memory-efficient (release after peak extraction)
- Configurable fallback to gradient method

**Estimated Effort:** 5-7 hours
**Dependencies:** GPU acceleration (Task 1)

---

### 5. NIMA Aesthetic Scorer 🎨 [PRIORITY 5]
**Goal:** Replace heuristic aesthetic score with learned model

**Current State:**
- AestheticScorer uses contrast + saturation (scorer.py:143-162)
- 0-10 scale output

**Tasks:**
- [ ] Research NIMA implementations (TensorFlow vs PyTorch)
- [ ] Source pre-trained NIMA model (AVA dataset)
- [ ] Integrate NIMA into AestheticScorer
- [ ] Validate 0-10 score range compatibility
- [ ] A/B test: heuristic vs NIMA on curated test set
- [ ] Add model to ModelRegistry with keep_loaded=True (small model)
- [ ] Document score interpretation in CLAUDE.md
- [ ] Add toggle: "aesthetic_method: heuristic | nima"

**Acceptance Criteria:**
- NIMA scores correlate with manual human ranking
- <500ms inference per image
- Seamless integration with existing FinalScorer weights
- Backwards compatible with Phase A exports

**Estimated Effort:** 6-8 hours
**Dependencies:** GPU acceleration (Task 1)

---

## Phase D: Testing & Quality (MEDIUM PRIORITY)

### 6. Test Suite 🧪 [PRIORITY 6]
**Goal:** Prevent regressions as models are upgraded

**Tasks:**
- [ ] Create `tests/` directory structure
- [ ] Add pytest and pytest-cov to requirements-dev.txt
- [ ] Unit tests for scorers with synthetic images
- [ ] Unit tests for detectors with known-good samples
- [ ] Integration test: full pipeline with 10 test images
- [ ] Regression test: compare scores against Phase A baseline
- [ ] Memory leak test: process 1000 images, check RAM stable
- [ ] Config validation tests
- [ ] CI/CD setup (GitHub Actions) - optional
- [ ] Achieve >80% code coverage

**Deliverables:**
- `tests/unit/test_scorers.py`
- `tests/unit/test_detectors.py`
- `tests/integration/test_pipeline.py`
- `tests/fixtures/` - Sample images with known properties
- `requirements-dev.txt`

**Estimated Effort:** 8-10 hours
**Dependencies:** None (but recommend before Task 3-5)

---

### 7. Logging & Observability 📝 [PRIORITY 7]
**Goal:** Debug production issues and track performance

**Tasks:**
- [ ] Add structured logging (loguru or stdlib logging)
- [ ] Log level: DEBUG, INFO, WARNING, ERROR
- [ ] Log pipeline stage timing
- [ ] Log skipped files with reason (corrupt, unsupported, etc.)
- [ ] Log device selection (CPU/CUDA/MPS)
- [ ] Log model loading/unloading events
- [ ] Optional: export logs to file with rotation
- [ ] Add --verbose flag for CLI mode (future)
- [ ] Update CLAUDE.md with logging conventions

**Deliverables:**
- `pipeline/logger.py`
- Configurable log level in config.yaml
- Log output in Streamlit sidebar (optional)

**Estimated Effort:** 3-4 hours
**Dependencies:** None

---

## Phase E: UX Enhancements (LOW PRIORITY)

### 8. Streamlit UI Improvements 🖥️ [PRIORITY 8]
**Goal:** Better user experience for photo culling workflow

**Tasks:**
- [ ] Add batch export options (copy files to folder, not just JSON)
- [ ] Add comparison view (side-by-side top vs bottom)
- [ ] Add manual override: mark image as keep/reject
- [ ] Add search/filter by metadata (ISO, focal length, timestamp)
- [ ] Add histogram of final scores (distribution chart)
- [ ] Add "re-run scoring" without re-processing (adjust weights live)
- [ ] Add export format: CSV for Excel analysis
- [ ] Add thumbnail grid view (replace top10/bottom10 lists)
- [ ] Add keyboard shortcuts (j/k navigation, space to toggle keep)
- [ ] Persist UI state across sessions (session_state in JSON)

**Estimated Effort:** 6-10 hours
**Dependencies:** None

---

### 9. CLI Interface 🖱️ [PRIORITY 9]
**Goal:** Scriptable batch processing without Streamlit

**Tasks:**
- [ ] Create `cli.py` with argparse
- [ ] Support: `python cli.py --input /path/to/photos --output results.json`
- [ ] Add flags: --config, --batch-size, --device, --verbose
- [ ] Add progress bar (tqdm)
- [ ] Support stdin/stdout for piping
- [ ] Add dry-run mode (report what would be culled)
- [ ] Add filter flags: --min-score, --no-duplicates, --faces-only
- [ ] Document CLI usage in README
- [ ] Add shell completion (optional)

**Estimated Effort:** 4-5 hours
**Dependencies:** None

---

## Phase F: Advanced Features (FUTURE)

### 10. Duplicate Detection Improvements 🔍
**Goal:** Catch more duplicate types (burst mode, similar but not identical)

**Tasks:**
- [ ] Add visual similarity threshold (not just perceptual hash)
- [ ] Detect burst sequences (timestamp + motion analysis)
- [ ] Keep best from burst (highest sharpness)
- [ ] Add manual "mark as duplicate" override in UI
- [ ] Improve timestamp clustering (EXIF + filesystem fallback)

**Estimated Effort:** 4-6 hours

---

### 11. Composition Analysis Enhancements 📐
**Goal:** More sophisticated composition tagging

**Tasks:**
- [ ] Leading lines detection (Hough transform + clustering)
- [ ] Golden ratio detection (extend rule of thirds)
- [ ] Frame within frame detection
- [ ] Color harmony detection (complementary, analogous)
- [ ] Add composition score (weighted tags)
- [ ] Train custom composition classifier (optional, long-term)

**Estimated Effort:** 8-12 hours

---

### 12. Export & Integration 🔌
**Goal:** Integrate with photographer workflows

**Tasks:**
- [ ] Export to Lightroom sidecar files (.xmp)
- [ ] Export to star ratings (5-star from final_score)
- [ ] Export to Adobe Bridge metadata
- [ ] Copy kept files to new folder (preserve structure)
- [ ] Generate contact sheet PDF
- [ ] Integration with cloud storage (Google Photos, Dropbox)

**Estimated Effort:** 6-8 hours per integration

---

## Immediate Next Steps (Recommended Order)

### Week 1: GPU Foundation
1. **Task 1: CUDA & Apple Silicon Support** (2-3 hrs)
   - Get PyTorch GPU acceleration working
   - Document platform-specific setup

2. **Task 2: Performance Profiling** (3-4 hrs)
   - Baseline current performance
   - Identify bottlenecks before adding heavy models

### Week 2: Model Upgrades (Choose One)
Pick based on impact:
- **Option A: Task 3 (YOLO)** - Best for portrait/people photographers
- **Option B: Task 5 (NIMA)** - Best for landscape/aesthetic-focused photographers
- **Option C: Task 6 (Testing)** - Best if planning multiple model changes

### Week 3-4: Expand
- Add remaining ML models (Tasks 3-5)
- Build test suite (Task 6)
- Add logging (Task 7)

---

## Success Metrics

**Phase B (GPU):**
- [ ] 3-10x speedup on GPU vs CPU for future models
- [ ] <5 min to process 1000 photos on GPU

**Phase C (ML Models):**
- [ ] YOLO improves subject detection accuracy >20% vs heuristic
- [ ] Deep saliency improves scene composition detection
- [ ] NIMA correlates >0.7 with manual aesthetic rankings

**Phase D (Quality):**
- [ ] >80% test coverage
- [ ] Zero regressions when upgrading models
- [ ] All pipeline stages logged with timing

---

## Resource Links

### CUDA/MPS Setup
- PyTorch Install: https://pytorch.org/get-started/locally/
- CUDA Toolkit: https://developer.nvidia.com/cuda-downloads
- Apple MPS Guide: https://developer.apple.com/metal/pytorch/

### ML Models
- YOLOv8: https://github.com/ultralytics/ultralytics
- U2-Net Saliency: https://github.com/xuebinqin/U-2-Net
- NIMA PyTorch: https://github.com/titu1994/neural-image-assessment

### Testing
- Pytest: https://docs.pytest.org/
- Image Test Fixtures: https://github.com/recurser/exif-samples

---

## Notes

- All tasks maintain backwards compatibility with Phase A exports
- Config toggles allow A/B testing (heuristic vs model)
- Memory management patterns stay consistent
- ModelRegistry ready for multi-model orchestration
- Device detection already implemented, just needs utilization
