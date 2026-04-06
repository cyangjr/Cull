# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Cull is an AI-powered photo selection tool that automatically screens and ranks photography sessions using computer vision, saliency detection, and composition analysis. It provides a Streamlit web interface for processing folders of images and exporting curated selections.

## Development Setup

```bash
# Create virtual environment with Python 3.11
py -3.11 -m venv .venv

# Activate environment
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py

# Alternative: Run with explicit Python path
.venv\Scripts\python.exe -m streamlit run app.py
```

**Note:** This Phase A MVP runs on CPU. Future phases can leverage CUDA-enabled PyTorch builds for GPU acceleration.

## Architecture

### Core Pipeline Flow

The pipeline processes images in configurable batches through the following stages (orchestrator.py:43-154):

1. **Load** - ImageLoader reads images from folder, extracts EXIF data, applies EXIF orientation
2. **Route** - SceneRouter classifies images as "object" (has faces) or "scene" based on face detection
3. **Subject Detection** - ObjectDetector (uses face bbox as proxy) or SaliencyDetector (gradient-based heatmap)
4. **Face Detection** - FaceDetector uses MediaPipe BlazeFace to detect faces and extract eye regions
5. **Sharpness Scoring** - SharpnessScorer uses Laplacian variance on subject region (gate input)
6. **Motion Blur Detection** - MotionBlurDetector uses FFT directional analysis (metadata only, not gate)
7. **Exposure Scoring** - ExposureScorer analyzes histogram clipping (dark/bright)
8. **White Balance Scoring** - WhiteBalanceScorer measures RGB channel deviation
9. **Gate Check** - Only sharpness score evaluated; failed images released from memory immediately
10. **Aesthetic Scoring** - AestheticScorer uses contrast+saturation heuristic (0-10 scale, gate-passed only)
11. **Composition Tags** - CompositionTagger detects rule_of_thirds, symmetry, negative_space
12. **Duplicate Detection** - DuplicateFilter groups by timestamp window, compares perceptual hashes
13. **Final Scoring** - FinalScorer computes weighted sum (configurable), applies motion blur penalty
14. **Ranking** - Records sorted by final_score descending

### Key Components

**SessionManager** (session.py) - Top-level session lifecycle manager
- Creates CullPipeline instance with config
- Runs pipeline on folder with progress callbacks
- Exports gate-passed, non-duplicate records to JSON
- Loads previous results from JSON (pixel arrays excluded)

**CullPipeline** (orchestrator.py) - Main orchestration class
- Processes images in batches to control peak RAM usage
- Conditionally enables/disables stages via config toggles
- Releases pixel data after gate-fail or post-scoring based on config.release_pixel_data
- Applies perceptual hashing before pixel release for deduplication

**ImageRecord** (utils.py:38-111) - Central data structure
- Holds path, filename, pixel arrays (image, saliency_map), EXIF metadata
- Scene classification, detection results (subject_bbox, eye_region, saliency_peak_region)
- All scores: sharpness, exposure, white_balance, aesthetic, motion_blur_detected
- Composition tags, duplicate flags, perceptual hash, final score, gate status
- Exports to JSON dict (excludes pixel arrays and eye_region)

**PipelineConfig** (config.py) - YAML-based configuration
- Loads from config.yaml with dataclass defaults as fallback
- final_score_weights: adjustable component weights (sharpness, exposure, white_balance, aesthetic)
- sharpness_gate_threshold: gate decision threshold (default 0.3)
- Feature toggles: enable_router, enable_object_detector, enable_saliency_detector, enable_face_detector, enable_motion_blur, enable_aesthetic, enable_composition_tags, enable_dedup
- Memory/perf: batch_size (default 16), release_pixel_data (default true)
- Duplicate detection: hash_threshold (Hamming distance), timestamp_window_s

**FaceDetector** (detector.py:77-177) - MediaPipe-based face detection
- Downloads BlazeFace model to `.cache/mediapipe/` on first run
- Detects faces with configurable min_detection_confidence
- Extracts eye_region from upper ~65% of face bounding box
- Provides detailed error messages for MediaPipe import/initialization failures

### Streamlit UI (app.py)

- **Sidebar inputs:** folder path, sharpness threshold slider, min score filter, show gate-failed toggle
- **Load previous results:** JSON upload reconstructs records without pixel arrays
- **Run cull button:** starts pipeline with progress bar and status updates
- **Export button:** saves gate-passed non-duplicates to export.json
- **Results table:** sortable DataFrame showing all scores, metadata, flags
- **Thumbnails:** displays top 10 and bottom 10 images with scores (lazy-loaded via load_thumbnail cache)

### Configuration Tuning

The Streamlit UI allows quick sharpness_gate_threshold tuning without editing config.yaml (app.py:56). Other config values require editing config.yaml and restarting the app.

Final score weights can be adjusted in config.yaml to prioritize different quality aspects:
```yaml
final_score_weights:
  sharpness: 0.4
  exposure: 0.15
  white_balance: 0.15
  aesthetic: 0.3
```

### Phase Notes

This is a **Phase A MVP** implementing a complete end-to-end pipeline with:
- Heuristic-based subject detection (no YOLO yet)
- Gradient-based saliency detection (no deep learning saliency yet)
- Contrast/saturation aesthetic scoring (no NIMA model yet)
- MediaPipe face detection (production-ready)

Future phases can upgrade ObjectDetector (YOLO), SaliencyDetector (deep learning), and AestheticScorer (NIMA) while maintaining the same pipeline interface.

### Memory Management

The pipeline aggressively releases pixel data to handle large photo sessions:
- Gate-failed images: pixels released immediately after gate check (orchestrator.py:106-107)
- Gate-passed images: saliency_map released after saliency detection if enabled (orchestrator.py:82-83)
- Gate-passed images: perceptual hash computed, then image and saliency_map released before batch completes (orchestrator.py:124-136)

### Device Detection

DeviceManager (utils.py:15-35) auto-detects CUDA/MPS/CPU at runtime. Currently only used by ModelRegistry (not yet active in Phase A). Future phases will pass device to PyTorch-based models.

### Testing Approach

When adding new features or fixing bugs:
1. Test with a small folder (5-10 images) covering edge cases: blurry, dark, overexposed, portraits, landscapes
2. Verify gate behavior by adjusting sharpness threshold in UI
3. Check export.json structure and reloading via JSON upload
4. For scorer/detector changes, inspect individual scores in results DataFrame
5. For memory-intensive changes, test with 100+ image folder and monitor RAM usage

## Developer Workflow

### Core Principles

- **Simplicity First**: Make every change as simple as possible. Impact minimal code.
- **No Laziness**: Find root causes. No temporary fixes. Senior developer standards.
- **Minimal Impact**: Changes should only touch what's necessary. Avoid introducing bugs.

### 1. Plan Node Default
- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan immediately – don't keep pushing
- Use plan mode for verification steps, not just building
- Write detailed specs upfront to reduce ambiguity

### 2. Subagent Strategy
- Use subagents liberally to keep main context window clean
- Offload research, exploration, and parallel analysis to subagents
- For complex problems, throw more compute at it via subagents
- One tack per subagent for focused execution

### 3. Self-Improvement Loop
- After ANY correction from the user: update `tasks/lessons.md` with the pattern
- Write rules for yourself that prevent the same mistake
- Ruthlessly iterate on these lessons until mistake rate drops
- Review lessons at session start for relevant project

### 4. Verification Before Done
- Never mark a task complete without proving it works
- Diff behavior between main and your changes when relevant
- Ask yourself: "Would a staff engineer approve this?"
- Run tests, check logs, demonstrate correctness

### 5. Demand Elegance (Balanced)
- For non-trivial changes: pause and ask "is there a more elegant way?"
- If a fix feels hacky: "Knowing everything I know now, implement the elegant solution"
- Skip this for simple, obvious fixes – don't over-engineer
- Challenge your own work before presenting it

### 6. Autonomous Bug Fixing
- When given a bug report: just fix it. Don't ask for hand-holding
- Point at logs, errors, failing tests – then resolve them
- Zero context switching required from the user
- Go fix failing CI tests without being told how

### Task Management

1. **Plan First**: Write plan to `tasks/todo.md` with checkable items
2. **Verify Plan**: Check in before starting implementation
3. **Track Progress**: Mark items complete as you go
4. **Explain Changes**: High-level summary at each step
5. **Document Results**: Add review section to `tasks/todo.md`
6. **Capture Lessons**: Update `tasks/lessons.md` after corrections
