from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Callable

from .config import PipelineConfig
from .detector import FaceDetector, ObjectDetector, SaliencyDetector
from .router import SceneRouter
from .scorer import (
    AestheticScorer,
    CompositionTagger,
    DuplicateFilter,
    ExposureScorer,
    FinalScorer,
    MotionBlurDetector,
    SharpnessScorer,
    WhiteBalanceScorer,
)
from .utils import DeviceManager, ImageLoader, ImageRecord, ModelRegistry


class _StageTimer:
    """Accumulates wall-clock time per named stage and prints a summary report."""

    def __init__(self) -> None:
        self._totals: dict[str, float] = defaultdict(float)

    def add(self, stage: str, elapsed: float) -> None:
        self._totals[stage] += elapsed

    def report(self, n_images: int) -> None:
        total = sum(self._totals.values())
        per_img = total / max(1, n_images)
        print(f"\n[Cull] Timing — {n_images} images | {total:.1f}s total | {per_img:.2f}s/img")
        print(f"  {'Stage':<28} {'Total':>8}  {'%':>6}  {'Per img':>8}")
        print(f"  {'-' * 28} {'-' * 8}  {'-' * 6}  {'-' * 8}")
        for stage, t in sorted(self._totals.items(), key=lambda kv: -kv[1]):
            pct = 100.0 * t / (total + 1e-9)
            avg = t / max(1, n_images)
            print(f"  {stage:<28} {t:>7.1f}s  {pct:>5.1f}%  {avg:>7.3f}s")
        print()


class CullPipeline:
    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig.load()
        self.device = DeviceManager().get_device(self.config.device)
        print(f"[Cull] Device: {DeviceManager.describe(self.device)}")

        # Phase A: ModelRegistry not used yet, but present for Phase C shape.
        self.model_registry = ModelRegistry(device=self.device)

        self.image_loader = ImageLoader()
        self.router = SceneRouter()
        self.object_detector = ObjectDetector()
        self.saliency_detector = SaliencyDetector()
        self.face_detector = FaceDetector(min_detection_confidence=self.config.min_face_detection_confidence)
        self.sharpness_scorer = SharpnessScorer()
        self.motion_blur_detector = MotionBlurDetector()
        self.exposure_scorer = ExposureScorer()
        self.white_balance_scorer = WhiteBalanceScorer()
        self.aesthetic_scorer = AestheticScorer()
        self.composition_tagger = CompositionTagger()
        self.duplicate_filter = DuplicateFilter(self.config)
        self.final_scorer = FinalScorer(self.config)

    def run(
        self,
        folder_path: str,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> list[ImageRecord]:
        """
        Run the culling pipeline on a folder of images.

        Chooses the appropriate execution mode:
          - GPU (cuda/mps): sequential GPU-batch mode — multiprocessing is disabled
            because CUDA contexts cannot be safely shared across spawned processes.
          - CPU + workers > 1: parallel multiprocessing mode (existing).
          - CPU + workers ≤ 1: sequential CPU mode (existing).

        Args:
            folder_path: Path to folder containing images
            progress_callback: Optional callback(fraction, message) for progress updates

        Returns:
            List of ImageRecords sorted by final_score descending
        """
        # GPU mode bypasses multiprocessing entirely.
        if self.device in ("cuda", "mps"):
            return self._run_gpu_batch(folder_path, progress_callback)

        from .cpu_utils import get_safe_worker_count

        workers = get_safe_worker_count(self.config.num_workers)
        if workers <= 1:
            return self._run_sequential(folder_path, progress_callback)
        else:
            return self._run_parallel(folder_path, progress_callback, workers)

    def _run_sequential(
        self,
        folder_path: str,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> list[ImageRecord]:
        """Original sequential implementation (Phase A MVP)."""
        folder = ImageLoader._normalise_path(folder_path)
        paths = [p for p in sorted(folder.iterdir()) if p.is_file() and self.image_loader.is_supported(p)]
        n = max(1, len(paths))
        batch_size = max(1, int(self.config.batch_size))

        records: list[ImageRecord] = []
        timer = _StageTimer()

        # Process in batches to cap peak RAM.
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start : start + batch_size]
            batch: list[ImageRecord] = []
            for p in batch_paths:
                try:
                    t0 = time.perf_counter()
                    batch.append(self.image_loader.load(str(p)))
                    timer.add("load", time.perf_counter() - t0)
                except Exception:
                    continue

            # Stages 2-9: pre-gate processing
            for i, r in enumerate(batch, start=start):
                if progress_callback:
                    progress_callback(i / n, f"Processing {r.filename}")

                # 2) Route — face detection runs here; skip duplicate call in step 4
                if self.config.enable_router:
                    if self.config.enable_face_detector:
                        t0 = time.perf_counter()
                        self.face_detector.detect(r)
                        timer.add("face_detection", time.perf_counter() - t0)
                    t0 = time.perf_counter()
                    self.router.classify(r)
                    timer.add("routing", time.perf_counter() - t0)

                # 3) Detect subjects
                if self.config.enable_object_detector and r.scene_type == "object":
                    t0 = time.perf_counter()
                    self.object_detector.detect(r)
                    timer.add("object_detection", time.perf_counter() - t0)
                if self.config.enable_saliency_detector and r.scene_type == "scene":
                    t0 = time.perf_counter()
                    self.saliency_detector.detect(r)
                    timer.add("saliency", time.perf_counter() - t0)
                    if self.config.release_pixel_data:
                        r.saliency_map = None

                # 4) Face detection — only when router did not already run it
                if self.config.enable_face_detector and not self.config.enable_router:
                    t0 = time.perf_counter()
                    self.face_detector.detect(r)
                    timer.add("face_detection", time.perf_counter() - t0)

                # 5) Score sharpness
                t0 = time.perf_counter()
                self.sharpness_scorer.score(r)
                timer.add("sharpness", time.perf_counter() - t0)

                # 6) Motion blur detect (metadata; not gate input)
                if self.config.enable_motion_blur:
                    t0 = time.perf_counter()
                    self.motion_blur_detector.detect(r)
                    timer.add("motion_blur", time.perf_counter() - t0)

                # 7) Exposure score
                t0 = time.perf_counter()
                self.exposure_scorer.score(r)
                timer.add("exposure", time.perf_counter() - t0)

                # 8) White balance score
                t0 = time.perf_counter()
                self.white_balance_scorer.score(r)
                timer.add("white_balance", time.perf_counter() - t0)

                # 9) Gate check (sharpness only)
                self._run_gate(r)

                # Release full image for gate-failed records immediately.
                if self.config.release_pixel_data and not r.passed_gate:
                    r.image = None

            # Post-gate stages for this batch only.
            passed_batch = [r for r in batch if r.passed_gate]
            for r in passed_batch:
                # 10) Aesthetics
                if self.config.enable_aesthetic:
                    t0 = time.perf_counter()
                    self.aesthetic_scorer.score(r)
                    timer.add("aesthetic", time.perf_counter() - t0)

                # 11) Composition tags
                if self.config.enable_composition_tags:
                    t0 = time.perf_counter()
                    self.composition_tagger.tag(r)
                    timer.add("composition", time.perf_counter() - t0)

                # 13) Final score (pre-dedup)
                self.final_scorer.compute(r)

                # Precompute perceptual hash for dedup, then release pixels.
                if self.config.enable_dedup and r.image is not None:
                    try:
                        import imagehash
                        from PIL import Image

                        t0 = time.perf_counter()
                        r.perceptual_hash = str(imagehash.phash(Image.fromarray(r.image)))
                        timer.add("perceptual_hash", time.perf_counter() - t0)
                    except Exception:
                        pass

                if self.config.release_pixel_data:
                    r.image = None
                    r.saliency_map = None

            records.extend(batch)

        passed = [r for r in records if r.passed_gate]

        # Stage 12) Deduplicate across all gate-passed records
        if self.config.enable_dedup and len(passed) > 1:
            t0 = time.perf_counter()
            self.duplicate_filter.filter(passed)
            timer.add("dedup", time.perf_counter() - t0)

        # Recompute final scores for passed records after dedup flags.
        for r in passed:
            self.final_scorer.compute(r)

        # Progress wrap-up
        if progress_callback:
            progress_callback(1.0, "Ranking")

        timer.report(len(paths))
        return self.final_scorer.rank(records)

    def _run_gpu_batch(
        self,
        folder_path: str,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> list[ImageRecord]:
        """
        GPU-batch execution mode for CUDA and Apple Silicon (MPS).

        Images are loaded sequentially (I/O bound), then processed in batches
        of gpu_batch_size using GPU-accelerated torch ops for the expensive stages.
        Every GPU call has a CPU fallback — if torch raises for any reason the
        scorer's original CPU method is used for that batch.

        Stages per batch:
            1.  Load images (CPU, I/O)
            2.  Face detection + scene routing (CPU, MediaPipe)
            3.  Sharpness — GPU Laplacian conv2d
            4.  Exposure + white balance (CPU, trivial)
            5.  Gate check
            6.  Motion blur — GPU FFT  (pre-gate metadata, run on all images)
            7.  Release gate-failed pixel data
            8.  Saliency — GPU Sobel conv2d  (scene-type gate-passed only)
            9.  Aesthetic — GPU RGB→Gray/HSV  (gate-passed only)
            10. Composition tagging (CPU, geometric)
            11. Final score + perceptual hash + pixel release
        After all batches: dedup + re-score + rank (same as other modes).
        """
        folder = ImageLoader._normalise_path(folder_path)
        paths = [p for p in sorted(folder.iterdir()) if p.is_file() and self.image_loader.is_supported(p)]
        n = max(1, len(paths))
        gpu_batch_size = max(1, int(self.config.gpu_batch_size))
        device = self.device

        records: list[ImageRecord] = []
        processed_count = 0
        timer = _StageTimer()

        for batch_start in range(0, len(paths), gpu_batch_size):
            batch_paths = paths[batch_start: batch_start + gpu_batch_size]

            # --- 1. Load ---
            batch: list[ImageRecord] = []
            for p in batch_paths:
                try:
                    t0 = time.perf_counter()
                    batch.append(self.image_loader.load(str(p)))
                    timer.add("load", time.perf_counter() - t0)
                except Exception:
                    continue
            if not batch:
                continue

            # --- 2. Face detection + routing (CPU) ---
            # Face detection runs once here; step 4 in sequential is intentionally omitted
            # because re-running the same detector on the same image yields the same result.
            for r in batch:
                if self.config.enable_router:
                    if self.config.enable_face_detector:
                        t0 = time.perf_counter()
                        try:
                            self.face_detector.detect(r)
                        except Exception:
                            pass
                        timer.add("face_detection", time.perf_counter() - t0)
                    t0 = time.perf_counter()
                    self.router.classify(r)
                    timer.add("routing", time.perf_counter() - t0)
                elif self.config.enable_face_detector:
                    # Router disabled — still need face detection for sharpness region selection
                    t0 = time.perf_counter()
                    try:
                        self.face_detector.detect(r)
                    except Exception:
                        pass
                    timer.add("face_detection", time.perf_counter() - t0)

            # --- 3. Sharpness — GPU ---
            valid = [r for r in batch if r.image is not None]
            regions = [self.sharpness_scorer._select_region(r) for r in valid]
            t0 = time.perf_counter()
            try:
                scores = self.sharpness_scorer.score_batch_gpu(regions, device)
                for r, s in zip(valid, scores):
                    r.sharpness_score = s
            except Exception:
                for r in batch:
                    self.sharpness_scorer.score(r)
            timer.add("sharpness", time.perf_counter() - t0)

            # --- 4. Exposure + white balance (CPU) ---
            t0 = time.perf_counter()
            for r in batch:
                self.exposure_scorer.score(r)
            timer.add("exposure", time.perf_counter() - t0)
            t0 = time.perf_counter()
            for r in batch:
                self.white_balance_scorer.score(r)
            timer.add("white_balance", time.perf_counter() - t0)

            # --- 5. Gate check ---
            for r in batch:
                self._run_gate(r)

            # --- 6. Motion blur — GPU (metadata; not gate input) ---
            if self.config.enable_motion_blur:
                mb_batch = [r for r in batch if r.image is not None]
                if mb_batch:
                    t0 = time.perf_counter()
                    try:
                        blur_results = self.motion_blur_detector.detect_batch_gpu(
                            [r.image for r in mb_batch], device
                        )
                        for r, detected in zip(mb_batch, blur_results):
                            r.motion_blur_detected = detected
                    except Exception:
                        for r in mb_batch:
                            self.motion_blur_detector.detect(r)
                    timer.add("motion_blur", time.perf_counter() - t0)

            # --- 7. Release gate-failed pixel data ---
            if self.config.release_pixel_data:
                for r in batch:
                    if not r.passed_gate:
                        r.image = None

            # --- 8. Saliency — GPU (scene-type, gate-passed) ---
            passed_batch = [r for r in batch if r.passed_gate]
            if self.config.enable_saliency_detector:
                scene_batch = [r for r in passed_batch if r.scene_type == "scene" and r.image is not None]
                if scene_batch:
                    t0 = time.perf_counter()
                    try:
                        self.saliency_detector.detect_batch_gpu(scene_batch, device)
                    except Exception:
                        for r in scene_batch:
                            self.saliency_detector.detect(r)
                    timer.add("saliency", time.perf_counter() - t0)
                    if self.config.release_pixel_data:
                        for r in scene_batch:
                            r.saliency_map = None

            # Object detection (CPU, heuristic — no GPU benefit)
            if self.config.enable_object_detector:
                t0 = time.perf_counter()
                for r in passed_batch:
                    if r.scene_type == "object":
                        self.object_detector.detect(r)
                timer.add("object_detection", time.perf_counter() - t0)

            # --- 9. Aesthetic — CPU (gate-passed) ---
            # Global mean/std statistics are memory-bandwidth-bound; GPU transfer
            # overhead exceeds any compute gain for MPS/CUDA on variable-size images.
            if self.config.enable_aesthetic:
                t0 = time.perf_counter()
                for r in passed_batch:
                    if r.image is not None:
                        self.aesthetic_scorer.score(r)
                timer.add("aesthetic", time.perf_counter() - t0)

            # --- 10. Composition tagging (CPU) ---
            if self.config.enable_composition_tags:
                t0 = time.perf_counter()
                for r in passed_batch:
                    if r.image is not None:
                        try:
                            self.composition_tagger.tag(r)
                        except Exception:
                            pass
                timer.add("composition", time.perf_counter() - t0)

            # --- 11. Final score + hash + pixel release ---
            for r in passed_batch:
                self.final_scorer.compute(r)
                if self.config.enable_dedup and r.image is not None:
                    try:
                        import imagehash
                        from PIL import Image
                        t0 = time.perf_counter()
                        r.perceptual_hash = str(imagehash.phash(Image.fromarray(r.image)))
                        timer.add("perceptual_hash", time.perf_counter() - t0)
                    except Exception:
                        pass
                if self.config.release_pixel_data:
                    r.image = None
                    r.saliency_map = None

            records.extend(batch)
            processed_count += len(batch)
            if progress_callback:
                progress_callback(
                    processed_count / n,
                    f"GPU batch {batch_start // gpu_batch_size + 1}: {processed_count}/{n}",
                )

        # --- Dedup + re-score + rank (same as other modes) ---
        passed = [r for r in records if r.passed_gate]

        if self.config.enable_dedup and len(passed) > 1:
            if progress_callback:
                progress_callback(0.95, "Deduplicating...")
            t0 = time.perf_counter()
            self.duplicate_filter.filter(passed)
            timer.add("dedup", time.perf_counter() - t0)

        for r in passed:
            self.final_scorer.compute(r)

        if progress_callback:
            progress_callback(1.0, "Ranking...")

        timer.report(len(paths))
        return self.final_scorer.rank(records)

    def _run_gate(self, record: ImageRecord) -> None:
        thr = float(self.config.sharpness_gate_threshold)
        record.passed_gate = bool((record.sharpness_score or 0.0) >= thr)

    def _run_parallel(
        self,
        folder_path: str,
        progress_callback: Callable[[float, str], None] | None = None,
        num_workers: int = 4,
    ) -> list[ImageRecord]:
        """
        Parallel implementation using ProcessPoolExecutor.

        Strategy:
            1. Parallel image loading (I/O bound)
            2. Parallel pre-gate processing (CPU bound)
            3. Parallel post-gate processing (CPU bound, smaller set)
            4. Sequential deduplication (needs all records)
            5. Sequential ranking (fast, needs all records)

        Args:
            folder_path: Path to folder containing images
            progress_callback: Optional callback(fraction, message)
            num_workers: Number of worker processes

        Returns:
            List of ImageRecords sorted by final_score
        """
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from .parallel import config_to_dict, worker_load_image, worker_process_postgate, worker_process_pregate

        folder = ImageLoader._normalise_path(folder_path)
        paths = [p for p in sorted(folder.iterdir()) if p.is_file() and self.image_loader.is_supported(p)]
        n = max(1, len(paths))

        # Convert config to dict for serialization
        config_dict = config_to_dict(self.config)

        records: list[ImageRecord] = []
        processed_count = 0

        # Phase 1: Parallel image loading + pre-gate processing
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all image loading tasks
            future_to_path = {executor.submit(worker_load_image, str(p)): p for p in paths}

            for future in as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    record = future.result()
                    if record is None:
                        # Load failed, skip this image
                        processed_count += 1
                        if progress_callback:
                            progress_callback(processed_count / n, f"Loading images...")
                        continue

                    # Immediately submit for pre-gate processing
                    pregate_future = executor.submit(worker_process_pregate, record, config_dict)
                    processed_record = pregate_future.result()
                    records.append(processed_record)

                    processed_count += 1
                    if progress_callback:
                        progress_callback(
                            processed_count / n,
                            f"Processing {processed_record.filename}",
                        )

                except Exception:
                    # Skip failed images
                    processed_count += 1
                    if progress_callback:
                        progress_callback(processed_count / n, "Processing...")
                    continue

        # Phase 2: Parallel post-gate processing (only gate-passed images)
        passed = [r for r in records if r.passed_gate]

        if passed:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                future_to_record = {executor.submit(worker_process_postgate, r, config_dict): r for r in passed}

                for i, future in enumerate(as_completed(future_to_record), 1):
                    try:
                        processed_record = future.result()
                        # Update record in-place (same object reference)
                        original = future_to_record[future]
                        # Copy scores back to original record
                        original.motion_blur_detected = processed_record.motion_blur_detected
                        original.aesthetic_score = processed_record.aesthetic_score
                        original.composition_tags = processed_record.composition_tags
                        original.final_score = processed_record.final_score
                        original.perceptual_hash = processed_record.perceptual_hash
                        original.saliency_peak_region = processed_record.saliency_peak_region
                        # Pixel data already released by worker
                        original.image = None
                        original.saliency_map = None

                        if progress_callback:
                            progress_callback(
                                (n + i) / (n + len(passed)),
                                f"Post-processing {original.filename}",
                            )
                    except Exception:
                        # Post-gate processing failed, keep record with partial scores
                        continue

        # Phase 3: Sequential deduplication (needs all records at once)
        if self.config.enable_dedup and len(passed) > 1:
            if progress_callback:
                progress_callback(0.95, "Deduplicating...")
            self.duplicate_filter.filter(passed)

        # Recompute final scores after dedup flags
        for r in passed:
            self.final_scorer.compute(r)

        # Phase 4: Sequential ranking
        if progress_callback:
            progress_callback(1.0, "Ranking...")

        return self.final_scorer.rank(records)

