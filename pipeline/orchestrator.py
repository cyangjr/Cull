from __future__ import annotations

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


class CullPipeline:
    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig.load()
        self.device = DeviceManager().get_device()

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

        Automatically chooses sequential or parallel execution based on config.num_workers.

        Args:
            folder_path: Path to folder containing images
            progress_callback: Optional callback(fraction, message) for progress updates

        Returns:
            List of ImageRecords sorted by final_score descending
        """
        from .cpu_utils import get_safe_worker_count

        # Determine execution mode
        workers = get_safe_worker_count(self.config.num_workers)

        if workers <= 1:
            # Sequential mode
            return self._run_sequential(folder_path, progress_callback)
        else:
            # Parallel mode
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

        # Process in batches to cap peak RAM.
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start : start + batch_size]
            batch: list[ImageRecord] = []
            for p in batch_paths:
                try:
                    batch.append(self.image_loader.load(str(p)))
                except Exception:
                    continue

            # Stages 2-9: pre-gate processing
            for i, r in enumerate(batch, start=start):
                if progress_callback:
                    progress_callback(i / n, f"Processing {r.filename}")

                # 2) Route
                if self.config.enable_router:
                    if self.config.enable_face_detector:
                        self.face_detector.detect(r)
                    self.router.classify(r)

                # 3) Detect subjects
                if self.config.enable_object_detector and r.scene_type == "object":
                    self.object_detector.detect(r)
                if self.config.enable_saliency_detector and r.scene_type == "scene":
                    self.saliency_detector.detect(r)
                    # Saliency map is huge; keep only the peak bbox unless you explicitly need the map.
                    if self.config.release_pixel_data:
                        r.saliency_map = None

                # 4) Detect faces (runs on all photos)
                if self.config.enable_face_detector and not r.has_faces:
                    self.face_detector.detect(r)

                # 5) Score sharpness
                self.sharpness_scorer.score(r)

                # 6) Motion blur detect (metadata; not gate input)
                if self.config.enable_motion_blur:
                    self.motion_blur_detector.detect(r)

                # 7) Exposure score
                self.exposure_scorer.score(r)

                # 8) White balance score
                self.white_balance_scorer.score(r)

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
                    self.aesthetic_scorer.score(r)

                # 11) Composition tags
                if self.config.enable_composition_tags:
                    self.composition_tagger.tag(r)

                # 13) Final score (pre-dedup)
                self.final_scorer.compute(r)

                # Precompute perceptual hash for dedup, then release pixels.
                if self.config.enable_dedup and r.image is not None:
                    try:
                        # Use DuplicateFilter's internal hash method via compute on demand (kept local here).
                        import imagehash
                        from PIL import Image

                        r.perceptual_hash = str(imagehash.phash(Image.fromarray(r.image)))
                    except Exception:
                        pass

                if self.config.release_pixel_data:
                    r.image = None
                    r.saliency_map = None

            records.extend(batch)

        passed = [r for r in records if r.passed_gate]

        # Stage 12) Deduplicate across all gate-passed records
        if self.config.enable_dedup and len(passed) > 1:
            self.duplicate_filter.filter(passed)

        # Recompute final scores for passed records after dedup flags.
        for r in passed:
            self.final_scorer.compute(r)

        # Progress wrap-up
        if progress_callback:
            progress_callback(1.0, "Ranking")

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

