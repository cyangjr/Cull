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

