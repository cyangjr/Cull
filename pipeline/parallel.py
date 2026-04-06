"""Parallel processing workers for CPU optimization."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import PipelineConfig
    from .utils import ImageRecord


def worker_load_image(path: str) -> ImageRecord | None:
    """
    Worker function for parallel image loading.

    Args:
        path: Absolute path to image file

    Returns:
        ImageRecord with loaded image, or None if load failed

    Note:
        - Runs in separate process (no shared state)
        - Must import modules inside worker
        - Returns serializable ImageRecord
    """
    try:
        from .utils import ImageLoader

        loader = ImageLoader()
        return loader.load(path)
    except Exception:
        # Silent failure - orchestrator will skip this image
        return None


def worker_process_pregate(record: ImageRecord, config_dict: dict) -> ImageRecord:
    """
    Worker function for parallel pre-gate processing.

    Runs all scoring stages up to and including gate check.

    Args:
        record: ImageRecord with loaded image
        config_dict: Serialized PipelineConfig (dict, not object)

    Returns:
        ImageRecord with scores and gate decision

    Stages:
        1. Face detection (if enabled)
        2. Sharpness scoring
        3. Exposure scoring
        4. White balance scoring
        5. Gate check
        6. Release pixel data if gate failed

    Note:
        - Runs in separate process
        - Must recreate scorer instances
        - Releases pixel data for gate-failed images
    """
    try:
        from .config import PipelineConfig
        from .detector import FaceDetector
        from .scorer import ExposureScorer, SharpnessScorer, WhiteBalanceScorer

        # Reconstruct config from dict
        config = PipelineConfig()
        for k, v in config_dict.items():
            if hasattr(config, k):
                setattr(config, k, v)

        # Initialize scorers (local to this worker)
        sharpness_scorer = SharpnessScorer()
        exposure_scorer = ExposureScorer()
        white_balance_scorer = WhiteBalanceScorer()

        # Optional: Face detection for sharpness region selection
        face_detector = None
        if config.enable_face_detector:
            try:
                face_detector = FaceDetector(config.min_face_detection_confidence)
            except Exception:
                # MediaPipe may not be available, continue without faces
                pass

        # Stage 1: Face detection (for sharpness region)
        if face_detector and record.image is not None:
            try:
                face_detector.detect(record)
            except Exception:
                pass

        # Stage 2: Sharpness scoring
        sharpness_scorer.score(record)

        # Stage 3: Exposure scoring
        exposure_scorer.score(record)

        # Stage 4: White balance scoring
        white_balance_scorer.score(record)

        # Stage 5: Gate check (sharpness only)
        threshold = float(config.sharpness_gate_threshold)
        record.passed_gate = bool((record.sharpness_score or 0.0) >= threshold)

        # Stage 6: Release pixel data if gate failed
        if config.release_pixel_data and not record.passed_gate:
            record.image = None
            record.gray_cache = None  # Future: when gray caching is added

        return record

    except Exception:
        # If processing fails, mark as gate-failed and return
        record.passed_gate = False
        record.image = None
        return record


def worker_process_postgate(record: ImageRecord, config_dict: dict) -> ImageRecord:
    """
    Worker function for parallel post-gate processing.

    Runs expensive operations only on gate-passed images.

    Args:
        record: ImageRecord that passed gate
        config_dict: Serialized PipelineConfig

    Returns:
        ImageRecord with all scores computed

    Stages:
        1. Saliency detection (if scene type)
        2. Motion blur detection (expensive FFT)
        3. Aesthetic scoring
        4. Composition tagging
        5. Final score computation
        6. Perceptual hash for deduplication
        7. Release pixel data

    Note:
        - Only called on gate-passed images (~30% of total)
        - More expensive operations justified by smaller set
    """
    try:
        from .config import PipelineConfig
        from .detector import SaliencyDetector
        from .scorer import (
            AestheticScorer,
            CompositionTagger,
            FinalScorer,
            MotionBlurDetector,
        )

        # Reconstruct config
        config = PipelineConfig()
        for k, v in config_dict.items():
            if hasattr(config, k):
                setattr(config, k, v)

        # Initialize scorers
        motion_blur_detector = MotionBlurDetector()
        aesthetic_scorer = AestheticScorer()
        composition_tagger = CompositionTagger()
        final_scorer = FinalScorer(config)
        saliency_detector = SaliencyDetector()

        # Stage 1: Saliency detection (scene images only)
        if config.enable_saliency_detector and record.scene_type == "scene" and record.image is not None:
            try:
                saliency_detector.detect(record)
                # Release saliency map immediately (keep only bbox)
                if config.release_pixel_data:
                    record.saliency_map = None
            except Exception:
                pass

        # Stage 2: Motion blur detection
        if config.enable_motion_blur and record.image is not None:
            try:
                motion_blur_detector.detect(record)
            except Exception:
                pass

        # Stage 3: Aesthetic scoring
        if config.enable_aesthetic and record.image is not None:
            try:
                aesthetic_scorer.score(record)
            except Exception:
                pass

        # Stage 4: Composition tagging
        if config.enable_composition_tags and record.image is not None:
            try:
                composition_tagger.tag(record)
            except Exception:
                pass

        # Stage 5: Final score computation
        final_scorer.compute(record)

        # Stage 6: Perceptual hash for deduplication
        if config.enable_dedup and record.image is not None:
            try:
                import imagehash
                from PIL import Image

                record.perceptual_hash = str(imagehash.phash(Image.fromarray(record.image)))
            except Exception:
                pass

        # Stage 7: Release pixel data
        if config.release_pixel_data:
            record.image = None
            record.saliency_map = None
            record.gray_cache = None  # Future

        return record

    except Exception:
        # If post-gate processing fails, return record as-is
        # Final score may be 0 but record is still included
        return record


def config_to_dict(config: PipelineConfig) -> dict:
    """
    Convert PipelineConfig to serializable dict for multiprocessing.

    Args:
        config: PipelineConfig instance

    Returns:
        Dictionary with all config fields
    """
    from dataclasses import asdict

    return asdict(config)
