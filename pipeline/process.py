"""Per-image culling stages shared by every execution mode."""

from __future__ import annotations

import time
from dataclasses import dataclass

from .config import PipelineConfig
from .detector import FaceDetector, ObjectDetector, SaliencyDetector
from .router import SceneRouter
from .scorer import (
    AestheticScorer,
    CompositionTagger,
    ExposureScorer,
    FinalScorer,
    MotionBlurDetector,
    SharpnessScorer,
    WhiteBalanceScorer,
)
from .utils import ImageRecord


@dataclass
class ProcessContext:
    config: PipelineConfig
    face_detector: FaceDetector | None
    router: SceneRouter
    object_detector: ObjectDetector
    saliency_detector: SaliencyDetector
    sharpness_scorer: SharpnessScorer
    motion_blur_detector: MotionBlurDetector
    exposure_scorer: ExposureScorer
    white_balance_scorer: WhiteBalanceScorer
    aesthetic_scorer: AestheticScorer
    composition_tagger: CompositionTagger
    final_scorer: FinalScorer


def build_context(config: PipelineConfig) -> ProcessContext:
    """Build the scorers once. Face detection is optional and stays off the hot path."""
    face_detector = None
    if config.enable_face_detector:
        face_detector = FaceDetector(
            min_detection_confidence=float(getattr(config, "min_face_detection_confidence", 0.5)),
            blink_openness_threshold=float(getattr(config, "blink_openness_threshold", 0.28)),
        )
    long_edge = int(getattr(config, "sharpness_analysis_long_edge", 512))
    return ProcessContext(
        config=config,
        face_detector=face_detector,
        router=SceneRouter(),
        object_detector=ObjectDetector(),
        saliency_detector=SaliencyDetector(),
        sharpness_scorer=SharpnessScorer(long_edge=long_edge),
        motion_blur_detector=MotionBlurDetector(),
        exposure_scorer=ExposureScorer(),
        white_balance_scorer=WhiteBalanceScorer(),
        aesthetic_scorer=AestheticScorer(),
        composition_tagger=CompositionTagger(),
        final_scorer=FinalScorer(config),
    )


def process_record(record: ImageRecord, ctx: ProcessContext, timer=None) -> None:
    """Run the canonical stage order on one image.

    Sharpness runs only after the subject and saliency crops exist, so every
    device scores the same region. A blink never fails the gate.
    """
    config = ctx.config

    def run(stage: str, fn) -> None:
        if timer is None:
            fn()
            return
        started = time.perf_counter()
        try:
            fn()
        finally:
            timer.add(stage, time.perf_counter() - started)

    face_detector = ctx.face_detector
    if face_detector is not None and config.enable_face_detector:
        def _faces() -> None:
            saved = (
                record.has_faces,
                record.face_count,
                record.eye_region,
                record.eyes_open_score,
                record.blink_detected,
            )
            try:
                face_detector.detect(record)
            except Exception:
                (
                    record.has_faces,
                    record.face_count,
                    record.eye_region,
                    record.eyes_open_score,
                    record.blink_detected,
                ) = saved

        run("face_detection", _faces)

    if config.enable_router:
        run("routing", lambda: ctx.router.classify(record))

    if config.enable_object_detector and record.scene_type == "object":
        run("object_detection", lambda: ctx.object_detector.detect(record))

    if config.enable_saliency_detector and record.scene_type == "scene" and record.image is not None:
        def _saliency() -> None:
            ctx.saliency_detector.detect(record)
            if config.release_pixel_data:
                record.saliency_map = None

        run("saliency", _saliency)

    run("sharpness", lambda: ctx.sharpness_scorer.score(record))
    run("exposure", lambda: ctx.exposure_scorer.score(record))
    run("white_balance", lambda: ctx.white_balance_scorer.score(record))

    if config.enable_motion_blur:
        run("motion_blur", lambda: ctx.motion_blur_detector.detect(record))

    def _gate() -> None:
        threshold = float(config.sharpness_gate_threshold)
        record.passed_gate = (record.sharpness_score or 0.0) >= threshold
        if not record.passed_gate and config.release_pixel_data:
            record.image = None

    run("gate", _gate)
    if not record.passed_gate:
        return

    if config.enable_aesthetic:
        run("aesthetic", lambda: ctx.aesthetic_scorer.score(record))

    if config.enable_composition_tags and record.image is not None:
        def _tags() -> None:
            try:
                ctx.composition_tagger.tag(record)
            except Exception:
                pass

        run("composition", _tags)

    run("final_score", lambda: ctx.final_scorer.compute(record))

    if config.enable_dedup and record.image is not None:
        def _hash() -> None:
            try:
                import imagehash
                from PIL import Image

                record.perceptual_hash = str(imagehash.phash(Image.fromarray(record.image)))
            except Exception:
                pass

        run("perceptual_hash", _hash)

    if config.release_pixel_data:
        def _release() -> None:
            record.image = None
            record.saliency_map = None

        run("release_pixels", _release)
