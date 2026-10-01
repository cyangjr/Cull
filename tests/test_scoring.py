"""Technical-quality scoring: style is not a penalty."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from pipeline.config import PipelineConfig
from pipeline.scorer import (
    AestheticScorer,
    CompositionTagger,
    DuplicateFilter,
    ExposureScorer,
    FinalScorer,
    MotionBlurDetector,
    SharpnessScorer,
    WhiteBalanceScorer,
)
from pipeline.utils import ImageRecord

_SAME_HASH = "0000000000000000"
_FAR_HASH = "ffffffffffffffff"


def _record(
    name: str,
    final_score: float,
    perceptual_hash: str,
    *,
    second: int | None = 0,
    blink_detected: bool | None = None,
    eyes_open_score: float | None = None,
) -> ImageRecord:
    record = ImageRecord(path=f"/{name}.jpg", filename=f"{name}.jpg")
    record.final_score = final_score
    record.perceptual_hash = perceptual_hash
    record.blink_detected = blink_detected
    record.eyes_open_score = eyes_open_score
    if second is None:
        record.exif = {}
    else:
        record.exif = {"timestamp": f"2024-05-01T10:00:{second:02d}"}
    return record


def _duplicate_filter() -> DuplicateFilter:
    config = PipelineConfig()
    config.hash_threshold = 8
    config.timestamp_window_s = 2.0
    return DuplicateFilter(config)


def _checker(width: int, height: int, period: int) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    block = ((xx // period) + (yy // period)) % 2
    image = np.zeros((height, width, 3), np.uint8)
    image[block == 1] = 255
    return image


def _edge_grid(width: int, height: int, step: int, thick: int) -> np.ndarray:
    image = np.full((height, width, 3), 30, np.uint8)
    for offset in range(thick):
        rows = np.arange(0, height, step) + offset
        rows = rows[rows < height]
        cols = np.arange(0, width, step) + offset
        cols = cols[cols < width]
        image[rows, :, :] = 250
        image[:, cols, :] = 250
    return image


def _score_image(scorer, image: np.ndarray, name: str):
    record = ImageRecord(path=f"/{name}.jpg", filename=f"{name}.jpg", image=image)
    scorer.score(record)
    return record


def test_sharpness_is_stable_across_resolution_and_rejects_blur_and_noise() -> None:
    scorer = SharpnessScorer(long_edge=512)
    large = _checker(2200, 1466, period=8)
    small_h = int(round(large.shape[0] * (600 / large.shape[1])))
    small = cv2.resize(large, (600, small_h), interpolation=cv2.INTER_AREA)

    large_score = _score_image(scorer, large, "large").sharpness_score
    small_score = _score_image(scorer, small, "small").sharpness_score
    assert large_score == pytest.approx(small_score, abs=0.05)

    blurred = cv2.GaussianBlur(large, (0, 0), 6)
    assert _score_image(scorer, blurred, "blur").sharpness_score < 0.25

    noise = np.random.default_rng(0).integers(0, 256, size=(800, 3600, 3), dtype=np.uint8)
    assert _score_image(scorer, noise, "noise").sharpness_score < 0.25

    grid = np.zeros((640, 960, 3), np.uint8)
    grid[::8, :, :] = 255
    grid[:, ::8, :] = 255
    assert _score_image(scorer, grid, "grid").sharpness_score > 0.6


def test_exposure_penalizes_clipping_not_a_low_key_mean() -> None:
    scorer = ExposureScorer()
    low_key = np.full((120, 160, 3), 40, np.uint8)
    assert _score_image(scorer, low_key, "low").exposure_score > 0.95

    half_blown = np.full((100, 100, 3), 128, np.uint8)
    half_blown[:, :50, :] = 255
    assert _score_image(scorer, half_blown, "blown").exposure_score < 0.5


def test_white_balance_keeps_golden_hour_and_rejects_a_severe_cast() -> None:
    scorer = WhiteBalanceScorer()
    orange = np.full((32, 32, 3), (220, 140, 80), np.uint8)
    severe = np.full((32, 32, 3), (240, 30, 20), np.uint8)
    assert _score_image(scorer, orange, "orange").white_balance_score > 0.95
    assert _score_image(scorer, severe, "severe").white_balance_score < 0.5


def test_aesthetic_uses_tonal_separation_not_saturation() -> None:
    scorer = AestheticScorer()
    flat = np.full((240, 320, 3), 128, np.uint8)
    assert _score_image(scorer, flat, "flat").aesthetic_score < 1.0

    split = np.zeros((240, 320, 3), np.uint8)
    split[:, :160, :] = 255
    assert _score_image(scorer, split, "split").aesthetic_score > 8.0

    height, width = 180, 240
    yy, xx = np.mgrid[0:height, 0:width]
    color = np.stack(
        [
            (30 + (xx * 3) % 200).astype(np.uint8),
            (15 + (yy * 2) % 180).astype(np.uint8),
            (5 + ((xx + yy) % 150)).astype(np.uint8),
        ],
        axis=-1,
    )
    gray = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)
    gray_rgb = np.stack([gray, gray, gray], axis=-1)
    color_score = _score_image(scorer, color, "color").aesthetic_score
    gray_score = _score_image(scorer, gray_rgb, "gray").aesthetic_score
    assert gray_score == pytest.approx(color_score, abs=0.05)


def test_motion_blur_flags_directional_smear_not_a_horizon() -> None:
    detector = MotionBlurDetector()
    scene = _edge_grid(900, 600, step=20, thick=2)
    clean = ImageRecord(path="/scene.jpg", filename="scene.jpg", image=scene)
    detector.detect(clean)
    assert clean.motion_blur_detected is False

    smeared = cv2.blur(scene, (51, 1))
    blurred = ImageRecord(path="/smear.jpg", filename="smear.jpg", image=smeared)
    detector.detect(blurred)
    assert blurred.motion_blur_detected is True

    horizon = np.zeros((900, 1400, 3), np.uint8)
    horizon[:450] = (180, 200, 230)
    horizon[450:] = (40, 70, 30)
    skyline = ImageRecord(path="/horizon.jpg", filename="horizon.jpg", image=horizon)
    detector.detect(skyline)
    assert skyline.motion_blur_detected is False

    # Sharp blinds are directional and crisp. They are not a smear.
    blinds = np.zeros((480, 640, 3), np.uint8)
    blinds[::8] = 255
    lines = ImageRecord(path="/blinds.jpg", filename="blinds.jpg", image=blinds)
    detector.detect(lines)
    assert lines.motion_blur_detected is False


def test_final_score_uses_eyes_and_a_neutral_composition_baseline() -> None:
    config = PipelineConfig()
    config.final_score_weights = {
        "sharpness": 0.55,
        "exposure": 0.20,
        "subject": 0.15,
        "composition": 0.10,
    }
    config.motion_blur_score_penalty = 0.75
    scorer = FinalScorer(config)

    def scored(**overrides: object) -> float:
        record = ImageRecord(path="/a.jpg", filename="a.jpg")
        record.sharpness_score = 0.8
        record.exposure_score = 1.0
        record.composition_score = 0.6
        record.has_faces = True
        record.eyes_open_score = 1.0
        for key, value in overrides.items():
            setattr(record, key, value)
        scorer.compute(record)
        assert record.final_score is not None
        return record.final_score

    closed = scored(eyes_open_score=0.0, has_faces=True)
    open_eyes = scored(eyes_open_score=1.0, has_faces=True)
    no_face = scored(eyes_open_score=0.0, has_faces=False)
    assert closed < open_eyes
    assert no_face == pytest.approx(open_eyes)

    # A blink is already in the subject term. It is not a second multiplier.
    assert scored(eyes_open_score=0.0, blink_detected=True) == pytest.approx(closed)

    base = scored()
    assert scored(motion_blur_detected=True) == pytest.approx(base * 0.75)

    neutral = ImageRecord(path="/b.jpg", filename="b.jpg")
    neutral.sharpness_score = 1.0
    neutral.exposure_score = 1.0
    neutral.has_faces = False
    neutral.composition_score = None
    scorer.compute(neutral)
    expected = 0.55 * 1.0 + 0.20 * 1.0 + 0.15 * 1.0 + 0.10 * 0.6
    assert neutral.final_score == pytest.approx(expected)
    assert neutral.final_score > 0.90


def test_duplicate_burst_keeps_the_highest_score_once() -> None:
    filt = _duplicate_filter()
    low = _record("low", 0.5, _SAME_HASH, second=0)
    hero = _record("hero", 0.9, _SAME_HASH, second=1)
    mid = _record("mid", 0.6, _SAME_HASH, second=2)
    far = _record("far", 0.99, _FAR_HASH, second=1)
    filt.filter([low, hero, mid, far])

    kept = [record for record in (low, hero, mid) if not record.is_duplicate]
    assert kept == [hero]
    assert low.is_duplicate is True
    assert mid.is_duplicate is True
    assert hero.duplicate_group
    assert low.duplicate_group == hero.duplicate_group
    assert mid.duplicate_group == hero.duplicate_group
    assert far.is_duplicate is False
    assert far.duplicate_group is None


def test_duplicate_hero_prefers_open_eyes_over_a_sharper_blink() -> None:
    filt = _duplicate_filter()
    blink = _record(
        "blink",
        0.95,
        _SAME_HASH,
        second=0,
        blink_detected=True,
        eyes_open_score=0.1,
    )
    open_eyes = _record(
        "open",
        0.70,
        _SAME_HASH,
        second=1,
        blink_detected=False,
        eyes_open_score=0.9,
    )
    other = _record(
        "other",
        0.50,
        _SAME_HASH,
        second=2,
        blink_detected=False,
        eyes_open_score=0.9,
    )
    filt.filter([blink, open_eyes, other])

    kept = [record for record in (blink, open_eyes, other) if not record.is_duplicate]
    assert kept == [open_eyes]
    assert blink.is_duplicate is True
    assert other.is_duplicate is True


def test_equal_scores_keep_the_earliest_filename() -> None:
    filt = _duplicate_filter()
    later = _record("sharp_b", 0.80, _SAME_HASH, second=0)
    earlier = _record("sharp_a", 0.80, _SAME_HASH, second=1)
    filt.filter([later, earlier])
    assert earlier.is_duplicate is False
    assert later.is_duplicate is True


def test_untimestamped_frames_are_not_one_duplicate_group() -> None:
    filt = _duplicate_filter()
    first = _record("first", 0.9, _SAME_HASH, second=None)
    second = _record("second", 0.4, _SAME_HASH, second=None)
    filt.filter([first, second])
    assert first.is_duplicate is False
    assert second.is_duplicate is False
    assert first.duplicate_group is None
    assert second.duplicate_group is None


def test_composition_score_starts_at_the_neutral_baseline() -> None:
    image = np.full((200, 300, 3), 128, np.uint8)
    record = ImageRecord(path="/c.jpg", filename="c.jpg", image=image)
    CompositionTagger().tag(record)
    assert record.composition_score is not None
    assert record.composition_score >= 0.6

    empty = ImageRecord(path="/empty.jpg", filename="empty.jpg")
    empty.composition_score = 0.42
    CompositionTagger().tag(empty)
    assert empty.composition_score == 0.42


def test_config_defaults_and_legacy_weight_keys(tmp_path) -> None:
    config = PipelineConfig()
    assert config.final_score_weights == {
        "sharpness": 0.55,
        "exposure": 0.20,
        "subject": 0.15,
        "composition": 0.10,
    }
    assert config.motion_blur_score_penalty == 0.75
    assert config.timestamp_window_s == 8.0
    assert config.sharpness_analysis_long_edge == 512
    assert config.blink_openness_threshold == 0.28

    legacy = tmp_path / "legacy.yaml"
    legacy.write_text(
        "\n".join(
            [
                "final_score_weights:",
                "  sharpness: 0.4",
                "  exposure: 0.15",
                "  white_balance: 0.15",
                "  aesthetic: 0.3",
                "sharpness_analysis_long_edge: 384",
                "blink_openness_threshold: 0.28",
            ]
        ),
        encoding="utf-8",
    )
    loaded = PipelineConfig.load(str(legacy))
    assert loaded.final_score_weights["white_balance"] == pytest.approx(0.15)
    assert loaded.final_score_weights["aesthetic"] == pytest.approx(0.3)
    assert loaded.sharpness_analysis_long_edge == 384
    assert loaded.blink_openness_threshold == pytest.approx(0.28)
