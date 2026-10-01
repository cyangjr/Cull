"""In-process, parallel, and GPU-device runs must make the same decisions."""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from pipeline.config import PipelineConfig
from pipeline.cpu_utils import get_safe_worker_count
from pipeline.orchestrator import CullPipeline
from pipeline.process import build_context, process_record
from pipeline.utils import ImageLoader

_SCORE_FIELDS = ("sharpness_score", "exposure_score", "final_score")
_SHARP_NAMES = ("sharp_a.png", "sharp_b.png")


def _config(**overrides) -> PipelineConfig:
    config = PipelineConfig()
    config.enable_face_detector = False
    config.enable_router = True
    config.enable_object_detector = True
    config.enable_saliency_detector = True
    config.enable_motion_blur = True
    config.enable_aesthetic = True
    config.enable_composition_tags = True
    config.enable_dedup = True
    config.release_pixel_data = True
    config.device = "cpu"
    config.sharpness_gate_threshold = 0.3
    config.batch_size = 3
    config.gpu_batch_size = 1
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def _checker(height: int, width: int, cell: int, low: int, high: int) -> np.ndarray:
    yy, xx = np.indices((height, width))
    board = ((xx // cell) + (yy // cell)) % 2
    gray = np.where(board == 1, high, low).astype(np.uint8)
    return np.dstack([gray, gray, gray])


def _write_images(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 256, size=(700, 900, 3), dtype=np.uint8)
    blur = cv2.GaussianBlur(noise, (0, 0), sigmaX=25)

    bars = np.zeros((480, 640, 3), dtype=np.uint8)
    bars[::8] = 255

    # Moderate contrast keeps the full-frame Laplacian from saturating, so the
    # saliency crop (about 20% of the frame) scores clearly sharper.
    height, width = 640, 960
    scene_rng = np.random.default_rng(1)
    background = np.clip(scene_rng.normal(128, 6, size=(height, width, 1)), 0, 255).astype(np.uint8)
    background = np.repeat(background, 3, axis=2)
    background = cv2.GaussianBlur(background, (0, 0), sigmaX=12)
    patch_h, patch_w = 320, 384
    scene = background.copy()
    scene[30 : 30 + patch_h, 40 : 40 + patch_w] = _checker(patch_h, patch_w, 48, 64, 192)

    from PIL import Image

    Image.fromarray(blur).save(folder / "blur.png")
    Image.fromarray(bars).save(folder / "sharp_a.png")
    Image.fromarray(bars).save(folder / "sharp_b.png")
    Image.fromarray(scene).save(folder / "scene.png")
    (folder / "bad.png").write_text("not an image", encoding="utf-8")
    (folder / "notes.txt").write_text("ignore me", encoding="utf-8")

    stamp = 1_700_000_000
    for path in folder.iterdir():
        os.utime(path, (stamp, stamp))


def _run(folder: Path, num_workers: int, **overrides) -> list:
    config = _config(num_workers=num_workers, **overrides)
    return CullPipeline(config).run(str(folder))


def _by_name(records) -> dict:
    return {record.filename: record for record in records}


def _close(left, right, label: str) -> None:
    if left is None or right is None:
        assert left is None and right is None, label
        return
    assert abs(float(left) - float(right)) <= 1e-6, f"{label}: {left} vs {right}"


def _assert_one_kept_duplicate(records) -> None:
    found = _by_name(records)
    flags = [found[name].is_duplicate for name in _SHARP_NAMES]
    assert flags.count(True) == 1
    assert flags.count(False) == 1
    assert found["sharp_a.png"].passed_gate is True
    assert found["sharp_b.png"].passed_gate is True


def _assert_parity(left, right) -> None:
    by_left = _by_name(left)
    by_right = _by_name(right)
    assert set(by_left) == set(by_right)
    for name in by_left:
        a = by_left[name]
        b = by_right[name]
        for field in _SCORE_FIELDS:
            _close(getattr(a, field), getattr(b, field), f"{name}.{field}")
        assert a.passed_gate == b.passed_gate, name
        assert a.is_duplicate == b.is_duplicate, name
        assert a.motion_blur_detected == b.motion_blur_detected, name
    _assert_one_kept_duplicate(left)
    _assert_one_kept_duplicate(right)


def _assert_session_shape(records) -> None:
    found = _by_name(records)
    assert "bad.png" not in found
    assert "notes.txt" not in found
    assert set(found) == {"blur.png", "scene.png", "sharp_a.png", "sharp_b.png"}
    blur = found["blur.png"]
    assert blur.passed_gate is False
    assert (blur.sharpness_score or 0.0) < 0.3
    _assert_one_kept_duplicate(records)


def test_in_process_saliency_before_sharpness_and_gate(tmp_path, monkeypatch) -> None:
    folder = tmp_path / "session"
    _write_images(folder)

    def _forbid_auto(*_args, **_kwargs):
        raise AssertionError("num_workers=0 must not call get_safe_worker_count")

    monkeypatch.setattr("pipeline.cpu_utils.get_safe_worker_count", _forbid_auto)
    sequenced = _run(folder, num_workers=0)
    monkeypatch.setattr("pipeline.cpu_utils.get_safe_worker_count", get_safe_worker_count)

    _assert_session_shape(sequenced)
    saliency_off = _run(folder, num_workers=0, enable_saliency_detector=False)
    on_score = _by_name(sequenced)["scene.png"].sharpness_score
    off_score = _by_name(saliency_off)["scene.png"].sharpness_score
    assert on_score is not None and off_score is not None
    assert on_score > off_score + 0.15, f"saliency-on {on_score} saliency-off {off_score}"

    record = ImageLoader().load(str(folder / "sharp_a.png"))
    record.blink_detected = True
    process_record(record, build_context(_config(num_workers=0)))
    assert record.blink_detected is True
    assert record.passed_gate is True

    direct = ImageLoader().load(str(folder / "scene.png"))
    process_record(direct, build_context(_config(num_workers=0)))
    pipeline_scene = _by_name(sequenced)["scene.png"]
    _close(direct.sharpness_score, pipeline_scene.sharpness_score, "direct.sharpness_score")
    _close(direct.exposure_score, pipeline_scene.exposure_score, "direct.exposure_score")
    assert direct.passed_gate == pipeline_scene.passed_gate
    assert direct.motion_blur_detected == pipeline_scene.motion_blur_detected


def test_parallel_matches_in_process(tmp_path) -> None:
    folder = tmp_path / "session"
    _write_images(folder)
    sequenced = _run(folder, num_workers=0)
    _assert_session_shape(sequenced)

    on_score = _by_name(sequenced)["scene.png"].sharpness_score
    off_score = _by_name(_run(folder, num_workers=0, enable_saliency_detector=False))["scene.png"].sharpness_score
    assert on_score is not None and off_score is not None
    assert on_score > off_score + 0.15, f"saliency-on {on_score} saliency-off {off_score}"

    try:
        parallel = _run(folder, num_workers=2)
    except Exception as exc:
        pytest.fail(f"Parallel pool failed to start: {type(exc).__name__}: {exc}")
    _assert_parity(sequenced, parallel)


def test_cuda_and_mps_call_process_record(tmp_path, monkeypatch) -> None:
    folder = tmp_path / "session"
    _write_images(folder)
    sequenced = _run(folder, num_workers=0)

    import pipeline.orchestrator as orchestrator

    calls: list[str] = []
    real = orchestrator.process_record

    def _spy(record, ctx, timer=None):
        calls.append(record.filename)
        return real(record, ctx, timer)

    monkeypatch.setattr(orchestrator, "process_record", _spy)

    for device in ("cuda", "mps"):
        calls.clear()
        config = _config(num_workers=2)
        pipeline = CullPipeline(config)
        pipeline.device = device
        device_records = pipeline.run(str(folder))
        assert calls, f"{device} did not call process_record"
        assert sorted(calls) == ["blur.png", "scene.png", "sharp_a.png", "sharp_b.png"]
        _assert_parity(sequenced, device_records)
