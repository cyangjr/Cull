"""Gate decisions: sharpness, exposure, and a usable subject face."""
from __future__ import annotations

import sys
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.config import PipelineConfig
from pipeline.parallel import config_to_dict, worker_process_pregate
from pipeline.scorer import ExposureScorer, SharpnessScorer, apply_gate, mark_subject_face
from pipeline.session import SessionManager
from pipeline.utils import ImageRecord


def _rgb(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return np.stack([arr, arr, arr], axis=-1)
    return arr


def _record(image: np.ndarray, name: str = "frame.jpg") -> ImageRecord:
    return ImageRecord(path=name, filename=name, image=_rgb(image))


def _checker(h: int, w: int, lo: int, hi: int, step: int = 2) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    tile = ((yy // step + xx // step) % 2).astype(np.uint8)
    return np.where(tile == 0, lo, hi).astype(np.uint8)


def test_exposure_levels() -> None:
    scorer = ExposureScorer()
    mid = scorer._analyse_histogram(_rgb(np.full((64, 64), 128, np.uint8)))
    dark = scorer._analyse_histogram(_rgb(np.full((64, 64), 8, np.uint8)))
    bright = scorer._analyse_histogram(_rgb(np.full((64, 64), 250, np.uint8)))
    blown = np.full((64, 64, 3), 140, np.uint8)
    blown[:, 32:] = 255
    clipped = scorer._analyse_histogram(blown)
    print(f"exposure mid={mid:.3f} dark={dark:.3f} bright={bright:.3f} clipped={clipped:.3f}")
    assert mid >= 0.8, mid
    assert dark < 0.4, dark
    assert bright < 0.4, bright
    assert clipped < 0.4, clipped


def test_subject_face_and_gate() -> None:
    cfg = PipelineConfig()
    cfg.min_subject_face_side = 64

    # A face smaller than min_subject_face_side is not the subject, so sharpness
    # stays on the full frame instead of a noisy few pixels.
    soft = np.full((400, 400), 128, np.uint8)
    tiny = _record(soft, "tiny.jpg")
    tiny.has_faces = True
    tiny.eye_region = (10, 10, 16, 16)
    tiny.image[10:26, 10:26] = np.stack([_checker(16, 16, 0, 255, step=1)] * 3, axis=-1)
    mark_subject_face(tiny, cfg)
    assert tiny.face_is_subject is False
    region = SharpnessScorer()._select_region(tiny)
    assert region.shape[0] == 400 and region.shape[1] == 400, region.shape

    # Large, sharp, well-lit face on a nearly black frame still passes.
    frame = np.full((400, 400), 8, np.uint8)
    portrait = _record(frame, "portrait.jpg")
    portrait.image[80:240, 120:320] = np.stack([_checker(160, 200, 70, 190, step=2)] * 3, axis=-1)
    portrait.has_faces = True
    portrait.eye_region = (120, 80, 200, 160)
    mark_subject_face(portrait, cfg)
    assert portrait.face_is_subject is True
    SharpnessScorer().score(portrait)
    ExposureScorer().score(portrait)
    apply_gate(portrait, cfg)
    print(
        f"portrait sharp={portrait.sharpness_score:.3f} exp={portrait.exposure_score:.3f} "
        f"pass={portrait.passed_gate} reason={portrait.gate_reason!r}"
    )
    assert portrait.passed_gate is True
    assert portrait.gate_reason == ""

    # Same face crop, but crushed dark: reject as an unusable face.
    dark_face = _record(np.full((400, 400), 128, np.uint8), "dark_face.jpg")
    dark_face.image[80:240, 120:320] = np.stack([_checker(160, 200, 0, 18, step=2)] * 3, axis=-1)
    dark_face.has_faces = True
    dark_face.eye_region = (120, 80, 200, 160)
    mark_subject_face(dark_face, cfg)
    SharpnessScorer().score(dark_face)
    ExposureScorer().score(dark_face)
    apply_gate(dark_face, cfg)
    print(
        f"dark face sharp={dark_face.sharpness_score:.3f} exp={dark_face.exposure_score:.3f} "
        f"pass={dark_face.passed_gate} reason={dark_face.gate_reason!r}"
    )
    assert dark_face.passed_gate is False
    assert dark_face.gate_reason == "face"


def test_worker_gate() -> None:
    cfg = PipelineConfig()
    cfg.enable_face_detector = False
    cfg.release_pixel_data = False
    image = np.stack([_checker(128, 128, 0, 16, step=2)] * 3, axis=-1)
    record = ImageRecord(path="dark_sharp.jpg", filename="dark_sharp.jpg", image=image)
    out = worker_process_pregate(record, config_to_dict(cfg))
    print(
        f"worker sharp={out.sharpness_score:.3f} exp={out.exposure_score:.3f} "
        f"pass={out.passed_gate} reason={out.gate_reason!r}"
    )
    assert out.passed_gate is False
    assert "exposure" in (out.gate_reason or "")


def test_pipeline_folder() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        bars = np.full((240, 320), 90, np.uint8)
        for i in range(0, 320, 40):
            bars[:, i : i + 20] = 200
        Image.fromarray(np.stack([bars, bars, bars], axis=-1)).save(folder / "sharp.jpg")
        rng = np.random.default_rng(0)
        Image.fromarray(rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)).save(folder / "noise.jpg")
        Image.fromarray(np.full((240, 320, 3), 8, np.uint8)).save(folder / "dark.jpg")
        Image.fromarray(np.full((240, 320, 3), 250, np.uint8)).save(folder / "bright.jpg")
        blur = np.zeros((240, 320, 3), np.uint8)
        blur[:] = (90, 140, 160)
        Image.fromarray(blur).save(folder / "blurry.jpg")

        session = SessionManager()
        session.pipeline.config.num_workers = 1
        session.start(str(folder))
        by_name = {r.filename: r for r in session.records}
        for name, rec in by_name.items():
            print(
                f"pipeline {name:12} sharp={rec.sharpness_score} exp={rec.exposure_score} "
                f"pass={rec.passed_gate} reason={rec.gate_reason!r} final={rec.final_score}"
            )
        assert by_name["sharp.jpg"].passed_gate is True
        assert by_name["noise.jpg"].passed_gate is False
        assert "sharpness" in by_name["noise.jpg"].gate_reason
        assert by_name["dark.jpg"].passed_gate is False
        assert "exposure" in by_name["dark.jpg"].gate_reason
        assert by_name["bright.jpg"].passed_gate is False
        assert "exposure" in by_name["bright.jpg"].gate_reason
        assert by_name["blurry.jpg"].passed_gate is False
        assert "sharpness" in by_name["blurry.jpg"].gate_reason
        assert len(session.get_kept()) == 1


def test_closed_eyes_gate() -> None:
    cfg = PipelineConfig()
    image = np.stack([_checker(200, 240, 70, 190, step=4)] * 3, axis=-1)
    record = _record(image, "blink.jpg")
    record.has_faces = True
    record.eye_region = (40, 40, 160, 100)
    record.face_is_subject = True
    record.eye_blink_score = 0.05
    SharpnessScorer().score(record)
    ExposureScorer().score(record)
    apply_gate(record, cfg)
    print(f"open eyes pass={record.passed_gate} sharp={record.sharpness_score:.3f} exp={record.exposure_score:.3f}")
    assert record.passed_gate is True

    record.eye_blink_score = 0.8
    apply_gate(record, cfg)
    print(f"closed eyes pass={record.passed_gate} reason={record.gate_reason!r}")
    assert record.passed_gate is False
    assert record.gate_reason == "eyes"


def test_open_portrait() -> None:
    """A real open-eyed portrait is a subject face and is kept."""
    url = "https://storage.googleapis.com/mediapipe-assets/business-person.png"
    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp) / "portrait.png"
        urllib.request.urlretrieve(url, dest)  # noqa: S310
        session = SessionManager()
        session.pipeline.config.num_workers = 1
        session.start(tmp)
        rec = session.records[0]
        print(
            f"portrait blink={rec.eye_blink_score} subject={rec.face_is_subject} "
            f"pass={rec.passed_gate} reason={rec.gate_reason!r} sharp={rec.sharpness_score}"
        )
        assert rec.has_faces is True
        assert rec.face_is_subject is True
        assert rec.eye_blink_score is not None and rec.eye_blink_score < 0.5
        assert rec.passed_gate is True
        assert rec.gate_reason == ""


def main() -> None:
    test_exposure_levels()
    test_subject_face_and_gate()
    test_closed_eyes_gate()
    test_worker_gate()
    test_pipeline_folder()
    test_open_portrait()
    print("ALL GATE TESTS PASSED")


if __name__ == "__main__":
    main()
