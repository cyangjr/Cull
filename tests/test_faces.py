"""Face selection and blink scoring. No MediaPipe model and no network."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np

from pipeline.detector import (
    FaceDetector,
    _annotate_record,
    _eye_region_from_bbox,
    blink_from_openness,
    eye_openness,
    select_primary_face,
)
from pipeline.utils import ImageRecord


def _skin_canvas(height: int = 90, width: int = 140) -> np.ndarray:
    return np.full((height, width, 3), 190, dtype=np.uint8)


def _draw_open_eye() -> np.ndarray:
    import cv2

    image = _skin_canvas()
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    cv2.ellipse(
        image,
        center,
        (int(width * 0.42), int(height * 0.32)),
        0,
        0,
        360,
        (255, 255, 255),
        thickness=-1,
    )
    iris_radius = max(1, int(round(0.16 * height)))
    cv2.circle(image, center, iris_radius, (30, 25, 20), thickness=-1)
    pupil_radius = max(1, int(round(iris_radius * 0.45)))
    cv2.circle(image, center, pupil_radius, (5, 5, 5), thickness=-1)
    return image


def _draw_closed_eye() -> np.ndarray:
    image = _skin_canvas()
    height, _width = image.shape[:2]
    row = height // 2
    image[row : row + 2, :, :] = (15, 15, 15)
    return image


def _bbox(origin_x: float, origin_y: float, width: float, height: float) -> SimpleNamespace:
    return SimpleNamespace(origin_x=origin_x, origin_y=origin_y, width=width, height=height)


def _detection(bbox, score: float = 0.0, keypoints=None) -> SimpleNamespace:
    return SimpleNamespace(
        bounding_box=bbox,
        categories=[SimpleNamespace(score=score)],
        keypoints=[] if keypoints is None else keypoints,
    )


def test_select_primary_face_prefers_larger_area():
    faces = [
        {"area": 10.0, "score": 0.99},
        {"area": 50.0, "score": 0.40},
        {"area": 30.0, "score": 0.80},
    ]
    assert select_primary_face(faces) == 1


def test_select_primary_face_tie_breaks_on_higher_score():
    faces = [
        {"area": 40.0, "score": 0.20},
        {"area": 40.0, "score": 0.91},
        {"area": 15.0, "score": 0.99},
    ]
    assert select_primary_face(faces) == 1


def test_select_primary_face_empty_is_none():
    assert select_primary_face([]) is None


def test_eye_openness_open_eye_is_high():
    score = eye_openness(_draw_open_eye())
    assert score > 0.45


def test_eye_openness_closed_eye_is_low():
    score = eye_openness(_draw_closed_eye())
    assert score < 0.28


def test_blink_from_openness_default_threshold():
    assert blink_from_openness(0.10) is True
    assert blink_from_openness(0.63) is False


def test_face_detector_init_accepts_blink_threshold():
    signature = inspect.signature(FaceDetector.__init__)
    params = signature.parameters
    assert params["min_detection_confidence"].default == 0.5
    assert params["blink_openness_threshold"].default == 0.28


def test_missing_image_clears_face_fields():
    record = ImageRecord(path="x", filename="x.jpg", image=None)
    record.has_faces = True
    record.face_count = 4
    record.eye_region = (1, 2, 3, 4)
    record.eyes_open_score = 0.8
    record.blink_detected = False
    FaceDetector.detect(object(), record)
    assert record.has_faces is False
    assert record.face_count == 0
    assert record.eye_region is None
    assert record.eyes_open_score is None
    assert record.blink_detected is None


def test_primary_face_not_the_first_detection_supplies_eye_region():
    image = np.full((400, 400, 3), 190, dtype=np.uint8)
    small = _bbox(10, 10, 40, 40)
    large = _bbox(100, 80, 200, 160)
    record = ImageRecord(path="a", filename="a.jpg", image=image)
    _annotate_record(
        record,
        image,
        [
            _detection(small, score=0.99),
            _detection(large, score=0.40),
        ],
        blink_openness_threshold=0.28,
    )
    assert record.face_count == 2
    assert record.has_faces is True
    assert record.eye_region == _eye_region_from_bbox(large, 400, 400)
    assert record.eye_region == (70, 48, 260, 124)


def test_one_closed_eye_counts_as_blink():
    image = np.full((200, 280, 3), 190, dtype=np.uint8)
    image[0:90, 0:140] = _draw_open_eye()
    image[0:90, 140:280] = _draw_closed_eye()
    # No keypoints: upper 45% of a 280x200 box splits into the two drawn eyes.
    record = ImageRecord(path="b", filename="b.jpg", image=image)
    _annotate_record(
        record,
        image,
        [_detection(_bbox(0, 0, 280, 200), score=0.8)],
        blink_openness_threshold=0.28,
    )
    assert record.eyes_open_score is not None
    assert record.eyes_open_score < 0.28
    assert record.blink_detected is True


def _eyes_on_canvas(closed_right: bool) -> tuple[np.ndarray, list]:
    """Two 90x140 eyes. Crop size is the full eye when face width is 400."""
    image = np.full((90, 320, 3), 190, dtype=np.uint8)
    image[:, 0:140] = _draw_open_eye()
    image[:, 180:320] = _draw_closed_eye() if closed_right else _draw_open_eye()
    height, width = image.shape[:2]
    keypoints = [
        SimpleNamespace(x=70 / width, y=45 / height),
        SimpleNamespace(x=250 / width, y=45 / height),
    ]
    return image, keypoints


def test_keypoint_score_is_the_minimum_eye():
    image, keypoints = _eyes_on_canvas(closed_right=True)
    record = ImageRecord(path="d", filename="d.jpg", image=image)
    _annotate_record(
        record,
        image,
        [_detection(_bbox(0, 0, 400, 200), score=0.9, keypoints=keypoints)],
        blink_openness_threshold=0.28,
    )
    assert record.eyes_open_score is not None
    assert record.eyes_open_score < 0.28
    assert record.blink_detected is True


def test_two_open_eyes_are_not_a_blink():
    image, keypoints = _eyes_on_canvas(closed_right=False)
    record = ImageRecord(path="e", filename="e.jpg", image=image)
    _annotate_record(
        record,
        image,
        [_detection(_bbox(0, 0, 400, 200), score=0.9, keypoints=keypoints)],
        blink_openness_threshold=0.28,
    )
    assert record.eyes_open_score is not None
    assert record.eyes_open_score > 0.45
    assert record.blink_detected is False


def test_no_detections_clears_blink_fields():
    image = np.full((32, 32, 3), 190, dtype=np.uint8)
    record = ImageRecord(path="c", filename="c.jpg", image=image)
    record.has_faces = True
    record.face_count = 2
    record.eye_region = (0, 0, 10, 10)
    record.eyes_open_score = 0.5
    record.blink_detected = False
    _annotate_record(record, image, [], blink_openness_threshold=0.28)
    assert record.has_faces is False
    assert record.face_count == 0
    assert record.eye_region is None
    assert record.eyes_open_score is None
    assert record.blink_detected is None
