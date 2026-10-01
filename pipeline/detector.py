from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path

import numpy as np

from .utils import ImageRecord


def select_primary_face(faces: list[dict]) -> int | None:
    """Index of the largest face. Equal area keeps the higher detection score."""
    if not faces:
        return None
    best_idx = 0
    best_area = float(faces[0]["area"])
    best_score = float(faces[0]["score"])
    for idx, face in enumerate(faces[1:], start=1):
        area = float(face["area"])
        score = float(face["score"])
        if area > best_area or (area == best_area and score > best_score):
            best_idx = idx
            best_area = area
            best_score = score
    return best_idx


def eye_openness(roi: np.ndarray) -> float:
    """
    Openness of an RGB eye crop in 0..1 (1 = open).

    Downscales to 64x40, measures the tallest dark band in the center
    strip, and maps that band's height into the calibrated 0..1 score.
    """
    import cv2

    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY).astype(np.float32)
    small = cv2.resize(gray, (64, 40), interpolation=cv2.INTER_AREA)
    row = small[:, 24:40].min(axis=1)
    lo = float(row.min())
    hi = float(row.max())
    if hi - lo <= 1e-6:
        return 0.0
    row = (row - lo) / (hi - lo)
    darkness = 1.0 - row
    dark = darkness >= 0.45
    run = 0
    longest = 0
    for is_dark in dark:
        if is_dark:
            run += 1
            if run > longest:
                longest = run
        else:
            run = 0
    height_frac = longest / len(row)
    score = (height_frac - 0.08) / 0.35
    return float(np.clip(score, 0.0, 1.0))


def blink_from_openness(openness: float, threshold: float = 0.28) -> bool:
    """True when the eye-openness score is below the blink threshold."""
    return bool(openness < threshold)


def _detection_score(det) -> float:
    categories = getattr(det, "categories", None) or []
    if not categories:
        return 0.0
    score = getattr(categories[0], "score", 0.0)
    if score is None:
        return 0.0
    return float(score)


def _keypoint_norm_xy(keypoint) -> tuple[float, float] | None:
    if keypoint is None:
        return None
    if isinstance(keypoint, dict):
        if "x" not in keypoint or "y" not in keypoint:
            return None
        return float(keypoint["x"]), float(keypoint["y"])
    x = getattr(keypoint, "x", None)
    y = getattr(keypoint, "y", None)
    if x is None or y is None:
        return None
    return float(x), float(y)


def _crop_if_large_enough(image: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> np.ndarray | None:
    h, w = image.shape[:2]
    x0 = max(0, min(w, x0))
    y0 = max(0, min(h, y0))
    x1 = max(0, min(w, x1))
    y1 = max(0, min(h, y1))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    return np.ascontiguousarray(image[y0:y1, x0:x1])


def _centered_square_crop(image: np.ndarray, cx: float, cy: float, size: float) -> np.ndarray | None:
    side = int(round(size))
    if side < 8:
        return None
    x0 = int(round(cx)) - side // 2
    y0 = int(round(cy)) - side // 2
    return _crop_if_large_enough(image, x0, y0, x0 + side, y0 + side)


def _eye_crops_from_keypoints(image: np.ndarray, det, img_w: int, img_h: int) -> list[np.ndarray]:
    keypoints = getattr(det, "keypoints", None) or []
    if len(keypoints) < 1:
        return []
    bb = det.bounding_box
    crop_size = 0.35 * float(bb.width)
    crops: list[np.ndarray] = []
    for idx in (0, 1):
        if idx >= len(keypoints):
            break
        xy = _keypoint_norm_xy(keypoints[idx])
        if xy is None:
            continue
        nx, ny = xy
        crop = _centered_square_crop(image, nx * img_w, ny * img_h, crop_size)
        if crop is not None:
            crops.append(crop)
    return crops


def _fallback_eye_crops(image: np.ndarray, det) -> list[np.ndarray]:
    """Left and right halves of the upper 45% of the face box."""
    bb = det.bounding_box
    h, w = image.shape[:2]
    x0 = int(np.floor(float(bb.origin_x)))
    y0 = int(np.floor(float(bb.origin_y)))
    x1 = int(np.ceil(float(bb.origin_x) + float(bb.width)))
    y1 = int(np.ceil(float(bb.origin_y) + float(bb.height) * 0.45))
    x0 = max(0, min(w, x0))
    y0 = max(0, min(h, y0))
    x1 = max(0, min(w, x1))
    y1 = max(0, min(h, y1))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return []
    mid = x0 + (x1 - x0) // 2
    crops: list[np.ndarray] = []
    for a, b in ((x0, mid), (mid, x1)):
        crop = _crop_if_large_enough(image, a, y0, b, y1)
        if crop is not None:
            crops.append(crop)
    return crops


def _eye_region_from_bbox(bb, img_w: int, img_h: int) -> tuple[int, int, int, int]:
    """Upper-face crop: pad the box, then keep the top ~65%."""
    x0, y0 = float(bb.origin_x), float(bb.origin_y)
    bw, bh = float(bb.width), float(bb.height)
    x1, y1 = x0 + bw, y0 + bh

    pad_x = 0.15 * (x1 - x0)
    pad_y = 0.20 * (y1 - y0)
    x0 = max(0, int(x0 - pad_x))
    y0 = max(0, int(y0 - pad_y))
    x1 = min(img_w, int(x1 + pad_x))
    y1 = min(img_h, int(y0 + (y1 - y0) * 0.65))

    rw = max(1, x1 - x0)
    rh = max(1, y1 - y0)
    return (x0, y0, rw, rh)


def _clear_face_fields(record: ImageRecord) -> None:
    record.has_faces = False
    record.face_count = 0
    record.eye_region = None
    record.eyes_open_score = None
    record.blink_detected = None


def _openness_from_detection(image: np.ndarray, det) -> float | None:
    h, w = image.shape[:2]
    crops = _eye_crops_from_keypoints(image, det, w, h)
    if not crops:
        crops = _fallback_eye_crops(image, det)
    if not crops:
        return None
    return min(eye_openness(crop) for crop in crops)


def _annotate_record(
    record: ImageRecord,
    image: np.ndarray | None,
    detections: list,
    blink_openness_threshold: float,
) -> None:
    """Write face, eye-region, and blink fields from detector results."""
    dets = list(detections or [])
    if image is None or not dets:
        _clear_face_fields(record)
        return

    h, w = image.shape[:2]
    faces = [
        {
            "area": float(det.bounding_box.width) * float(det.bounding_box.height),
            "score": _detection_score(det),
        }
        for det in dets
    ]
    primary_idx = select_primary_face(faces)
    if primary_idx is None:
        _clear_face_fields(record)
        return

    primary = dets[primary_idx]
    record.face_count = len(dets)
    record.has_faces = record.face_count > 0
    record.eye_region = _eye_region_from_bbox(primary.bounding_box, w, h)

    openness = _openness_from_detection(image, primary)
    if openness is None:
        record.eyes_open_score = None
        record.blink_detected = None
        return
    record.eyes_open_score = float(openness)
    record.blink_detected = blink_from_openness(openness, blink_openness_threshold)


class ObjectDetector:
    """
    Milestone C fallback: without YOLO, approximate subject bbox from face bbox if present.
    """
    def detect(self, record: ImageRecord) -> None:
        if record.subject_bbox is not None:
            return
        if record.eye_region is None:
            return
        x, y, w, h = record.eye_region
        # Expand to a coarse "subject" box around the face region.
        pad_x = int(w * 0.8)
        pad_y = int(h * 0.8)
        x0 = max(0, x - pad_x)
        y0 = max(0, y - pad_y)
        x1 = x + w + pad_x
        y1 = y + h + pad_y
        record.subject_bbox = (x0, y0, max(1, x1 - x0), max(1, y1 - y0))


class SaliencyDetector:
    """
    Milestone C fallback: simple spectral-residual-like saliency proxy using gradient magnitude.
    Produces a heatmap and a peak region bbox.
    """
    def detect(self, record: ImageRecord) -> None:
        if record.image is None:
            record.saliency_map = None
            record.saliency_peak_region = None
            return

        import cv2
        import numpy as np

        gray = cv2.cvtColor(record.image, cv2.COLOR_RGB2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        mag = cv2.GaussianBlur(mag, (0, 0), sigmaX=3.0)
        mag_norm = mag / (float(mag.max()) + 1e-9)
        record.saliency_map = mag_norm.astype(np.float32)
        record.saliency_peak_region = self.get_peak_region(record.saliency_map)

    def get_peak_region(self, saliency_map):
        import numpy as np

        if saliency_map is None:
            return None
        h, w = saliency_map.shape[:2]
        # Take top 5% saliency pixels and compute bbox.
        thr = float(np.quantile(saliency_map, 0.95))
        ys, xs = np.where(saliency_map >= thr)
        if len(xs) == 0:
            return None
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        # Clamp + ensure non-zero
        x0 = max(0, min(w - 1, x0))
        y0 = max(0, min(h - 1, y0))
        x1 = max(x0 + 1, min(w, x1 + 1))
        y1 = max(y0 + 1, min(h, y1 + 1))
        return (x0, y0, x1 - x0, y1 - y0)

    def detect_batch_gpu(self, records: list, device: str) -> None:
        """Same maps as detect(). `device` is ignored so the peak box cannot drift."""
        del device
        for record in records:
            self.detect(record)


class FaceDetector:
    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        blink_openness_threshold: float = 0.28,
    ) -> None:
        self.blink_openness_threshold = float(blink_openness_threshold)
        try:
            import mediapipe as mp  # type: ignore
            from mediapipe.tasks import python  # type: ignore
            from mediapipe.tasks.python import vision  # type: ignore

            self._mp = mp
            self._python_tasks = python
            self._vision_tasks = vision

            self._model_path = self._ensure_model()
            base_options = python.BaseOptions(model_asset_path=str(self._model_path))
            options = vision.FaceDetectorOptions(
                base_options=base_options,
                min_detection_confidence=float(min_detection_confidence),
            )
            self._detector = vision.FaceDetector.create_from_options(options)
        except Exception as e:
            raise RuntimeError(
                "FaceDetector failed to import/initialize MediaPipe.\n"
                f"- python: {sys.executable}\n"
                f"- error: {type(e).__name__}: {e}\n\n"
                "Fix:\n"
                "- Ensure you installed into the same environment that runs Streamlit.\n"
                "  Recommended run command:\n"
                "    .venv\\Scripts\\python.exe -m streamlit run app.py\n"
                "- And install:\n"
                "    .venv\\Scripts\\python.exe -m pip install mediapipe\n"
            ) from e

    def _ensure_model(self) -> Path:
        """
        Downloads the BlazeFace short-range TFLite model for MediaPipe Tasks.
        Cached under `.cache/mediapipe/` in the project root by default.
        """
        url = (
            "https://storage.googleapis.com/mediapipe-models/face_detector/"
            "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
        )

        # Allow override for advanced users.
        override = os.environ.get("CULL_FACE_MODEL_PATH")
        if override:
            p = Path(override)
            if not p.exists():
                raise FileNotFoundError(f"CULL_FACE_MODEL_PATH not found: {override}")
            return p

        root = Path.cwd()
        cache_dir = root / ".cache" / "mediapipe"
        cache_dir.mkdir(parents=True, exist_ok=True)
        model_path = cache_dir / "blaze_face_short_range.tflite"
        if model_path.exists() and model_path.stat().st_size > 0:
            return model_path

        urllib.request.urlretrieve(url, model_path)  # noqa: S310
        return model_path

    def detect(self, record: ImageRecord) -> None:
        if record.image is None:
            _clear_face_fields(record)
            return

        img = record.image

        # MediaPipe expects RGB uint8.
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)

        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=img)
        res = self._detector.detect(mp_image)
        dets = getattr(res, "detections", None) or []
        _annotate_record(record, img, dets, self.blink_openness_threshold)

