from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path

import numpy as np

from .utils import ImageRecord


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


class FaceDetector:
    def __init__(self, min_detection_confidence: float = 0.5) -> None:
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
            record.has_faces = False
            record.eye_region = None
            return

        img = record.image
        h, w = img.shape[:2]

        # MediaPipe expects RGB uint8.
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)

        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=img)
        res = self._detector.detect(mp_image)
        dets = getattr(res, "detections", None) or []
        if not dets:
            record.has_faces = False
            record.eye_region = None
            return

        det = dets[0]
        record.has_faces = True

        # Tasks API provides a face bounding box; use its upper region as an eye-proxy crop.
        bb = det.bounding_box
        x0, y0 = float(bb.origin_x), float(bb.origin_y)
        bw, bh = float(bb.width), float(bb.height)
        x1, y1 = x0 + bw, y0 + bh

        # Expand region a bit to be robust, and bias toward upper face (eyes).
        pad_x = 0.15 * (x1 - x0)
        pad_y = 0.20 * (y1 - y0)
        x0 = max(0, int(x0 - pad_x))
        y0 = max(0, int(y0 - pad_y))
        x1 = min(w, int(x1 + pad_x))
        y1 = min(h, int(y0 + (y1 - y0) * 0.65))  # top ~65% of face bbox

        rw = max(1, x1 - x0)
        rh = max(1, y1 - y0)
        record.eye_region = (x0, y0, rw, rh)

