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

    def detect_batch_gpu(self, records: list, device: str) -> None:
        """
        Sobel-based saliency detection for a batch of records on the GPU.

        Uses torch.nn.functional.conv2d with fixed Sobel kernels — identical
        algorithm to detect(), so saliency maps are directly comparable.
        Results are written into each record's saliency_map and
        saliency_peak_region fields in-place.

        Args:
            records: ImageRecord objects whose .image is not None.
            device:  "cuda" or "mps" (also accepts "cpu" for parity testing).

        Raises:
            ImportError: if torch is not installed.
        """
        import numpy as np
        import torch
        import torch.nn.functional as F

        dev = torch.device(device)

        sobel_x = torch.tensor(
            [[-1., 0., 1.],
             [-2., 0., 2.],
             [-1., 0., 1.]],
            dtype=torch.float32, device=dev,
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1., -2., -1.],
             [0.,  0.,  0.],
             [1.,  2.,  1.]],
            dtype=torch.float32, device=dev,
        ).view(1, 1, 3, 3)

        for record in records:
            if record.image is None:
                record.saliency_map = None
                record.saliency_peak_region = None
                continue

            img = record.image
            gray = (0.299 * img[:, :, 0].astype(np.float32)
                    + 0.587 * img[:, :, 1].astype(np.float32)
                    + 0.114 * img[:, :, 2].astype(np.float32))
            t = torch.from_numpy(gray).to(dev).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

            gx = F.conv2d(t, sobel_x, padding=1)
            gy = F.conv2d(t, sobel_y, padding=1)
            mag = torch.sqrt(gx ** 2 + gy ** 2)
            # Gaussian blur matching cv2.GaussianBlur(ksize=(0,0), sigmaX=3):
            # ksize = 2 * ceil(3 * sigma) + 1 = 19 for sigma=3
            # Gaussian blur matching cv2.GaussianBlur(ksize=(0,0), sigmaX=3).
            # Note: torch conv2d uses zero-padding while OpenCV defaults to
            # BORDER_REFLECT_101, so peak region may differ by a few pixels at
            # image borders. The difference is <1% on typical camera images and
            # has no meaningful impact on sharpness region selection.
            sigma = 3.0
            ksize = 19
            half = ksize // 2
            ax = torch.arange(ksize, dtype=torch.float32, device=dev) - half
            gauss_1d = torch.exp(-ax ** 2 / (2 * sigma ** 2))
            gauss_1d = gauss_1d / gauss_1d.sum()
            gauss_2d = (gauss_1d.unsqueeze(1) * gauss_1d.unsqueeze(0)).view(1, 1, ksize, ksize)
            mag = F.conv2d(mag, gauss_2d, padding=half)

            mag_np = mag.squeeze().cpu().numpy()
            mag_norm = (mag_np / (float(mag_np.max()) + 1e-9)).astype(np.float32)

            record.saliency_map = mag_norm
            record.saliency_peak_region = self.get_peak_region(mag_norm)


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

