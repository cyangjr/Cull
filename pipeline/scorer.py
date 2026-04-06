from __future__ import annotations

import math

import cv2
import numpy as np

from .config import PipelineConfig
from .utils import ImageRecord


class SharpnessScorer:
    def score(self, record: ImageRecord) -> None:
        if record.image is None:
            record.sharpness_score = None
            return
        region = self._select_region(record)
        record.sharpness_score = self._laplacian_variance(region)

    def _select_region(self, record: ImageRecord) -> np.ndarray:
        assert record.image is not None
        # Phase B: if face detected, score the eye_region crop.
        if record.has_faces and record.eye_region is not None:
            x, y, w, h = record.eye_region
            x2 = min(record.image.shape[1], x + w)
            y2 = min(record.image.shape[0], y + h)
            if x2 > x and y2 > y:
                return record.image[y:y2, x:x2]
        # Milestone C: if subject bbox exists, score within bbox.
        if record.subject_bbox is not None:
            x, y, w, h = record.subject_bbox
            x2 = min(record.image.shape[1], x + w)
            y2 = min(record.image.shape[0], y + h)
            if x2 > x and y2 > y:
                return record.image[y:y2, x:x2]
        # Milestone C: if saliency peak region exists, score within it.
        if record.saliency_peak_region is not None:
            x, y, w, h = record.saliency_peak_region
            x2 = min(record.image.shape[1], x + w)
            y2 = min(record.image.shape[0], y + h)
            if x2 > x and y2 > y:
                return record.image[y:y2, x:x2]
        return record.image

    def _laplacian_variance(self, region: np.ndarray) -> float:
        gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
        v = cv2.Laplacian(gray, cv2.CV_64F).var()
        # Normalise to 0..1 with a soft saturation curve (empirical).
        score = 1.0 - math.exp(-float(v) / 500.0)
        return float(max(0.0, min(1.0, score)))


class ExposureScorer:
    def score(self, record: ImageRecord) -> None:
        if record.image is None:
            record.exposure_score = None
            return
        record.exposure_score = self._analyse_histogram(record.image)

    def _analyse_histogram(self, image: np.ndarray) -> float:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).reshape(-1)
        total = float(hist.sum()) + 1e-9
        dark_clip = float(hist[:5].sum()) / total
        bright_clip = float(hist[251:].sum()) / total
        penalty = min(1.0, (dark_clip + bright_clip) * 3.0)
        return float(max(0.0, 1.0 - penalty))


class WhiteBalanceScorer:
    def score(self, record: ImageRecord) -> None:
        if record.image is None:
            record.white_balance_score = None
            return
        record.white_balance_score = self._channel_deviation(record.image)

    def _channel_deviation(self, image: np.ndarray) -> float:
        # Mean channel deviation from neutral grey: lower is better.
        means = image.reshape(-1, 3).mean(axis=0)  # R,G,B
        m = float(means.mean()) + 1e-9
        dev = float(np.abs(means - m).mean() / m)  # relative deviation
        # Map 0..~0.3 into 1..0 with a gentle curve.
        score = 1.0 / (1.0 + dev * 8.0)
        return float(max(0.0, min(1.0, score)))


class FinalScorer:
    def __init__(self, config: PipelineConfig) -> None:
        self.weights = dict(config.final_score_weights)
        self.motion_blur_penalty = float(config.motion_blur_score_penalty)

    def compute(self, record: ImageRecord) -> None:
        # Missing components contribute 0.0 (weights remain as configured).
        components: dict[str, float] = {
            "sharpness": float(record.sharpness_score or 0.0),
            "exposure": float(record.exposure_score or 0.0),
            "white_balance": float(record.white_balance_score or 0.0),
            "aesthetic": float((record.aesthetic_score or 0.0) / 10.0),
        }

        weights = {k: float(v) for k, v in self.weights.items() if v is not None}
        denom = sum(weights.values()) or 1.0
        score = sum(components.get(k, 0.0) * w for k, w in weights.items()) / denom

        # Motion blur penalty applies later when motion_blur_detected exists (Phase C),
        # but keep behavior aligned with XML: penalty modifies final score, never gate.
        if record.motion_blur_detected:
            score *= float(self.motion_blur_penalty)

        record.final_score = float(max(0.0, min(1.0, score)))

    def rank(self, records: list[ImageRecord]) -> list[ImageRecord]:
        return sorted(records, key=lambda r: (r.final_score or 0.0), reverse=True)


class MotionBlurDetector:
    def detect(self, record: ImageRecord) -> None:
        if record.image is None:
            record.motion_blur_detected = None
            return
        record.motion_blur_detected = bool(self._fft_directional_check(record.image))

    def _fft_directional_check(self, image: np.ndarray) -> bool:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)
        gray = gray.astype(np.float32) / 255.0

        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        mag = np.log(np.abs(fshift) + 1e-6)

        # Compare energy distribution along horizontal vs vertical axes.
        h, w = mag.shape
        cy, cx = h // 2, w // 2
        band = max(2, min(h, w) // 20)
        horiz = mag[cy - band : cy + band, :].mean()
        vert = mag[:, cx - band : cx + band].mean()

        ratio = max(horiz, vert) / (min(horiz, vert) + 1e-6)
        return bool(ratio > 1.35)


class AestheticScorer:
    """
    Milestone C fallback: heuristic aesthetic score (0..10) based on contrast + saturation.
    Replaced by NIMA model in a later upgrade.
    """

    def score(self, record: ImageRecord) -> None:
        if record.image is None:
            record.aesthetic_score = None
            return

        img = record.image.astype(np.float32) / 255.0
        hsv = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        sat = float(hsv[:, :, 1].mean()) / 255.0
        gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        contrast = float(gray.std())

        # Map to 0..10 (simple, bounded).
        score = (0.6 * min(1.0, contrast * 3.0) + 0.4 * min(1.0, sat * 1.5)) * 10.0
        record.aesthetic_score = float(max(0.0, min(10.0, score)))


class CompositionTagger:
    def tag(self, record: ImageRecord) -> None:
        if record.image is None:
            return
        tags: list[str] = []

        if self._check_rule_of_thirds(record):
            tags.append("rule_of_thirds")
        if self._check_symmetry(record):
            tags.append("symmetry")
        if self._check_negative_space(record):
            tags.append("negative_space")

        # leading_lines is intentionally omitted for now (needs more robust line clustering).
        record.composition_tags = sorted(set(record.composition_tags + tags))

    def _subject_point(self, record: ImageRecord) -> tuple[float, float] | None:
        if record.subject_bbox is not None:
            x, y, w, h = record.subject_bbox
            return (x + w / 2.0, y + h / 2.0)
        if record.saliency_peak_region is not None:
            x, y, w, h = record.saliency_peak_region
            return (x + w / 2.0, y + h / 2.0)
        if record.eye_region is not None:
            x, y, w, h = record.eye_region
            return (x + w / 2.0, y + h / 2.0)
        return None

    def _check_rule_of_thirds(self, record: ImageRecord) -> bool:
        h, w = record.image.shape[:2]  # type: ignore[union-attr]
        p = self._subject_point(record)
        if p is None:
            return False
        x, y = p
        thirds_x = [w / 3.0, 2.0 * w / 3.0]
        thirds_y = [h / 3.0, 2.0 * h / 3.0]
        tol = 0.07 * min(w, h)
        return any(abs(x - tx) < tol for tx in thirds_x) and any(abs(y - ty) < tol for ty in thirds_y)

    def _check_symmetry(self, record: ImageRecord) -> bool:
        img = record.image
        assert img is not None
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (256, 256))
        flipped = cv2.flip(gray, 1)
        diff = np.mean(np.abs(gray.astype(np.float32) - flipped.astype(np.float32))) / 255.0
        return diff < 0.12

    def _check_negative_space(self, record: ImageRecord) -> bool:
        img = record.image
        assert img is not None
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (256, 256))
        edges = cv2.Canny(gray, 60, 120)
        edge_frac = float((edges > 0).mean())
        return edge_frac < 0.03


class DuplicateFilter:
    def __init__(self, config: PipelineConfig) -> None:
        self.hash_threshold = int(config.hash_threshold)
        self.timestamp_window_s = float(config.timestamp_window_s)

    def filter(self, records: list[ImageRecord]) -> None:
        # Expects only gate-passed records.
        groups = self._group_by_timestamp(records)
        for gi, g in enumerate(groups):
            self._mark_duplicates_in_group(g, group_id=f"t{gi}")

    def _perceptual_hash(self, image: np.ndarray) -> str:
        import imagehash
        from PIL import Image

        pil = Image.fromarray(image)
        return str(imagehash.phash(pil))

    def _group_by_timestamp(self, records: list[ImageRecord]) -> list[list[ImageRecord]]:
        # Simple clustering: if timestamps are missing, treat as one group.
        with_ts = []
        without_ts = []
        for r in records:
            ts = (r.exif or {}).get("timestamp")
            if ts:
                with_ts.append(r)
            else:
                without_ts.append(r)

        groups: list[list[ImageRecord]] = []
        if without_ts:
            groups.append(without_ts)

        # Parse ISO strings best-effort; fallback to unsorted.
        import datetime as _dt

        def parse(r: ImageRecord):
            try:
                return _dt.datetime.fromisoformat(str((r.exif or {}).get("timestamp")))
            except Exception:
                return None

        parsed = [(r, parse(r)) for r in with_ts]
        parsed = [(r, t) for r, t in parsed if t is not None]
        parsed.sort(key=lambda rt: rt[1])

        current: list[ImageRecord] = []
        last_t = None
        for r, t in parsed:
            if last_t is None or (t - last_t).total_seconds() <= self.timestamp_window_s:
                current.append(r)
            else:
                groups.append(current)
                current = [r]
            last_t = t
        if current:
            groups.append(current)

        return groups

    def _mark_duplicates_in_group(self, records: list[ImageRecord], group_id: str) -> None:
        # Ensure hashes exist (avoid keeping full images resident)
        hashes: list[tuple[ImageRecord, str]] = []
        for r in records:
            if r.perceptual_hash:
                hashes.append((r, r.perceptual_hash))
                continue
            if r.image is None:
                continue
            try:
                r.perceptual_hash = self._perceptual_hash(r.image)
                hashes.append((r, r.perceptual_hash))
            except Exception:
                continue

        # Pairwise compare (group sizes are typically small).
        for i, (ri, hi) in enumerate(hashes):
            if ri.is_duplicate:
                continue
            ri.duplicate_group = group_id
            for rj, hj in hashes[i + 1 :]:
                if rj.is_duplicate:
                    continue
                dist = _hamming_hex(hi, hj)
                if dist <= self.hash_threshold:
                    # Keep higher-scoring record.
                    si = ri.final_score or 0.0
                    sj = rj.final_score or 0.0
                    if si >= sj:
                        rj.is_duplicate = True
                        rj.duplicate_group = group_id
                    else:
                        ri.is_duplicate = True
                        ri.duplicate_group = group_id


def _hamming_hex(a: str, b: str) -> int:
    # imagehash string is hex; compare bits via integer xor
    try:
        ia = int(a, 16)
        ib = int(b, 16)
    except Exception:
        return 999
    return int((ia ^ ib).bit_count())

