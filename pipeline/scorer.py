from __future__ import annotations

import math

import cv2
import numpy as np

from .config import PipelineConfig
from .utils import ImageRecord

# ---------------------------------------------------------------------------
# Torch availability — checked once and cached.
# All GPU methods guard with this so the pipeline works without PyTorch.
# ---------------------------------------------------------------------------
_TORCH_AVAILABLE: bool | None = None


def _torch_available() -> bool:
    global _TORCH_AVAILABLE
    if _TORCH_AVAILABLE is None:
        try:
            import torch  # noqa: F401
            _TORCH_AVAILABLE = True
        except ImportError:
            _TORCH_AVAILABLE = False
    return _TORCH_AVAILABLE


def _resize_long_edge(image: np.ndarray, long_edge: int) -> np.ndarray:
    """Resize so the long edge is exactly `long_edge` (up or down)."""
    height, width = image.shape[:2]
    long_side = max(height, width)
    if long_side == long_edge:
        return image
    if width >= height:
        new_w = long_edge
        new_h = max(1, int(round(height * (long_edge / float(width)))))
    else:
        new_h = long_edge
        new_w = max(1, int(round(width * (long_edge / float(height)))))
    interp = cv2.INTER_AREA if long_side > long_edge else cv2.INTER_CUBIC
    return cv2.resize(image, (new_w, new_h), interpolation=interp)


class SharpnessScorer:
    def __init__(self, long_edge: int = 512) -> None:
        self.long_edge = int(long_edge)

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
        # Normalise resolution first so a 24MP frame and its preview score alike.
        resized = _resize_long_edge(region, self.long_edge)
        gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
        # Sigma 1 suppresses high-ISO grain before the Laplacian.
        denoised = cv2.GaussianBlur(gray, (0, 0), 1.0)
        variance = float(cv2.Laplacian(denoised, cv2.CV_64F).var())
        score = 1.0 - math.exp(-variance / 40.0)
        return float(max(0.0, min(1.0, score)))

    def score_batch_gpu(self, regions: list[np.ndarray], device: str) -> list[float]:
        """
        Same scores as score(), one region at a time.

        `device` is accepted so older callers still import, and ignored so a
        GPU path cannot drift from the resolution-stable Laplacian.

        Args:
            regions: Pre-selected H×W×3 uint8 numpy arrays (one per image).
            device:  "cuda" or "mps" (also accepts "cpu" for parity testing).

        Returns:
            List of float sharpness scores in [0, 1].

        """
        del device
        return [self._laplacian_variance(region) for region in regions]


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
        # Technical clipping only. A dark or bright mean is not a penalty.
        shadow = float(hist[:5].sum()) / total          # bins 0..4
        highlight = float(hist[251:].sum()) / total     # bins 251..255
        penalty = min(
            1.0,
            max(0.0, shadow - 0.02) * 4.0 + max(0.0, highlight - 0.05) * 4.0,
        )
        return float(max(0.0, 1.0 - penalty))


class WhiteBalanceScorer:
    def score(self, record: ImageRecord) -> None:
        if record.image is None:
            record.white_balance_score = None
            return
        record.white_balance_score = self._channel_deviation(record.image)

    def _channel_deviation(self, image: np.ndarray) -> float:
        # Mean channel deviation from neutral grey: lower is better.
        # dtype=float32 avoids numpy's default int64 accumulation on uint8 arrays,
        # which is slow on large images.
        means = image.mean(axis=(0, 1), dtype=np.float32)  # R,G,B
        m = float(means.mean()) + 1e-9
        dev = float(np.abs(means - m).mean() / m)  # relative deviation
        # Flat through intentional color (golden hour, etc.). Only a severe
        # cast past 0.40 relative deviation starts to fall.
        if dev <= 0.40:
            score = 1.0
        else:
            score = 1.0 / (1.0 + (dev - 0.40) * 4.0)
        return float(max(0.0, min(1.0, score)))


class FinalScorer:
    def __init__(self, config: PipelineConfig) -> None:
        self.weights = dict(config.final_score_weights)
        self.motion_blur_penalty = float(config.motion_blur_score_penalty)

    def compute(self, record: ImageRecord) -> None:
        # Missing exposure / white balance stay neutral so a disabled stage
        # does not punish the frame. Unknown eyes (or no face) stay neutral too.
        if record.has_faces and record.eyes_open_score is not None:
            subject = float(max(0.0, min(1.0, record.eyes_open_score)))
        else:
            subject = 1.0
        composition = 0.60 if record.composition_score is None else float(record.composition_score)
        exposure = 1.0 if record.exposure_score is None else float(record.exposure_score)
        white_balance = 1.0 if record.white_balance_score is None else float(record.white_balance_score)

        components: dict[str, float] = {
            "sharpness": float(record.sharpness_score or 0.0),
            "exposure": exposure,
            "white_balance": white_balance,
            "aesthetic": float((record.aesthetic_score or 0.0) / 10.0),
            "subject": subject,
            "composition": composition,
        }

        weights = {k: float(v) for k, v in self.weights.items() if v is not None}
        denom = sum(weights.values()) or 1.0
        score = sum(components.get(k, 0.0) * w for k, w in weights.items()) / denom

        # Directional smear only. Eye openness is already the subject term.
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
        record.motion_blur_detected = bool(self._directional_smear(record.image))

    def _directional_smear(self, image: np.ndarray) -> bool:
        """True only for a strong one-axis smear, not a horizon or defocus."""
        resized = _resize_long_edge(image, 512)
        gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (0, 0), 0.8)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mx = float(np.mean(np.abs(gx)))
        my = float(np.mean(np.abs(gy)))
        weak = min(mx, my)
        strong = max(mx, my)
        aniso = strong / (weak + 1e-6)
        # A missing axis (blinds, a lone horizon) has essentially zero energy.
        # A smear leaves a residual ramp on the collapsed axis.
        return bool(aniso >= 3.0 and strong >= 25.0 and 2.0 <= weak <= strong / 3.5)

    def detect_batch_gpu(self, images: list[np.ndarray], device: str) -> list[bool]:
        """Same flags as detect(). `device` is ignored so results cannot drift."""
        del device
        return [self._directional_smear(image) for image in images]


class AestheticScorer:
    """
    Tonal separation only (0..10). Saturation is ignored so black-and-white
    is not punished. Replaced by NIMA in a later upgrade.
    """

    def score(self, record: ImageRecord) -> None:
        if record.image is None:
            record.aesthetic_score = None
            return

        resized = _resize_long_edge(record.image, 512)
        gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
        contrast = float(np.std(gray))
        score = (1.0 - math.exp(-contrast / 18.0)) * 10.0
        record.aesthetic_score = float(max(0.0, min(10.0, score)))

    def score_batch_gpu(self, images: list[np.ndarray], device: str) -> list[float]:
        """Same scores as score(). `device` is ignored so saturation cannot leak back in."""
        del device
        scores: list[float] = []
        for image in images:
            record = ImageRecord(path="", filename="", image=image)
            self.score(record)
            scores.append(float(record.aesthetic_score or 0.0))
        return scores


class CompositionTagger:
    def tag(self, record: ImageRecord) -> None:
        if record.image is None:
            return
        tags: list[str] = []

        thirds = self._check_rule_of_thirds(record)
        symmetry = self._check_symmetry(record)
        negative = self._check_negative_space(record)
        if thirds:
            tags.append("rule_of_thirds")
        if symmetry:
            tags.append("symmetry")
        if negative:
            tags.append("negative_space")

        # leading_lines is intentionally omitted for now (needs more robust line clustering).
        record.composition_tags = sorted(set(record.composition_tags + tags))

        score = 0.60
        if thirds:
            score += 0.20
        if symmetry:
            score += 0.10
        if negative:
            score += 0.10
        record.composition_score = float(max(0.0, min(1.0, score)))

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
        # Time-window clustering for parseable ISO timestamps. Records with no
        # timestamp (or an unparseable one) each form their own group.
        import datetime as _dt

        def parse(r: ImageRecord):
            ts = (r.exif or {}).get("timestamp")
            if not ts:
                return None
            try:
                return _dt.datetime.fromisoformat(str(ts))
            except Exception:
                return None

        groups: list[list[ImageRecord]] = []
        parsed: list[tuple[ImageRecord, _dt.datetime]] = []
        for r in records:
            t = parse(r)
            if t is None:
                groups.append([r])
            else:
                parsed.append((r, t))

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
        # Ensure hashes exist (avoid keeping full images resident).
        hashed: list[ImageRecord] = []
        for r in records:
            if r.perceptual_hash:
                hashed.append(r)
                continue
            if r.image is None:
                continue
            try:
                r.perceptual_hash = self._perceptual_hash(r.image)
                hashed.append(r)
            except Exception:
                continue

        n = len(hashed)
        parent = list(range(n))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i: int, j: int) -> None:
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[rj] = ri

        for i in range(n):
            hi = hashed[i].perceptual_hash or ""
            for j in range(i + 1, n):
                hj = hashed[j].perceptual_hash or ""
                if _hamming_hex(hi, hj) <= self.hash_threshold:
                    union(i, j)

        components: dict[int, list[ImageRecord]] = {}
        for i, record in enumerate(hashed):
            components.setdefault(find(i), []).append(record)

        for index, members in enumerate(components.values()):
            if len(members) < 2:
                continue
            hero = self._choose_hero(members)
            component_id = f"{group_id}-{index}"
            for member in members:
                member.duplicate_group = component_id
                member.is_duplicate = member is not hero

    @staticmethod
    def _hero_key(record: ImageRecord) -> tuple[int, float, float]:
        # Open eyes beat a sharper blink. Higher is better.
        # Scores are rounded so worker-to-worker float noise cannot flip the hero.
        open_rank = 0 if record.blink_detected else 1
        eyes = 1.0 if record.eyes_open_score is None else float(record.eyes_open_score)
        score = float(record.final_score or 0.0)
        return (open_rank, round(eyes, 4), round(score, 4))

    @classmethod
    def _choose_hero(cls, members: list[ImageRecord]) -> ImageRecord:
        # Highest eye rank and score. Equal frames keep the earliest filename.
        def sort_key(record: ImageRecord) -> tuple:
            open_rank, eyes, score = cls._hero_key(record)
            return (-open_rank, -eyes, -score, record.filename)

        return min(members, key=sort_key)


def _hamming_hex(a: str, b: str) -> int:
    # imagehash string is hex; compare bits via integer xor
    try:
        ia = int(a, 16)
        ib = int(b, 16)
    except Exception:
        return 999
    return int((ia ^ ib).bit_count())

