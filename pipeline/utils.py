from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image, ImageOps


SourceType = Literal["pipeline", "json"]


class DeviceManager:
    _device: str | None = None

    @staticmethod
    def detect() -> str:
        # CUDA-first with safe CPU fallback; MPS becomes relevant on Apple Silicon.
        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                return "cuda"
            if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"

    def get_device(self, requested: str = "auto") -> str:
        """
        Resolve device string to the actual device to use.

        Args:
            requested: "auto" | "cpu" | "cuda" | "mps"
                - "auto": prefer cuda > mps > cpu
                - "cpu":  always CPU (skips detection)
                - "cuda" | "mps": use if available, else fall back to auto-detect

        Returns:
            One of "cuda", "mps", or "cpu".
        """
        if requested == "cpu":
            self._device = "cpu"
            return "cpu"
        if requested in ("cuda", "mps"):
            detected = self.detect()
            self._device = requested if detected == requested else detected
            return self._device
        # auto
        if self._device is None:
            self._device = self.detect()
        return self._device


@dataclass(slots=True)
class ImageRecord:
    # identity
    path: str
    filename: str
    source: SourceType = "pipeline"

    # pixel data (nullable on reload)
    image: np.ndarray | None = None
    saliency_map: np.ndarray | None = None

    # EXIF
    exif: dict[str, Any] = field(default_factory=dict)

    # routing
    scene_type: str | None = None  # 'object' | 'scene' | 'abstract'

    # detection
    subject_bbox: tuple[int, int, int, int] | None = None
    has_faces: bool = False
    eye_region: tuple[int, int, int, int] | None = None
    saliency_peak_region: tuple[int, int, int, int] | None = None

    # scores
    sharpness_score: float | None = None
    exposure_score: float | None = None
    white_balance_score: float | None = None
    motion_blur_detected: bool | None = None
    aesthetic_score: float | None = None

    # composition
    composition_tags: list[str] = field(default_factory=list)

    # pipeline output
    is_duplicate: bool = False
    duplicate_group: str | None = None
    perceptual_hash: str | None = None
    final_score: float | None = None
    passed_gate: bool = False

    def to_export_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Exclude pixel arrays and any other non-JSON-friendly objects.
        d["image"] = None
        d["saliency_map"] = None
        d["eye_region"] = None
        return d

    @staticmethod
    def from_json_dict(d: dict[str, Any]) -> "ImageRecord":
        return ImageRecord(
            path=str(d.get("path", "")),
            filename=str(d.get("filename", Path(str(d.get("path", ""))).name)),
            source="json",
            image=None,
            saliency_map=None,
            exif=dict(d.get("exif") or {}),
            scene_type=d.get("scene_type"),
            subject_bbox=tuple(d["subject_bbox"]) if d.get("subject_bbox") else None,
            has_faces=bool(d.get("has_faces", False)),
            eye_region=None,
            saliency_peak_region=tuple(d["saliency_peak_region"]) if d.get("saliency_peak_region") else None,
            sharpness_score=d.get("sharpness_score"),
            exposure_score=d.get("exposure_score"),
            white_balance_score=d.get("white_balance_score"),
            motion_blur_detected=d.get("motion_blur_detected"),
            aesthetic_score=d.get("aesthetic_score"),
            composition_tags=list(d.get("composition_tags") or []),
            is_duplicate=bool(d.get("is_duplicate", False)),
            duplicate_group=d.get("duplicate_group"),
            perceptual_hash=d.get("perceptual_hash"),
            final_score=d.get("final_score"),
            passed_gate=bool(d.get("passed_gate", False)),
        )


class ImageLoader:
    supported_extensions = [".jpg", ".jpeg", ".png", ".raw", ".heic"]

    @staticmethod
    def _normalise_path(p: str | Path) -> Path:
        s = str(p).strip()
        # Windows users often paste paths wrapped in quotes.
        if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
            s = s[1:-1].strip()
        return Path(s)

    def is_supported(self, path: str | Path) -> bool:
        return Path(path).suffix.lower() in self.supported_extensions

    def load_folder(self, folder_path: str) -> list[ImageRecord]:
        folder = self._normalise_path(folder_path)
        if not folder.exists() or not folder.is_dir():
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        records: list[ImageRecord] = []
        for p in sorted(folder.iterdir()):
            if p.is_file() and self.is_supported(p):
                try:
                    records.append(self.load(str(p)))
                except Exception:
                    # Skip unreadable files; keep pipeline resilient for big folders.
                    continue
        return records

    def load(self, path: str) -> ImageRecord:
        p = self._normalise_path(path)
        img = Image.open(p)
        img = ImageOps.exif_transpose(img).convert("RGB")
        image_np = np.array(img)

        exif = self._extract_exif(path)
        return ImageRecord(
            path=str(p),
            filename=p.name,
            source="pipeline",
            image=image_np,
            exif=exif,
        )

    def _extract_exif(self, path: str) -> dict[str, Any]:
        exif: dict[str, Any] = {}

        # 1) Try exifread (works well for JPEGs, and doesn't require loading pixels)
        try:
            import exifread  # type: ignore

            with open(path, "rb") as f:
                tags = exifread.process_file(f, details=False)

            def _get(*keys: str) -> str | None:
                for k in keys:
                    if k in tags:
                        return str(tags[k])
                return None

            dt = _get("EXIF DateTimeOriginal", "Image DateTime")
            if dt:
                # Common format: "2024:01:31 12:34:56"
                try:
                    exif["timestamp"] = datetime.strptime(dt, "%Y:%m:%d %H:%M:%S").isoformat()
                except Exception:
                    exif["timestamp"] = dt

            iso = _get("EXIF ISOSpeedRatings", "EXIF PhotographicSensitivity")
            if iso:
                exif["iso"] = iso

            exif["shutter_speed"] = _get("EXIF ExposureTime", "EXIF ShutterSpeedValue")
            exif["focal_length"] = _get("EXIF FocalLength")
            exif["aperture"] = _get("EXIF FNumber", "EXIF ApertureValue")
            exif["lens_id"] = _get("EXIF LensModel", "EXIF LensSpecification")
            return exif
        except Exception:
            pass

        # 2) Fallback to Pillow EXIF (less consistent, but better than nothing)
        try:
            img = Image.open(path)
            exif_raw = getattr(img, "getexif", lambda: None)()
            if exif_raw:
                exif["timestamp"] = exif_raw.get(36867) or exif_raw.get(306)  # DateTimeOriginal / DateTime
                exif["iso"] = exif_raw.get(34855)  # ISOSpeedRatings
        except Exception:
            pass

        return exif


class ModelRegistry:
    # Milestone C: basic lifecycle manager. Keep interface aligned with XML.
    def __init__(self, device: str) -> None:
        self.device = device
        self.registry: dict[str, dict[str, Any]] = {}

    def register(self, name: str, model_class: type, keep_loaded: bool = False, **init_kwargs: Any) -> None:
        self.registry[name] = {
            "class": model_class,
            "instance": None,
            "loaded": False,
            "keep_loaded": bool(keep_loaded),
            "init_kwargs": dict(init_kwargs),
        }

    def get(self, name: str) -> Any:
        if name not in self.registry:
            raise KeyError(f"Model not registered: {name}")

        entry = self.registry[name]
        if entry["instance"] is not None:
            return entry["instance"]

        # Unload any other non-pinned models (best-effort; some backends are CPU-only).
        for other_name, other in list(self.registry.items()):
            if other_name == name:
                continue
            if other.get("instance") is None:
                continue
            if other.get("keep_loaded"):
                continue
            self.unload(other_name)

        cls: type = entry["class"]
        kwargs = dict(entry.get("init_kwargs") or {})
        instance = cls(**kwargs)
        entry["instance"] = instance
        entry["loaded"] = True
        return instance

    def unload(self, name: str) -> None:
        entry = self.registry.get(name)
        if not entry:
            return
        if entry.get("keep_loaded"):
            return
        inst = entry.get("instance")
        if inst is None:
            return
        # Best-effort explicit close hook.
        close = getattr(inst, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass
        entry["instance"] = None
        entry["loaded"] = False

    def unload_all(self) -> None:
        for name in list(self.registry.keys()):
            entry = self.registry.get(name) or {}
            if entry.get("instance") is None:
                continue
            # Allow unloading pinned models here (session end).
            entry["keep_loaded"] = False
            self.unload(name)
