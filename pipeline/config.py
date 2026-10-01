from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class PipelineConfig:
    config_path: str = "config.yaml"

    # scorer weights — technical quality. Legacy white_balance / aesthetic
    # keys are still accepted when present in a yaml file.
    final_score_weights: dict[str, float] = field(
        default_factory=lambda: {
            "sharpness": 0.55,
            "exposure": 0.20,
            "subject": 0.15,
            "composition": 0.10,
        }
    )

    # gate thresholds
    sharpness_gate_threshold: float = 0.3

    # motion blur penalty (applied in FinalScorer, not the gate)
    motion_blur_score_penalty: float = 0.75

    # detector settings (Phase C)
    yolo_confidence_threshold: float = 0.4

    # duplicate filter settings (Phase C)
    hash_threshold: int = 8
    timestamp_window_s: float = 8.0

    # analysis / blink knobs
    sharpness_analysis_long_edge: int = 512
    blink_openness_threshold: float = 0.28

    # milestone C toggles
    enable_router: bool = True
    enable_object_detector: bool = True
    enable_saliency_detector: bool = True
    enable_face_detector: bool = True
    enable_motion_blur: bool = True
    enable_aesthetic: bool = True
    enable_composition_tags: bool = True
    enable_dedup: bool = True

    # milestone C knobs
    min_face_detection_confidence: float = 0.5
    saliency_peak_threshold: float = 0.6

    # memory/perf
    batch_size: int = 16
    release_pixel_data: bool = True
    num_workers: int | None = None  # None = auto-detect (40% of cores)

    # device / GPU
    device: str = "auto"      # auto|cpu|cuda|mps
    gpu_batch_size: int = 8   # images per GPU batch

    @staticmethod
    def load(path: str | None = None) -> "PipelineConfig":
        cfg = PipelineConfig()
        cfg_path = Path(path or cfg.config_path)
        if not cfg_path.exists():
            return cfg

        data: dict[str, Any] = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}

        if isinstance(data.get("final_score_weights"), dict):
            cfg.final_score_weights = {
                k: float(v) for k, v in data["final_score_weights"].items() if v is not None
            }

        float_keys = {
            "sharpness_gate_threshold",
            "motion_blur_score_penalty",
            "yolo_confidence_threshold",
            "timestamp_window_s",
            "min_face_detection_confidence",
            "saliency_peak_threshold",
            "blink_openness_threshold",
        }
        int_keys = {
            "hash_threshold",
            "batch_size",
            "num_workers",
            "gpu_batch_size",
            "sharpness_analysis_long_edge",
        }
        bool_keys = {
            "enable_router",
            "enable_object_detector",
            "enable_saliency_detector",
            "enable_face_detector",
            "enable_motion_blur",
            "enable_aesthetic",
            "enable_composition_tags",
            "enable_dedup",
            "release_pixel_data",
        }

        for k, v in data.items():
            # Special handling for device: validate against allowed values
            if k == "device":
                if isinstance(v, str) and v in ("auto", "cpu", "cuda", "mps"):
                    cfg.device = v
                continue

            # Special handling for num_workers: None is valid (auto-detect)
            if k == "num_workers":
                if v is None:
                    cfg.num_workers = None
                else:
                    cfg.num_workers = int(v) if v != 0 else 0
                continue

            if v is None:
                continue
            if k in float_keys:
                setattr(cfg, k, float(v))
            elif k in int_keys:
                setattr(cfg, k, int(v))
            elif k in bool_keys:
                setattr(cfg, k, bool(v))

        return cfg

