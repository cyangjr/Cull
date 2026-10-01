"""One worker task per image: load, then the shared per-image pipeline."""

from __future__ import annotations

import multiprocessing
import traceback
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed

from .config import PipelineConfig
from .utils import ImageRecord

_WORKER_CTX = None


def config_to_dict(config: PipelineConfig) -> dict:
    """Convert PipelineConfig to a picklable dict for worker initializers."""
    from dataclasses import asdict

    return asdict(config)


def _config_from_dict(config_dict: dict) -> PipelineConfig:
    config = PipelineConfig()
    for key, value in config_dict.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config


def _init_worker(config_dict: dict) -> None:
    """Build one ProcessContext per worker. FaceDetector is not rebuilt per image."""
    global _WORKER_CTX
    from .process import build_context

    _WORKER_CTX = build_context(_config_from_dict(config_dict))


def worker_process_image(path: str) -> ImageRecord | None:
    """Load one image and run the shared stage order. None means skip it."""
    try:
        from .process import process_record
        from .utils import ImageLoader

        if _WORKER_CTX is None:
            return None
        record = ImageLoader().load(path)
        process_record(record, _WORKER_CTX)
        return record
    except Exception:
        traceback.print_exc()
        return None


def process_paths_parallel(
    paths: list[str],
    config: PipelineConfig,
    num_workers: int,
    progress_callback: Callable[[float, str], None] | None = None,
) -> list[ImageRecord]:
    """Score every path in a spawn pool. The parent still dedups and ranks."""
    ctx = multiprocessing.get_context("spawn")
    records: list[ImageRecord] = []
    total = max(1, len(paths))
    done = 0
    with ProcessPoolExecutor(
        max_workers=num_workers,
        mp_context=ctx,
        initializer=_init_worker,
        initargs=(config_to_dict(config),),
    ) as executor:
        futures = [executor.submit(worker_process_image, path) for path in paths]
        for future in as_completed(futures):
            done += 1
            record = future.result()
            if record is not None:
                records.append(record)
            if progress_callback:
                progress_callback(done / total, "Processing...")
    return records
