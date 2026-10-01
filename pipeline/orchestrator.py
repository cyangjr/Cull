from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

from .config import PipelineConfig
from .parallel import process_paths_parallel
from .process import build_context, process_record
from .scorer import DuplicateFilter
from .utils import DeviceManager, ImageLoader, ImageRecord, ModelRegistry


class _StageTimer:
    """Accumulates wall-clock time per named stage and prints a summary report."""

    def __init__(self) -> None:
        self._totals: dict[str, float] = defaultdict(float)

    def add(self, stage: str, elapsed: float) -> None:
        self._totals[stage] += elapsed

    def report(self, n_images: int) -> None:
        total = sum(self._totals.values())
        per_img = total / max(1, n_images)
        print(f"\n[Cull] Timing — {n_images} images | {total:.1f}s total | {per_img:.2f}s/img")
        print(f"  {'Stage':<28} {'Total':>8}  {'%':>6}  {'Per img':>8}")
        print(f"  {'-' * 28} {'-' * 8}  {'-' * 6}  {'-' * 8}")
        for stage, t in sorted(self._totals.items(), key=lambda kv: -kv[1]):
            pct = 100.0 * t / (total + 1e-9)
            avg = t / max(1, n_images)
            print(f"  {stage:<28} {t:>7.1f}s  {pct:>5.1f}%  {avg:>7.3f}s")
        print()


class CullPipeline:
    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig.load()
        self.device = DeviceManager().get_device(self.config.device)
        print(f"[Cull] Device: {DeviceManager.describe(self.device)}")

        # Phase A: ModelRegistry not used yet, but present for Phase C shape.
        self.model_registry = ModelRegistry(device=self.device)

        self.image_loader = ImageLoader()
        self.ctx = build_context(self.config)
        self.duplicate_filter = DuplicateFilter(self.config)

    def run(
        self,
        folder_path: str,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> list[ImageRecord]:
        """
        Run the culling pipeline on a folder of images.

        Every mode calls process_record, so the stage order does not depend on
        the device. num_workers == 0 is always the in-process path: the worker
        helper treats 0 as "auto" and must not be consulted. CUDA and MPS use
        the same in-process path; the batch size is only a RAM window.
        """
        if self.config.num_workers == 0 or self.device in ("cuda", "mps"):
            return self._run_batched(folder_path, progress_callback)

        from .cpu_utils import get_safe_worker_count

        workers = get_safe_worker_count(self.config.num_workers)
        if workers <= 1:
            return self._run_batched(folder_path, progress_callback)
        return self._run_parallel(folder_path, progress_callback, workers)

    def _ram_batch_size(self) -> int:
        if self.device in ("cuda", "mps"):
            return max(1, int(self.config.gpu_batch_size))
        return max(1, int(self.config.batch_size))

    def _list_paths(self, folder_path: str) -> list[Path]:
        folder = ImageLoader._normalise_path(folder_path)
        return [p for p in sorted(folder.iterdir()) if p.is_file() and self.image_loader.is_supported(p)]

    def _run_batched(
        self,
        folder_path: str,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> list[ImageRecord]:
        paths = self._list_paths(folder_path)
        n = max(1, len(paths))
        batch_size = self._ram_batch_size()
        records: list[ImageRecord] = []
        timer = _StageTimer()

        for start in range(0, len(paths), batch_size):
            batch: list[ImageRecord] = []
            for path in paths[start : start + batch_size]:
                try:
                    started = time.perf_counter()
                    batch.append(self.image_loader.load(str(path)))
                    timer.add("load", time.perf_counter() - started)
                except Exception:
                    continue

            for offset, record in enumerate(batch):
                if progress_callback:
                    progress_callback((start + offset) / n, f"Processing {record.filename}")
                process_record(record, self.ctx, timer)
            records.extend(batch)

        return self._dedup_and_rank(records, timer, len(paths), progress_callback)

    def _run_parallel(
        self,
        folder_path: str,
        progress_callback: Callable[[float, str], None] | None = None,
        num_workers: int = 2,
    ) -> list[ImageRecord]:
        paths = [str(path) for path in self._list_paths(folder_path)]
        timer = _StageTimer()
        started = time.perf_counter()
        records = process_paths_parallel(paths, self.config, num_workers, progress_callback)
        timer.add("parallel", time.perf_counter() - started)
        return self._dedup_and_rank(records, timer, len(paths), progress_callback)

    def _dedup_and_rank(
        self,
        records: list[ImageRecord],
        timer: _StageTimer,
        n_images: int,
        progress_callback: Callable[[float, str], None] | None,
    ) -> list[ImageRecord]:
        passed = [record for record in records if record.passed_gate]
        if self.config.enable_dedup and len(passed) > 1:
            passed.sort(key=lambda record: record.filename)
            started = time.perf_counter()
            self.duplicate_filter.filter(passed)
            timer.add("dedup", time.perf_counter() - started)

        for record in passed:
            self.ctx.final_scorer.compute(record)

        if progress_callback:
            progress_callback(1.0, "Ranking")

        timer.report(n_images)
        return self.ctx.final_scorer.rank(records)
