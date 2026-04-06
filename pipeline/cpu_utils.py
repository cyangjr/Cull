"""CPU core management utilities with safe defaults."""
from __future__ import annotations

import os


def get_safe_worker_count(
    requested: int | None = None,
    max_fraction: float = 0.4,
    min_workers: int = 1,
    max_workers: int | None = None,
) -> int:
    """
    Calculate safe number of worker processes.

    Args:
        requested: User-requested worker count (None = auto)
        max_fraction: Maximum fraction of total cores to use (default 0.4 = 40%, tested optimal)
        min_workers: Minimum workers to return (default 1)
        max_workers: Maximum workers to return (default None = no limit)

    Returns:
        Safe worker count between min_workers and available cores

    Strategy:
        - Auto mode (requested=None): Use max_fraction of available cores
        - Explicit mode (requested=N): Use exactly N, clamped to safe range
        - Always ensure at least min_workers
        - Always leave at least 1 core free for OS/UI

    Examples:
        # 20-core machine, default 40%:
        get_safe_worker_count()  # Returns 8 (40% of 20, tested optimal)

        # 8-core machine, default 40%:
        get_safe_worker_count()  # Returns 3 (40% of 8)

        # 8-core machine, request 12:
        get_safe_worker_count(12)  # Returns 7 (max available - 1)

        # 2-core machine:
        get_safe_worker_count()  # Returns 1 (always leaves 1 core free)

        # 1-core machine:
        get_safe_worker_count()  # Returns 1 (min_workers guarantee)
    """
    # Detect available cores
    try:
        cpu_count = os.cpu_count() or 1
    except Exception:
        cpu_count = 1

    # Calculate maximum safe workers (always leave 1 core free)
    max_safe = max(1, cpu_count - 1)

    # Auto mode: use fraction of available cores
    if requested is None or requested <= 0:
        workers = max(min_workers, int(cpu_count * max_fraction))
    else:
        # Explicit mode: use requested count
        workers = requested

    # Apply limits
    workers = max(min_workers, workers)  # At least min_workers
    workers = min(max_safe, workers)  # Don't exceed max_safe

    # Apply user-specified max if provided
    if max_workers is not None and max_workers > 0:
        workers = min(max_workers, workers)

    return workers


def get_cpu_info() -> dict[str, int | float]:
    """
    Get CPU information for diagnostics.

    Returns:
        Dictionary with:
        - total_cores: Total logical cores
        - recommended_workers: Safe worker count (40% of cores, tested optimal)
        - max_workers: Maximum safe workers (cores - 1)
    """
    try:
        cpu_count = os.cpu_count() or 1
    except Exception:
        cpu_count = 1

    return {
        "total_cores": cpu_count,
        "recommended_workers": get_safe_worker_count(),
        "max_workers": max(1, cpu_count - 1),
    }
