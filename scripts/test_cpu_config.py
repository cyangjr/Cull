"""Test CPU worker configuration."""
from __future__ import annotations

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.cpu_utils import get_cpu_info, get_safe_worker_count


def test_scenarios():
    """Test different worker count scenarios."""
    info = get_cpu_info()
    print("=== CPU Information ===")
    print(f"Total cores: {info['total_cores']}")
    print(f"Recommended workers: {info['recommended_workers']}")
    print(f"Max safe workers: {info['max_workers']}")
    print()

    print("=== Test Scenarios ===")

    # Auto mode
    auto = get_safe_worker_count(None)
    print(f"Auto (None): {auto} workers")

    # Explicit values
    for requested in [0, 1, 2, 4, 8, 16, 100]:
        workers = get_safe_worker_count(requested)
        print(f"Requested {requested}: {workers} workers")

    print()
    print("=== Custom Fractions ===")

    # Different fractions
    for fraction in [0.25, 0.5, 0.75, 1.0]:
        workers = get_safe_worker_count(None, max_fraction=fraction)
        print(f"Fraction {fraction:.0%}: {workers} workers")


def test_edge_cases():
    """Test edge cases."""
    print("\n=== Edge Case Tests ===")

    # Test with different min_workers
    for min_w in [1, 2, 4]:
        workers = get_safe_worker_count(None, min_workers=min_w)
        print(f"Min workers {min_w}: {workers} workers")

    # Test with max_workers limit
    for max_w in [1, 2, 4]:
        workers = get_safe_worker_count(None, max_workers=max_w)
        print(f"Max workers {max_w}: {workers} workers")


def test_config_loading():
    """Test loading from config.yaml."""
    print("\n=== Config Loading Test ===")
    from pipeline.config import PipelineConfig

    config = PipelineConfig.load()
    print(f"Config num_workers: {config.num_workers}")

    # Calculate actual workers that will be used
    actual = get_safe_worker_count(config.num_workers)
    print(f"Actual workers: {actual}")

    if config.num_workers is None:
        print("[OK] Auto-detect mode (will use 50% of cores)")
    elif config.num_workers == 0:
        print("[OK] Sequential mode (multiprocessing disabled)")
    else:
        print(f"[OK] Explicit mode (requested {config.num_workers} workers)")


if __name__ == "__main__":
    test_scenarios()
    test_edge_cases()
    test_config_loading()

    print("\n=== Summary ===")
    info = get_cpu_info()
    print(f"Your system has {info['total_cores']} cores")
    print(f"Recommended: {info['recommended_workers']} workers (50% of cores)")
    print(f"Maximum safe: {info['max_workers']} workers (leaves 1 core free)")
    print("\nTo change this, edit config.yaml and set num_workers to:")
    print("  - null (auto, uses 50% of cores) <- recommended")
    print("  - 0 (disable multiprocessing, for debugging)")
    print(f"  - 1-{info['max_workers']} (explicit worker count)")
