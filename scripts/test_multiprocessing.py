"""Test multiprocessing implementation."""
from __future__ import annotations

import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_sequential_vs_parallel(test_folder: str, num_workers: int = 4):
    """
    Compare sequential vs parallel performance.

    Args:
        test_folder: Path to folder with test images
        num_workers: Number of parallel workers to test
    """
    from pipeline import CullPipeline, PipelineConfig

    test_path = Path(test_folder)
    if not test_path.exists():
        print(f"Error: Test folder not found: {test_folder}")
        print("Please provide a folder with test images")
        return

    # Count images
    image_count = len(list(test_path.glob("*.jpg"))) + len(list(test_path.glob("*.png")))
    if image_count == 0:
        print(f"Error: No images found in {test_folder}")
        return

    print(f"Found {image_count} images in {test_folder}")
    print()

    # Test 1: Sequential (num_workers=0)
    print("=== Test 1: Sequential (num_workers=0) ===")
    config_seq = PipelineConfig.load()
    config_seq.num_workers = 0
    pipeline_seq = CullPipeline(config_seq)

    start = time.time()
    records_seq = pipeline_seq.run(test_folder)
    elapsed_seq = time.time() - start

    print(f"Time: {elapsed_seq:.2f}s")
    print(f"Processed: {len(records_seq)} images")
    print(f"Passed gate: {sum(1 for r in records_seq if r.passed_gate)} images")
    print(f"Speed: {image_count / elapsed_seq:.1f} images/second")
    print()

    # Test 2: Parallel (num_workers=N)
    print(f"=== Test 2: Parallel (num_workers={num_workers}) ===")
    config_par = PipelineConfig.load()
    config_par.num_workers = num_workers
    pipeline_par = CullPipeline(config_par)

    start = time.time()
    records_par = pipeline_par.run(test_folder)
    elapsed_par = time.time() - start

    print(f"Time: {elapsed_par:.2f}s")
    print(f"Processed: {len(records_par)} images")
    print(f"Passed gate: {sum(1 for r in records_par if r.passed_gate)} images")
    print(f"Speed: {image_count / elapsed_par:.1f} images/second")
    print()

    # Compare
    print("=== Comparison ===")
    speedup = elapsed_seq / elapsed_par
    print(f"Sequential: {elapsed_seq:.2f}s")
    print(f"Parallel:   {elapsed_par:.2f}s")
    print(f"Speedup:    {speedup:.2f}x")
    print()

    if speedup > 1.5:
        print(f"[OK] Parallel is {speedup:.1f}x faster!")
    elif speedup > 1.0:
        print(f"[INFO] Parallel is {speedup:.1f}x faster (modest improvement)")
    else:
        print(f"[WARNING] Parallel is slower! This may indicate overhead or small dataset")

    # Verify results match (approximately)
    print()
    print("=== Result Verification ===")
    if len(records_seq) != len(records_par):
        print(f"[WARNING] Different record counts: {len(records_seq)} vs {len(records_par)}")
    else:
        print(f"[OK] Same number of records: {len(records_seq)}")

    # Check passed counts
    passed_seq = sum(1 for r in records_seq if r.passed_gate)
    passed_par = sum(1 for r in records_par if r.passed_gate)
    if passed_seq != passed_par:
        print(f"[WARNING] Different passed counts: {passed_seq} vs {passed_par}")
    else:
        print(f"[OK] Same passed gate count: {passed_seq}")


def test_worker_scaling(test_folder: str):
    """
    Test how performance scales with worker count.

    Args:
        test_folder: Path to folder with test images
    """
    from pipeline import CullPipeline, PipelineConfig
    from pipeline.cpu_utils import get_cpu_info

    test_path = Path(test_folder)
    if not test_path.exists():
        print(f"Error: Test folder not found: {test_folder}")
        return

    image_count = len(list(test_path.glob("*.jpg"))) + len(list(test_path.glob("*.png")))
    if image_count == 0:
        print(f"Error: No images found in {test_folder}")
        return

    cpu_info = get_cpu_info()
    print(f"CPU Info: {cpu_info['total_cores']} cores")
    print(f"Testing with {image_count} images")
    print()

    # Test different worker counts
    worker_counts = [0, 1, 2, 4, 8, cpu_info["recommended_workers"], cpu_info["max_workers"]]
    worker_counts = sorted(set(worker_counts))  # Remove duplicates

    results = []

    for workers in worker_counts:
        config = PipelineConfig.load()
        config.num_workers = workers
        pipeline = CullPipeline(config)

        print(f"Testing with {workers} workers...", end=" ", flush=True)

        start = time.time()
        records = pipeline.run(test_folder)
        elapsed = time.time() - start

        speed = image_count / elapsed
        results.append((workers, elapsed, speed, len(records)))

        print(f"{elapsed:.2f}s ({speed:.1f} img/s)")

    # Print summary table
    print()
    print("=== Performance Summary ===")
    print(f"{'Workers':<10} {'Time (s)':<12} {'Speed (img/s)':<15} {'Speedup':<10}")
    print("-" * 50)

    baseline_time = results[0][1]  # Sequential time
    for workers, elapsed, speed, count in results:
        speedup = baseline_time / elapsed
        mode = "sequential" if workers == 0 else f"parallel({workers})"
        print(f"{mode:<10} {elapsed:>8.2f}     {speed:>10.1f}       {speedup:>6.2f}x")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python test_multiprocessing.py <test_folder> [num_workers]")
        print()
        print("Example:")
        print("  python test_multiprocessing.py test_photos")
        print("  python test_multiprocessing.py test_photos 8")
        print()
        print("For scaling test:")
        print("  python test_multiprocessing.py test_photos scale")
        sys.exit(1)

    test_folder = sys.argv[1]

    if len(sys.argv) >= 3 and sys.argv[2] == "scale":
        # Scaling test
        test_worker_scaling(test_folder)
    else:
        # Simple comparison
        workers = int(sys.argv[2]) if len(sys.argv) >= 3 else 4
        test_sequential_vs_parallel(test_folder, workers)
