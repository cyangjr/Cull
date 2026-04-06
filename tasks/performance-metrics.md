# CPU Multiprocessing Performance Metrics

**Test Date:** 2026-04-06
**System:** 20-core CPU
**Dataset:** 154 test images
**Test Type:** Worker scaling analysis

---

## Results

| Workers | Time (s) | Speed (img/s) | Speedup |
|---------|----------|---------------|---------|
| 0 (sequential) | 112.85 | 1.4 | 1.00x (baseline) |
| 1 | 194.63 | 0.8 | 0.58x |
| 2 | 106.74 | 1.4 | 1.06x |
| 4 | 92.77 | 1.7 | 1.22x |
| 8 | 90.30 | 1.7 | **1.25x** |
| 10 (default) | 117.58 | 1.3 | 0.96x |
| 19 (max) | 111.16 | 1.4 | 1.02x |

---

## Key Findings

### Optimal Configuration
- **Best performance:** 8 workers (1.25x speedup)
- **Time improvement:** 22.55s faster (112.85s → 90.30s)
- **Percentage improvement:** 20% faster than sequential

### Performance Characteristics
- **1 worker:** Slower than sequential due to multiprocessing overhead
- **2-4 workers:** Progressive improvement (6-22% speedup)
- **8 workers:** Peak performance (25% speedup)
- **10+ workers:** Diminishing returns, performance degrades

### Unexpected Result
Default configuration (10 workers = 50% of 20 cores) actually underperformed compared to 8 workers. This suggests:
1. Optimal worker count is closer to 40% of cores (8/20)
2. Process spawning/coordination overhead increases beyond 8 workers
3. Memory bandwidth or I/O contention may be limiting factor

---

## Recommendations

### For This System (20 cores)
```yaml
num_workers: 8  # Optimal for 20-core system
```

### General Guidance
- **Small datasets (<50 images):** Use sequential (overhead not worth it)
- **Medium datasets (50-200 images):** Use ~40% of cores
- **Large datasets (500+ images):** Test scaling, may benefit from higher worker counts

---

## Comparison to Expectations

### Expected vs Actual
- **Expected speedup:** 3-6x (based on CPU-bound workload analysis)
- **Actual speedup:** 1.25x (25% improvement)
- **Discrepancy:** Significant - suggests bottleneck is NOT pure CPU

### Possible Bottlenecks
1. **I/O bound:** Image loading from disk may be the limiting factor
2. **Memory bandwidth:** 154 images × workers may saturate RAM bandwidth
3. **Process overhead:** ProcessPoolExecutor spawning/serialization costs
4. **Small dataset:** 154 images may not be large enough to amortize overhead
5. **Sequential stages:** Deduplication and ranking are still single-threaded

---

## Next Steps

1. **Update config.yaml:** Change default from `null` (10 workers) to `8`
2. **Test larger dataset:** Run with 500+ images to verify scaling
3. **Profile I/O:** Measure time spent in image loading vs scoring
4. **Consider hybrid approach:** ThreadPoolExecutor for I/O, ProcessPoolExecutor for CPU
5. **Benchmark disk speed:** May need SSD optimization or prefetching
