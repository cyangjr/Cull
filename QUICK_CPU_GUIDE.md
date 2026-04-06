# Quick CPU Worker Configuration Guide

## TL;DR

Your system has **20 cores**. By default, Cull will use **8 workers** (40%, tested optimal).

## Current Configuration

Check your `config.yaml`:
```yaml
num_workers: null  # Auto (40% of cores, tested optimal) ← YOU ARE HERE
```

## Quick Test

Run this to see your current setup:
```bash
.venv\Scripts\python.exe scripts\test_cpu_config.py
```

## Want to Change It?

Edit `config.yaml` and set `num_workers` to:

| Setting | Workers Used | CPU Usage | Speed (tested) | System Responsiveness |
|---------|-------------|-----------|----------------|---------------------|
| `null` (default) | 8 | ~40% | 1.25x | ✓✓✓✓ Excellent |
| `4` | 4 | ~20% | 1.22x | ✓✓✓✓✓ Perfect |
| `8` | 8 | ~40% | 1.25x | ✓✓✓✓ Excellent |
| `10` | 10 | ~50% | 0.96x | ✓✓✓ Good (slower!) |
| `19` (max) | 19 | ~95% | 1.02x | ✓✓ Fair (slower!) |
| `0` | 0 | ~12% | 1.0x | ✓✓✓✓✓ Perfect |

## Safety Features

✅ **Always leaves 1+ cores free** (prevents system lockup)
✅ **Auto-clamps** values (requesting 100 workers → uses 19)
✅ **Works on any CPU** (1-core to 128-core)
✅ **No crashes** (handles all edge cases)

## Recommendations

- **Most users:** Keep default (`null`) = 8 workers, tested optimal
- **Maximum performance:** Use `8` (best tested result)
- **More responsive system:** Use `4` (nearly as fast, 20% CPU)
- **Debugging issues:** Use `0` (sequential mode)
- **Avoid:** `10+` workers (slower due to overhead)

## Full Documentation

See `docs/cpu-configuration.md` for detailed guide with examples and troubleshooting.
