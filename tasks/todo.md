# Refine cull scoring, one pipeline, blinks

## 1. Photographic score
- [x] Resolution-stable sharpness (fixed preview, denoise before Laplacian)
- [x] Exposure penalizes real clipping only
- [x] White balance ignores intentional color
- [x] Aesthetic measures tonal separation, not saturation
- [x] Motion blur flags directional smear only
- [x] Composition and eye openness enter the weighted score

## 2. One pipeline and burst hero
- [x] Sequential, parallel, and GPU modes share one stage order
- [x] Saliency and subject box exist before sharpness in every mode
- [x] Duplicate groups keep a single best frame (union-find)
- [x] Wider burst window and mtime fallback

## 3. Faces
- [x] Primary face is the largest, not the first detection
- [x] Eye openness and blink flag on that face
- [x] Burst hero prefers open eyes over a higher technical score

## Review
Sequential and parallel runs share `process_record`. Sharpness is measured on a 512px denoised preview. Bursts keep one hero, and an open eye beats a sharper blink. 27 tests passed.
