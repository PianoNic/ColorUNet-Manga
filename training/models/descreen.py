"""Descreen: smooth halftone / screentone regions of a grayscale page, leave line art and text alone.

Used on the lightness channel, so toned skies get smooth color while ink stays exactly as drawn.

1. Dots are small connected blobs: dark dots on light tone and light dots on dark tone, each no
   bigger than a few pixels (scaled to the page resolution). Letters and line strokes are larger.
2. A tone region is where many such dot blobs sit close together.
3. Inside tone regions, the page is replaced by a blur that ignores ink pixels (normalized
   convolution), so lines don't bleed gray halos into the smoothed tone.
4. Large dark components (lines, text, solid blacks) are never touched.
"""

import cv2
import numpy as np


def _small_blobs(binary: np.ndarray, max_d: int):
    """Components no larger than max_d px each way: (labels, keep flags, centroids, centroid map)."""
    n, labels, stats, cents = cv2.connectedComponentsWithStats(binary, connectivity=8)
    w, h, area = stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT], stats[:, cv2.CC_STAT_AREA]
    keep = (w <= max_d) & (h <= max_d) & (area >= 2)
    keep[0] = False
    c = np.round(cents).astype(int)
    c[:, 0] = c[:, 0].clip(0, binary.shape[1] - 1)
    c[:, 1] = c[:, 1].clip(0, binary.shape[0] - 1)
    centers = np.zeros(binary.shape, np.float32)
    centers[c[keep, 1], c[keep, 0]] = 1
    return labels, keep, c, centers


def tone_mask(gray: np.ndarray, dot_frac: float = 0.0045, min_dots: int = 10):
    """(mask, ink, max_d): mask is float32 HxW in [0,1], 1 inside screentone, 0 on paper/ink/text."""
    H, W = gray.shape
    max_d = max(3, int(round(dot_frac * max(H, W))))  # biggest dot, px (~12 px at 2800)
    win = 6 * max_d + 1  # neighbourhood to count dots in

    local = cv2.GaussianBlur(gray, (0, 0), max_d)  # adaptive threshold handles light & dark tones
    dark = (gray.astype(np.int16) < local.astype(np.int16) - 12).astype(np.uint8)
    light = (gray.astype(np.int16) > local.astype(np.int16) + 12).astype(np.uint8)
    d_lab, d_keep, d_c, d_centers = _small_blobs(dark, max_d)
    l_lab, l_keep, l_c, l_centers = _small_blobs(light, max_d)

    count = cv2.boxFilter(d_centers + l_centers, -1, (win, win), normalize=False)
    # a small dark blob is a tone dot only if many others sit around it AND it is not much bigger
    # than its neighbours: ellipsis dots, periods and hatching strokes stay ink even inside a tone,
    # while darker tones (bigger dots everywhere) still count as tone
    areas = np.bincount(d_lab.ravel(), minlength=len(d_keep)).astype(np.float32)
    area_map = np.zeros(gray.shape, np.float32)
    area_map[d_c[d_keep, 1], d_c[d_keep, 0]] = areas[d_keep]
    n_dark = cv2.boxFilter(d_centers, -1, (win, win), normalize=False)
    mean_area = cv2.boxFilter(area_map, -1, (win, win), normalize=False) / np.maximum(n_dark, 1)
    at = (d_c[:, 1], d_c[:, 0])
    d_keep &= (count[at] >= min_dots) & (areas <= 2.5 * mean_area[at])
    dark_dots = d_keep[d_lab]

    count = cv2.GaussianBlur(count, (0, 0), 2 * max_d)
    # soft decision: partial tone (fading gradients, line tones breaking into dots) blends in gradually
    # instead of switching on and off in patches
    t = np.clip((count - 0.5 * min_dots) / min_dots, 0, 1)
    region = t * t * (3 - 2 * t)

    # ink = dark pixels that are not tone dots (lines, letters, solid black), slightly grown
    ink = ((gray < 110) & ~dark_dots).astype(np.uint8)
    ink = cv2.dilate(ink, np.ones((3, 3), np.uint8))
    mask = region.astype(np.float32)
    mask[ink > 0] = 0
    return mask, ink, max_d


def descreen(gray: np.ndarray, **kw) -> np.ndarray:
    """uint8 HxW page -> uint8 HxW page with screentone regions smoothed."""
    mask, ink, max_d = tone_mask(gray, **kw)
    if mask.max() == 0:
        return gray
    g = gray.astype(np.float32)
    valid = (1 - ink).astype(np.float32)
    sigma = 0.8 * max_d
    smooth = cv2.GaussianBlur(g * valid, (0, 0), sigma) / np.maximum(cv2.GaussianBlur(valid, (0, 0), sigma), 1e-3)
    out = g * (1 - mask) + smooth * mask
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)
