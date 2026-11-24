"""Training-input augmentations that make converted color pages look more like drawn manga.

white_paper: in real manga, skin, light hair and light clothing are bare white paper; in pairs made
by converting a color page to gray they are light gray. Pushing light grays of the INPUT to pure
white (the color target is unchanged) teaches the model that white areas inside a face or a hair
outline still get color. Line art and darker tones are untouched (smooth knee, no hard edge).
"""

import numpy as np


def white_paper(gray: np.ndarray, rng, threshold=(0.70, 0.88), knee=0.08) -> np.ndarray:
    """gray float32 HxW in [0,1] -> same, with tones above ~threshold lifted to paper white."""
    t = rng.uniform(*threshold)
    x = np.clip((gray - (t - knee)) / knee, 0, 1)
    k = x * x * (3 - 2 * x)  # smoothstep from t-knee to t
    return (gray * (1 - k) + k).astype(np.float32)
