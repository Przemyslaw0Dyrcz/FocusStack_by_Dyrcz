import cv2 as cv
import numpy as np

def hard_paste_then_feather(base: np.ndarray, src: np.ndarray, mask01: np.ndarray, feather: int = 21) -> np.ndarray:
    m = (mask01 > 0).astype(np.uint8)
    if int(m.sum()) == 0:
        return base

    out = base.copy()
    out[m > 0] = src[m > 0]

    k_core = np.ones((5, 5), np.uint8)
    core = cv.erode(m, k_core, iterations=1)
    if int(core.sum()) == 0:
        core = m.copy()

    k = max(3, int(feather) | 1)
    alpha = cv.GaussianBlur(m.astype(np.float32), (k, k), 0)
    alpha = np.clip(alpha, 0.0, 1.0)
    alpha[core > 0] = 1.0

    a = alpha[:, :, None]
    blended = base.astype(np.float32) * (1.0 - a) + out.astype(np.float32) * a
    return np.clip(blended, 0, 255).astype(np.uint8)
