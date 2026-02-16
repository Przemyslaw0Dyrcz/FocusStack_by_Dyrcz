import numpy as np
import cv2 as cv
from typing import List, Optional, Tuple

def _color_for_id(i: int) -> Tuple[int, int, int]:
    rng = np.random.default_rng(12345 + int(i) * 999)
    c = rng.integers(40, 230, size=3).tolist()
    return int(c[0]), int(c[1]), int(c[2])

def overlay_masks(bgr: np.ndarray, masks: List[np.ndarray], ids: Optional[List[int]] = None, alpha: float = 0.45) -> np.ndarray:
    out = bgr.copy()
    if not masks:
        return out
    if ids is None:
        ids = list(range(1, len(masks) + 1))
    h, w = bgr.shape[:2]
    overlay = np.zeros((h, w, 3), np.uint8)
    for m, mid in zip(masks, ids):
        if m is None: continue
        m01 = (m > 0).astype(np.uint8)
        if int(m01.sum()) == 0: continue
        color = _color_for_id(mid)
        overlay[m01 > 0] = color
        cnts, _ = cv.findContours(m01, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cv.drawContours(out, cnts, -1, color, 2)
    out = cv.addWeighted(out, 1.0, overlay, float(alpha), 0.0)
    return out
