import cv2 as cv
from typing import Tuple, List, Optional
import numpy as np
from .tracking import Track
from .sharpness import masked_sharpness_score
from .morphology import morph_clean, mask_hole_fill

def _det_mask_at_t(track: Track, t: int) -> Optional[np.ndarray]:
    for d in track.dets:
        if d.t == t:
            return d.mask01.astype(np.uint8)
    return None

def track_union_mask(
    track: Track,
    h: int,
    w: int,
    open_k: int,
    close_k: int,
    skip_area_ratio: float = 0.90,
    do_hole_fill: bool = False,
) -> np.ndarray:
    m = np.zeros((h, w), np.uint8)
    max_area = int(skip_area_ratio * h * w)

    for d in track.dets:
        if d.area <= 0:
            continue
        if d.area >= max_area:
            continue
        m = cv.bitwise_or(m, d.mask01.astype(np.uint8))

    m = (m > 0).astype(np.uint8)
    m = morph_clean(m, open_k=open_k, close_k=close_k)
    if do_hole_fill:
        m = mask_hole_fill(m)
    return m

def best_frame_for_union_mask(
    track: Track,
    grays: List[np.ndarray],
    union_mask01: np.ndarray,
    ksize: int,
    min_area: int,
) -> Tuple[int, float]:
    best_t = track.dets[0].t
    best_s = -1.0

    for d in track.dets:
        if d.area < min_area:
            continue
        m = cv.bitwise_and(union_mask01.astype(np.uint8), d.mask01.astype(np.uint8))
        if int(m.sum()) < min_area:
            continue
        s = masked_sharpness_score(grays[d.t], m, ksize=ksize)
        if s > best_s:
            best_s = s
            best_t = d.t

    if best_s < 0:
        return best_t, 0.0
    return best_t, best_s

def clamp_union_to_best(
    union_mask01: np.ndarray,
    best_mask01: np.ndarray,
    kernel: int = 11,
    iters: int = 2,
) -> np.ndarray:
    k = max(3, int(kernel) | 1)
    dil = cv.dilate(best_mask01.astype(np.uint8), np.ones((k, k), np.uint8), iterations=int(iters))
    out = cv.bitwise_and(union_mask01.astype(np.uint8), dil)
    return (out > 0).astype(np.uint8)
