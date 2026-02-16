import cv2 as cv
from typing import Tuple, List
import numpy as np
from .tracking import Track
from .sharpness import masked_sharpness_score
from .morphology import morph_clean, mask_hole_fill

def best_frame_for_track(track: Track, grays: List[np.ndarray], ksize: int, min_area: int) -> Tuple[int, float]:
    best_t = track.dets[0].t
    best_s = -1.0
    for d in track.dets:
        if d.area < min_area: continue
        s = masked_sharpness_score(grays[d.t], d.mask01, ksize=ksize)
        if s > best_s:
            best_s = s
            best_t = d.t
    if best_s < 0:
        return best_t, 0.0
    return best_t, best_s

def track_mask_core_and_fill(
    track: Track,
    h: int,
    w: int,
    best_t: int,
    core_thr: float,
    fill_thr: float,
    max_fill_dist: int,
    open_k: int,
    close_k: int,
) -> np.ndarray:
    anchor = None
    for d in track.dets:
        if d.t == best_t:
            anchor = d.mask01.astype(np.uint8)
            break
    if anchor is None:
        anchor = track.dets[0].mask01.astype(np.uint8)

    stack = np.stack([d.mask01.astype(np.uint8) for d in track.dets], axis=0)
    votes = stack.sum(axis=0).astype(np.float32)
    n = float(stack.shape[0])

    core = (votes >= core_thr * n).astype(np.uint8)
    fill = (votes >= fill_thr * n).astype(np.uint8)

    dist = cv.distanceTransform((1 - anchor).astype(np.uint8), cv.DIST_L2, 3)
    near_anchor = (dist <= float(max_fill_dist)).astype(np.uint8)

    m = (core | (fill & near_anchor) | anchor).astype(np.uint8)
    m = morph_clean(m, open_k=open_k, close_k=close_k)
    m = mask_hole_fill(m)
    return m
