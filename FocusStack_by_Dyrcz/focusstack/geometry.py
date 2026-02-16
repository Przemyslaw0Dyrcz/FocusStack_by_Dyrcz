import numpy as np
from typing import Tuple

def bbox_from_mask(mask01: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask01 > 0)
    if xs.size == 0:
        return (0, 0, 0, 0)
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    return (x1, y1, x2, y2)

def iou_xyxy(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return float(inter / (area_a + area_b - inter + 1e-6))

def centroid_from_mask(mask01: np.ndarray) -> Tuple[float, float]:
    ys, xs = np.where(mask01 > 0)
    if xs.size == 0:
        return (0.0, 0.0)
    return (float(xs.mean()), float(ys.mean()))

def mask_iou(a01: np.ndarray, b01: np.ndarray) -> float:
    a = (a01 > 0)
    b = (b01 > 0)
    inter = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum()) + 1e-6
    return inter / union

def _bbox_aspect(b: Tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = b
    ww = max(1.0, float(x2 - x1))
    hh = max(1.0, float(y2 - y1))
    return ww / hh
