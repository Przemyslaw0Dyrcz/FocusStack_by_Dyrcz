from dataclasses import dataclass
from typing import List
import numpy as np
from .geometry import iou_xyxy
from .segmentation import SegMask
from .utils import is_inside

@dataclass
class FilterCfg:
    min_area: int = 200
    max_area_ratio: float = 0.98
    nms_iou: float = 0.60
    max_masks: int = 120

def filter_masks(frame_masks: List[SegMask], h: int, w: int, cfg: FilterCfg) -> List[SegMask]:
    """
    Filtruje maski:
    - usuwa maski zagnieżdżone w innych
    - usuwa za małe i za duże
    - robi NMS po bboxach
    """

    # pracujemy na lokalnej liście
    masks = list(frame_masks)

    # --- usuwanie masek zagnieżdżonych ---
    keep = [True] * len(masks)

    for i in range(len(masks)):
        for j in range(len(masks)):
            if i != j and keep[i] and keep[j]:
                if is_inside(masks[i].mask01, masks[j].mask01):
                    keep[i] = False

    masks = [m for k, m in zip(keep, masks) if k]

    # --- filtr powierzchni ---
    filtered: List[SegMask] = []
    max_area = cfg.max_area_ratio * h * w

    for m in masks:
        if m.area < cfg.min_area:
            continue
        if m.area > max_area:
            continue
        filtered.append(m)

    # --- sort po quality ---
    filtered.sort(key=lambda x: (x.area * getattr(x, "score", 1.0)), reverse=True)

    # --- NMS ---
    kept: List[SegMask] = []

    for m in filtered:
        ok = True
        for k in kept:
            if iou_xyxy(m.bbox_xyxy, k.bbox_xyxy) >= cfg.nms_iou:
                ok = False
                break
        if ok:
            kept.append(m)

        if len(kept) >= cfg.max_masks:
            break

    return kept