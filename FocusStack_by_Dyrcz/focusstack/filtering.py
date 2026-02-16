from dataclasses import dataclass
from typing import List
import numpy as np
from .geometry import iou_xyxy
from .segmentation import SegMask

@dataclass
class FilterCfg:
    min_area: int = 200
    max_area_ratio: float = 0.98
    nms_iou: float = 0.60
    max_masks: int = 120

def filter_masks(frame_masks: List[SegMask], h: int, w: int, cfg: FilterCfg) -> List[SegMask]:
    filtered: List[SegMask] = []
    max_area = int(cfg.max_area_ratio * h * w)
    for m in frame_masks:
        if m.area < cfg.min_area: continue
        if m.area > max_area: continue
        filtered.append(m)
    filtered.sort(key=lambda x: (x.area * x.score), reverse=True)
    kept: List[SegMask] = []
    for m in filtered:
        ok = True
        for k in kept:
            if iou_xyxy(m.bbox_xyxy, k.bbox_xyxy) >= cfg.nms_iou:
                ok = False; break
        if ok:
            kept.append(m)
        if len(kept) >= cfg.max_masks:
            break
    return kept
