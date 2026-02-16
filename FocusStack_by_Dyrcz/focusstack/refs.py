from dataclasses import dataclass
from typing import List, Tuple
import cv2 as cv
import numpy as np
from .segmentation import SegMask, segment_fastsam_single
from .filtering import filter_masks, FilterCfg
from .merging import merge_similar_masks, MergeCfg
from .geometry import bbox_from_mask, iou_xyxy

@dataclass
class RefObj:
    ref_id: int
    mask01: np.ndarray
    bbox_xyxy: Tuple[float, float, float, float]
    area: int

def refs_from_base_stack(
    base_bgr: np.ndarray,
    model,
    device: str,
    imgsz: int,
    conf: float,
    fp16: bool,
    fcfg: FilterCfg,
    mcfg: MergeCfg,
    ref_nms_iou: float,
) -> List[RefObj]:
    h, w = base_bgr.shape[:2]
    frame = segment_fastsam_single(base_bgr, model, device=device, imgsz=imgsz, conf=conf, fp16=fp16)
    frame = filter_masks(frame, h, w, fcfg)
    frame = merge_similar_masks(frame, h, w, mcfg)

    frame.sort(key=lambda x: (x.area * x.score), reverse=True)

    kept: List[SegMask] = []
    for m in frame:
        ok = True
        for k in kept:
            if iou_xyxy(m.bbox_xyxy, k.bbox_xyxy) >= ref_nms_iou:
                ok = False
                break
        if ok:
            kept.append(m)

    refs: List[RefObj] = []
    rid = 1
    for m in kept:
        x1, y1, x2, y2 = bbox_from_mask(m.mask01)
        refs.append(
            RefObj(
                ref_id=rid,
                mask01=m.mask01.astype(np.uint8),
                bbox_xyxy=(float(x1), float(y1), float(x2), float(y2)),
                area=int(m.area),
            )
        )
        rid += 1

    return refs
