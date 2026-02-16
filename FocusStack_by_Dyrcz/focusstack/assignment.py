from typing import List, Dict
from .refs import RefObj
from .segmentation import SegMask
from .geometry import iou_xyxy, mask_iou

def assign_masks_to_refs(
    refs: List[RefObj],
    frame_masks: List[SegMask],
    bbox_gate_iou: float,
    mask_iou_thr: float,
) -> Dict[int, SegMask]:
    if not refs or not frame_masks:
        return {}
    pairs: List[tuple] = []
    for ri, r in enumerate(refs):
        for mi, m in enumerate(frame_masks):
            biou = iou_xyxy(r.bbox_xyxy, m.bbox_xyxy)
            if biou < bbox_gate_iou: continue
            iou = mask_iou(r.mask01, m.mask01)
            if iou >= mask_iou_thr:
                pairs.append((iou, ri, mi))
    pairs.sort(reverse=True, key=lambda x: x[0])
    used_refs = set()
    used_masks = set()
    assigned: Dict[int, SegMask] = {}
    for iou, ri, mi in pairs:
        ref_id = refs[ri].ref_id
        if ref_id in used_refs or mi in used_masks:
            continue
        assigned[ref_id] = frame_masks[mi]
        used_refs.add(ref_id)
        used_masks.add(mi)
    return assigned
