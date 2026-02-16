from dataclasses import dataclass
from typing import List, Dict
import numpy as np
import cv2 as cv
from .geometry import iou_xyxy, centroid_from_mask, _bbox_aspect
from .segmentation import SegMask
from .geometry import bbox_from_mask

@dataclass
class MergeCfg:
    merge_dist_px: float = 45.0
    merge_area_ratio_max: float = 3.0
    merge_aspect_diff: float = 0.9
    merge_bbox_iou_min: float = 0.02
    merge_max_iters: int = 3
    dup_iou_thr: float = 0.80
    dup_area_sim: float = 0.80

def merge_similar_masks(frame: List[SegMask], h: int, w: int, cfg: MergeCfg) -> List[SegMask]:
    if len(frame) <= 1:
        return frame

    def is_duplicate(a: SegMask, b: SegMask) -> bool:
        biou = iou_xyxy(a.bbox_xyxy, b.bbox_xyxy)
        if biou < cfg.dup_iou_thr:
            return False
        sim = min(a.area, b.area) / (max(a.area, b.area) + 1e-6)
        return sim >= cfg.dup_area_sim

    def can_merge(a: SegMask, b: SegMask) -> bool:
        if is_duplicate(a, b):
            return False

        ax, ay = centroid_from_mask(a.mask01)
        bx, by = centroid_from_mask(b.mask01)
        dist = float(np.hypot(ax - bx, ay - by))
        if dist > cfg.merge_dist_px:
            return False

        ar = max(a.area, b.area) / (min(a.area, b.area) + 1e-6)
        if ar > cfg.merge_area_ratio_max:
            return False

        aa = _bbox_aspect(a.bbox_xyxy)
        bb = _bbox_aspect(b.bbox_xyxy)
        if abs(aa - bb) > cfg.merge_aspect_diff:
            return False

        biou = iou_xyxy(a.bbox_xyxy, b.bbox_xyxy)
        if biou < cfg.merge_bbox_iou_min and dist > (0.6 * cfg.merge_dist_px):
            return False

        return True

    cur = frame
    for _ in range(cfg.merge_max_iters):
        n = len(cur)
        if n <= 1:
            break

        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(i: int, j: int) -> None:
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[rj] = ri

        for i in range(n):
            for j in range(i + 1, n):
                if can_merge(cur[i], cur[j]):
                    union(i, j)

        groups: Dict[int, List[int]] = {}
        for i in range(n):
            r = find(i)
            groups.setdefault(r, []).append(i)

        if all(len(g) == 1 for g in groups.values()):
            break

        merged: List[SegMask] = []
        for idxs in groups.values():
            if len(idxs) == 1:
                merged.append(cur[idxs[0]])
                continue

            m_union = np.zeros((h, w), np.uint8)
            score = 0.0
            for k in idxs:
                m_union = cv.bitwise_or(m_union, cur[k].mask01.astype(np.uint8))
                score = max(score, float(cur[k].score))

            area = int(m_union.sum())
            x1, y1, x2, y2 = bbox_from_mask(m_union)
            merged.append(SegMask(mask01=m_union, bbox_xyxy=(float(x1), float(y1), float(x2), float(y2)), area=area, score=score))

        merged.sort(key=lambda x: (x.area * x.score), reverse=True)
        cur = merged

    return cur
