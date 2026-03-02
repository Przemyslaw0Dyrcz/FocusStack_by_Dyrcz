import numpy as np
import cv2 as cv
from typing import List, Dict, Optional


def build_masks_by_category(packs, total_frames: int, h: int, w: int) -> Dict[str, Dict[int, np.ndarray]]:
    categories = {'static': {}, 'semi': {}, 'dynamic': {}}
    for cat in categories:
        for t in range(total_frames):
            categories[cat][t] = np.zeros((h, w), dtype=np.uint8)

    for p in packs:
        cat = getattr(p, 'category', None)
        if cat not in categories:
            continue
        for d in p.track.dets:
            categories[cat][d.t] = cv.bitwise_or(
                categories[cat][d.t],
                (d.mask01 > 0).astype(np.uint8)
            )
    return categories


def build_dynamic_masks(packs, total_frames: int, h: int, w: int) -> Dict[int, np.ndarray]:
    dynamic_masks = {t: np.zeros((h, w), dtype=np.uint8) for t in range(total_frames)}
    for p in packs:
        if getattr(p, 'category', None) != "dynamic":
            continue
        for d in p.track.dets:
            dynamic_masks[d.t] = cv.bitwise_or(
                dynamic_masks[d.t],
                (d.mask01 > 0).astype(np.uint8)
            )
    return dynamic_masks



def patch_fill_background(imgs: List[np.ndarray],
                          holes_mask: np.ndarray,
                          masks_by_category: Dict[str, Dict[int, np.ndarray]],
                          patch_size: int = 9,
                          mask_expand: int = 8,
                          debug: bool = False,
                          base: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Region-wise background reconstruction using multi-metric sharpness and confidence.
    This replaces the old patch-based filler: for each hole (connected component) we
    pick top candidate frames and either take single best frame or weighted-blend the top 2.
    """
    import numpy as _np
    H, W = holes_mask.shape
    out = np.zeros_like(imgs[0])

    total_frames = len(imgs)

    # build dynamic mask per-frame (if provided)
    dynamic_masks = {}
    for t in range(total_frames):
        dynamic_masks[t] = masks_by_category.get('dynamic', {}).get(t, np.zeros((H, W), dtype=_np.uint8))

    # compute sharpness and confidence volumes
    from .sharpness import compute_sharpness_volume, compute_confidence_maps, build_focus_depth_map
    sharp_maps = compute_sharpness_volume(imgs, ksize=3)
    conf_maps = compute_confidence_maps(sharp_maps)
    depth_map, maxsharp = build_focus_depth_map(sharp_maps, smooth=True)

    # connected components on holes_mask
    num_labels, labels = cv.connectedComponents(holes_mask.astype(_np.uint8), connectivity=8)
    if debug:
        print(f"[patch_fill_background] holes: {num_labels-1} regions")

    def best_frames_for_region(region_mask):
        # scores for each frame based on sharpness * confidence within region
        scores = []
        for t in range(total_frames):
            if dynamic_masks.get(t) is not None and (dynamic_masks[t] & region_mask).sum() > 0:
                # skip frames where dynamic object overlaps region
                scores.append(-1e9); continue
            vals = sharp_maps[t][region_mask.astype(bool)].astype(_np.float32) * conf_maps[t][region_mask.astype(bool)].astype(_np.float32)
            
            if vals.size == 0:
                s = -1e9
            else:
                s = float(_np.percentile(vals, 80))
            scores.append(float(s))
        order = _np.argsort(_np.array(scores))[::-1]
        return order, scores

    def blend_region_from_frames(frame_ids, region_mask):
        # weighted blend using sharpness * confidence as per-pixel weights
        acc = _np.zeros_like(out, dtype=_np.float32)
        wsum = _np.zeros((H, W), dtype=_np.float32) + 1e-8
        for fid in frame_ids:
            weight = sharp_maps[fid].astype(_np.float32) * conf_maps[fid].astype(_np.float32)
            # zero outside region and where dynamic mask overlaps
            w = weight * region_mask.astype(_np.float32)
            img = imgs[fid].astype(_np.float32)
            for c in range(3):
                acc[:, :, c] += img[:, :, c] * w
            wsum += w
        out_region = (acc / wsum[:, :, None])
        out_region = _np.clip(out_region, 0, 255).astype(_np.uint8)
        return out_region

    # process each region (skip label 0)
    for label in range(1, num_labels):
        region_mask = (labels == label).astype(_np.uint8)
        # skip tiny regions
        if region_mask.sum() < 4:
            if base is not None:
                out[region_mask.astype(bool)] = base[region_mask.astype(bool)]
            continue
        order, scores = best_frames_for_region(region_mask)
        if debug:
            print(f"[BG] region {label}: top frames {order[:3]} scores {[scores[i] for i in order[:3]]}")

        # choose top candidate(s)
        best0 = int(order[0])
        if scores[best0] < 0:
            # no valid frames -> fallback to base or black
            if base is not None:
                out[region_mask.astype(bool)] = base[region_mask.astype(bool)]
            continue
        # ALWAYS single best frame per region (no blending)
        out[region_mask.astype(bool)] = imgs[best0][region_mask.astype(bool)]

    # fill any remaining pixels from base if provided
    if base is not None:
        missing = (holes_mask > 0) & (out.sum(axis=2) == 0)
        out[missing] = base[missing]

    return out
