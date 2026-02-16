import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
import cv2 as cv

@dataclass
class SegMask:
    mask01: np.ndarray
    bbox_xyxy: Tuple[float, float, float, float]
    area: int
    score: float

def bbox_from_mask(mask01: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask01 > 0)
    if xs.size == 0:
        return (0, 0, 0, 0)
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    return (x1, y1, x2, y2)

def segment_fastsam_single(bgr: np.ndarray, model, device: str, imgsz: int, conf: float, fp16: bool) -> List[SegMask]:
    kwargs = dict(source=bgr, device=device, imgsz=imgsz, conf=conf, verbose=False)
    if fp16 and device == "cuda":
        kwargs["half"] = True

    res = model.predict(**kwargs)[0]
    h, w = bgr.shape[:2]
    frame: List[SegMask] = []

    masks = None
    if getattr(res, "masks", None) is not None and getattr(res.masks, "data", None) is not None:
        masks = res.masks.data
    if masks is None:
        return frame

    masks_np = masks.detach().cpu().numpy()

    boxes = None
    confs = None
    if getattr(res, "boxes", None) is not None and getattr(res.boxes, "xyxy", None) is not None:
        boxes = res.boxes.xyxy.detach().cpu().numpy()
        confs = res.boxes.conf.detach().cpu().numpy() if getattr(res.boxes, "conf", None) is not None else None

    for i in range(masks_np.shape[0]):
        m = masks_np[i]
        if m.shape[:2] != (h, w):
            m = cv.resize(m.astype(np.float32), (w, h), interpolation=cv.INTER_NEAREST)
        m01 = (m > 0.5).astype(np.uint8)
        area = int(m01.sum())
        if area <= 0:
            continue

        if boxes is not None and i < boxes.shape[0]:
            x1, y1, x2, y2 = boxes[i].tolist()
        else:
            x1, y1, x2, y2 = bbox_from_mask(m01)

        score = float(confs[i]) if confs is not None and i < confs.shape[0] else 1.0
        frame.append(SegMask(mask01=m01, bbox_xyxy=(float(x1), float(y1), float(x2), float(y2)), area=area, score=score))

    return frame
