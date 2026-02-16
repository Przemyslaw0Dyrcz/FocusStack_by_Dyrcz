from typing import List
import cv2 as cv
import numpy as np

def tenengrad(gray: np.ndarray, ksize: int = 3) -> np.ndarray:
    gx = cv.Sobel(gray, cv.CV_32F, 1, 0, ksize=ksize)
    gy = cv.Sobel(gray, cv.CV_32F, 0, 1, ksize=ksize)
    return cv.magnitude(gx, gy)

def masked_sharpness_score(gray: np.ndarray, mask01: np.ndarray, ksize: int = 3) -> float:
    if int(mask01.sum()) <= 0:
        return 0.0
    m = mask01.astype(np.float32)
    tg = tenengrad(gray, ksize=ksize)
    return float((tg * m).sum() / (m.sum() + 1e-6))

def argmax_focus_stack(images_bgr: List[np.ndarray], ksize: int = 3) -> np.ndarray:
    if len(images_bgr) == 1:
        return images_bgr[0].copy()
    grays = [cv.cvtColor(im, cv.COLOR_BGR2GRAY) for im in images_bgr]
    maps = [tenengrad(g, ksize=ksize) for g in grays]
    stack = np.stack(maps, axis=0)
    idx = np.argmax(stack, axis=0).astype(np.int32)
    h, w = idx.shape
    out = np.empty_like(images_bgr[0])
    for c in range(3):
        ch = np.stack([im[:, :, c] for im in images_bgr], axis=0)
        out[:, :, c] = ch[idx, np.arange(h)[:, None], np.arange(w)[None, :]]
    return out
