from typing import List, Tuple
import cv2 as cv
import numpy as np

def tenengrad(gray: np.ndarray, ksize: int = 3) -> np.ndarray:
    gx = cv.Sobel(gray, cv.CV_32F, 1, 0, ksize=ksize)
    gy = cv.Sobel(gray, cv.CV_32F, 0, 1, ksize=ksize)
    return cv.magnitude(gx, gy)

def laplacian_var(gray: np.ndarray, ksize: int = 3) -> np.ndarray:
    lap = cv.Laplacian(gray, cv.CV_32F, ksize=ksize)
    return np.abs(lap)

def local_contrast(gray: np.ndarray, ksize: int = 9) -> np.ndarray:
    # local contrast = abs(pixel - local_mean)
    blur = cv.GaussianBlur(gray, (ksize|1, ksize|1), 0)
    return np.abs(gray.astype(np.float32) - blur.astype(np.float32))

def local_std(gray: np.ndarray, ksize: int = 9) -> np.ndarray:
    # approximate local std using mean of squares - square of mean
    g = gray.astype(np.float32)
    mean = cv.GaussianBlur(g, (ksize|1, ksize|1), 0)
    mean_sq = cv.GaussianBlur(g*g, (ksize|1, ksize|1), 0)
    var = np.maximum(mean_sq - mean*mean, 0.0)
    return np.sqrt(var)

def compute_sharpness_map(gray: np.ndarray, ksize:int=3) -> np.ndarray:
    """
    Multi-metric sharpness per-pixel normalized to [0,1].
    """
    tg = tenengrad(gray, ksize=ksize).astype(np.float32)
    lv = laplacian_var(gray, ksize=ksize).astype(np.float32)
    lc = local_contrast(gray, ksize=9).astype(np.float32)
    ls = local_std(gray, ksize=9).astype(np.float32)

    # weighted sum (tuned empirically)
    sharp = 0.4 * tg + 0.3 * lv + 0.2 * lc + 0.1 * ls
        # return raw multi-metric sharpness (no per-frame normalization)
    return sharp


def compute_sharpness_volume(images_bgr: List[np.ndarray], ksize: int = 3) -> List[np.ndarray]:
    grays = [cv.cvtColor(im, cv.COLOR_BGR2GRAY) for im in images_bgr]
    maps = [compute_sharpness_map(g, ksize=ksize) for g in grays]
    return maps

def compute_confidence_maps(sharp_maps: List[np.ndarray]) -> List[np.ndarray]:
    """
    Confidence per-frame per-pixel. High when this frame dominates sharpness locally.
    """
    stack = np.stack(sharp_maps, axis=0)  # (T,H,W)
    # prevent division by zero
    sum_all = np.sum(stack, axis=0) + 1e-8
    # confidence = sharp / mean_sharp_across_frames (relative dominance)
    mean_sharp = np.mean(stack, axis=0) + 1e-8
    conf = []
    for t in range(stack.shape[0]):
        c = stack[t] / mean_sharp
        # clip and normalize per-frame to [0,1]
        c = np.clip(c, 0.0, None)
        mm = np.max(c) + 1e-8
        conf.append(c)
    return conf

def build_focus_depth_map(sharp_maps: List[np.ndarray], smooth: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    stack = np.stack(sharp_maps, axis=0)

    idx = np.argmax(stack, axis=0)
    maxv = np.max(stack, axis=0)

    depth = idx.astype(np.int32)

    if smooth:
        # OpenCV medianBlur obsługuje tylko uint8 / float32
        depth8 = depth.astype(np.uint8)

        # median wygładza drobne fluktuacje indeksów
        depth8 = cv.medianBlur(depth8, 3)

        # bilateral tylko jeśli liczba klatek < 256 (czyli mieści się w uint8)
        try:
            depth8 = cv.bilateralFilter(depth8, 5, 25, 25)
        except Exception:
            pass

        depth = depth8.astype(np.int32)

    return depth, maxv
def masked_sharpness_score(gray: np.ndarray, mask01: np.ndarray, ksize: int = 3, q: int = 85) -> float:
    """
    Compute a robust sharpness score inside mask: return the q-th percentile of
    the multi-metric sharpness values within the mask. Defaults to 85th percentile.
    """
    if int(mask01.sum()) <= 0:
        return 0.0
    sharp = compute_sharpness_map(gray, ksize=ksize)
    vals = sharp[mask01.astype(bool)]
    if vals.size == 0:
        return 0.0
    return float(np.percentile(vals, q))


def argmax_focus_stack(images_bgr: List[np.ndarray], ksize: int = 3) -> np.ndarray:
    """
    Build argmax focus stack using multi-metric sharpness volume.
    """
    if len(images_bgr) == 1:
        return images_bgr[0].copy()
    sharp_maps = compute_sharpness_volume(images_bgr, ksize=ksize)
    stack = np.stack(sharp_maps, axis=0)
    idx = np.argmax(stack, axis=0).astype(np.int32)
    h, w = idx.shape
    out = np.empty_like(images_bgr[0])
    for c in range(3):
        ch = np.stack([im[:, :, c] for im in images_bgr], axis=0)
        out[:, :, c] = ch[idx, np.arange(h)[:, None], np.arange(w)[None, :]]
    return out
