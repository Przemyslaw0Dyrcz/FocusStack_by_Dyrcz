import cv2 as cv
import numpy as np

def morph_clean(mask01: np.ndarray, open_k: int = 3, close_k: int = 7) -> np.ndarray:
    m = mask01.astype(np.uint8)
    if open_k > 0:
        k = np.ones((open_k, open_k), np.uint8)
        m = cv.morphologyEx(m, cv.MORPH_OPEN, k, iterations=1)
    if close_k > 0:
        k = np.ones((close_k, close_k), np.uint8)
        m = cv.morphologyEx(m, cv.MORPH_CLOSE, k, iterations=1)
    return m

def mask_hole_fill(mask01: np.ndarray) -> np.ndarray:
    m = (mask01.astype(np.uint8) * 255)
    h, w = m.shape
    inv = cv.bitwise_not(m)  # tło=255, obiekt=0, dziury=255
    ff = inv.copy()
    tmp = np.zeros((h + 2, w + 2), np.uint8)

    seed = None
    # znajdź punkt tła na brzegu
    for x in range(w):
        if ff[0, x] == 255:
            seed = (x, 0); break
        if ff[h - 1, x] == 255:
            seed = (x, h - 1); break
    if seed is None:
        for y in range(h):
            if ff[y, 0] == 255:
                seed = (0, y); break
            if ff[y, w - 1] == 255:
                seed = (w - 1, y); break
    if seed is None:
        return (m > 0).astype(np.uint8)

    # wypełnij tło (flood fill) i połącz dziury
    cv.floodFill(ff, tmp, seedPoint=seed, newVal=0)
    holes = ff
    filled = cv.bitwise_or(m, holes)
    return (filled > 0).astype(np.uint8)
