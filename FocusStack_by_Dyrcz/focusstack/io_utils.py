from pathlib import Path
from typing import List
import cv2 as cv
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def list_images(folder: Path, pattern: str) -> List[Path]:
    paths = sorted(folder.glob(pattern))
    if not paths:
        paths = []
        for ext in IMG_EXTS:
            paths += list(folder.glob(f"*{ext}"))
        paths = sorted(set(paths))
    if not paths:
        raise FileNotFoundError(f"Brak obrazów w {folder} (pattern={pattern})")
    return paths

def read_images(paths: List[Path], max_imgs: int = 0) -> List[np.ndarray]:
    if max_imgs and max_imgs > 0:
        paths = paths[:max_imgs]
    imgs: List[np.ndarray] = []
    for p in paths:
        im = cv.imread(str(p), cv.IMREAD_COLOR)
        if im is None:
            continue
        imgs.append(im)
    if not imgs:
        raise RuntimeError("Nie udało się wczytać obrazów.")
    h, w = imgs[0].shape[:2]
    out = []
    for im in imgs:
        if im.shape[:2] != (h, w):
            im = cv.resize(im, (w, h), interpolation=cv.INTER_AREA)
        out.append(im)
    return out

def save_png(path: Path, bgr: np.ndarray) -> None:
    ensure_dir(path.parent)
    cv.imwrite(str(path), bgr)

def save_jpg(path: Path, bgr: np.ndarray, q: int = 92) -> None:
    ensure_dir(path.parent)
    cv.imwrite(str(path), bgr, [int(cv.IMWRITE_JPEG_QUALITY), int(q)])

def save_mask(path: Path, mask01: np.ndarray) -> None:
    ensure_dir(path.parent)
    cv.imwrite(str(path), (mask01.astype(np.uint8) * 255))
