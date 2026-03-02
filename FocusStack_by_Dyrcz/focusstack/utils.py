import numpy as np
import cv2 as cv


def is_inside(mask_a: np.ndarray, mask_b: np.ndarray) -> bool:
    """Check if mask_a is fully inside mask_b"""
    # ensure boolean arrays
    a = (mask_a > 0)
    b = (mask_b > 0)
    return np.all(np.logical_or(~a, b))


def pad_mask(mask: np.ndarray, pad: int = 3) -> np.ndarray:
    """Dilate mask by pad pixels on all sides"""
    k = np.ones((pad*2+1, pad*2+1), np.uint8)
    return cv.dilate(mask.astype(np.uint8), k)