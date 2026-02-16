from typing import List
import cv2 as cv
import numpy as np

def align_ecc_affine(
    images_bgr: List[np.ndarray],
    iters: int = 80,
    eps: float = 1e-6,
    ecc_scale: float = 0.5,
) -> List[np.ndarray]:
    if len(images_bgr) <= 1:
        return [images_bgr[0].copy()]

    ref = images_bgr[0]
    h, w = ref.shape[:2]

    def to_gray(im: np.ndarray) -> np.ndarray:
        return cv.cvtColor(im, cv.COLOR_BGR2GRAY)

    ref_g = to_gray(ref)

    if 0.05 < ecc_scale < 1.0:
        ref_s = cv.resize(ref_g, (int(w * ecc_scale), int(h * ecc_scale)), interpolation=cv.INTER_AREA)
    else:
        ref_s = ref_g

    aligned = [ref.copy()]
    warp_mode = cv.MOTION_AFFINE
    criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, int(iters), float(eps))

    for i in range(1, len(images_bgr)):
        im = images_bgr[i]
        im_g = to_gray(im)

        if 0.05 < ecc_scale < 1.0:
            im_s = cv.resize(im_g, (ref_s.shape[1], ref_s.shape[0]), interpolation=cv.INTER_AREA)
        else:
            im_s = im_g

        warp = np.eye(2, 3, dtype=np.float32)
        try:
            _cc, warp_s = cv.findTransformECC(ref_s, im_s, warp, warp_mode, criteria, None, 1)

            if 0.05 < ecc_scale < 1.0:
                warp_full = warp_s.copy()
                warp_full[0, 2] = warp_s[0, 2] / ecc_scale
                warp_full[1, 2] = warp_s[1, 2] / ecc_scale
            else:
                warp_full = warp_s

            im_al = cv.warpAffine(
                im,
                warp_full,
                (w, h),
                flags=cv.INTER_LINEAR + cv.WARP_INVERSE_MAP,
                borderMode=cv.BORDER_REFLECT,
            )
            aligned.append(im_al)
        except cv.error:
            aligned.append(im.copy())

    return aligned
