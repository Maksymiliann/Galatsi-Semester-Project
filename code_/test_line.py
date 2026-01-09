import cv2
import numpy as np

"""
Post-processes a binary parking zones mask to produce uniform-width rectangular zones.
Steps:
- Thins the mask to a 1-pixel skeleton
- Cleans and reconnects small gaps
- Fits a minimum-area rectangle to each skeleton segment
- Redraws each segment with a fixed width and filtered length
Result is a clean, standardized rectangular mask for downstream analysis.
"""


# -----------------
# PATHS
# -----------------
IN_PATH  = r"C:/Users/makss/Git/Galatsi-Semester-Project/Results/parking_detection/dwell_mult_4/test4_thr_0.85/parking_dwell_state_MULTI_REG_zones_mask.png"
OUT_PATH = r"C:/Users/makss/Git/Galatsi-Semester-Project/Results/parking_detection/dwell_mult_4/test4_thr_0.85/mask_uniform_rects.png"

# -----------------
# PARAMS
# -----------------
# Reconnexion après thinning (petit, sinon ça fusionne)
CLOSE_K = (1, 1)
CLOSE_ITERS = 1

# Filtrage bruit (sur squelette)
MIN_AREA = 8

# Largeur fixe finale (en pixels) => adapte à ton use-case
FIXED_WIDTH = 12  # <-- ex: t12

# Longueur minimale (évite petits bouts)
MIN_LEN = 4

# -----------------
# THINNING (Zhang-Suen via ximgproc si dispo)
# -----------------
def thinning(binary_255):
    """
    binary_255: uint8 {0,255}
    returns: skeleton uint8 {0,255}
    """
    # OpenCV contrib: cv2.ximgproc.thinning
    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
        skel = cv2.ximgproc.thinning(binary_255, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
        return skel

    # Fallback simple: morphological skeleton (moins propre mais ok)
    img = (binary_255 > 0).astype(np.uint8) * 255
    skel = np.zeros_like(img)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while True:
        opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
        temp = cv2.subtract(img, opened)
        skel = cv2.bitwise_or(skel, temp)
        img = cv2.erode(img, element)

        if cv2.countNonZero(img) == 0:
            break
    return skel

def load_binary(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    assert img is not None, f"Impossible de lire {path}"
    _, b = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
    return b

def main():
    bw = load_binary(IN_PATH)

    # 1) Thinning => largeur uniforme (1px)
    skel = thinning(bw)

    # 2) Reconnecter un peu (optionnel)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, CLOSE_K)
    skel2 = cv2.morphologyEx(skel, cv2.MORPH_CLOSE, k, iterations=CLOSE_ITERS)

    # 3) Contours sur squelette
    contours, _ = cv2.findContours(skel2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    out = np.zeros_like(bw)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA:
            continue

        rect = cv2.minAreaRect(cnt)  # ((cx,cy),(w,h),angle)
        (cx, cy), (w, h), ang = rect

        # on veut une longueur stable: la plus grande dimension du rect
        L = max(w, h)
        if L < MIN_LEN:
            continue

        # on force la largeur à FIXED_WIDTH
        forced = ((cx, cy), (L, FIXED_WIDTH), ang) if w >= h else ((cx, cy), (FIXED_WIDTH, L), ang)

        box = cv2.boxPoints(forced)
        box = np.int32(box)

        cv2.drawContours(out, [box], 0, 255, thickness=-1)

    cv2.imwrite(OUT_PATH, out)
    print("Saved:", OUT_PATH)
    print("Contours:", len(contours))

if __name__ == "__main__":
    main()
