import cv2
import numpy as np
import os

# -------- CONFIG ---------
IMG_PATH = r"C:/Users/makss/Git/Galatsi-Semester-Project/Results/parking_detection/dwell_mult_4/test5_thr_0.85/parking_dwell_state_MULTI_REG_zones_mask.png"
OUT_DIR  = r"C:/Users/makss/Git/Galatsi-Semester-Project/Results/parking_detection/zones4/test3/"

# kernel sizes to try for morphological closing
KERNEL_SIZES = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51]   # you can adjust these
# --------------------------


def color_connected_components(binary_img, name_prefix):
    """
    Takes a binary image (0/255), runs connected components,
    and saves a color-labeled image.
    """
    h, w = binary_img.shape

    # connected components needs 0/1, so convert
    _, bin01 = cv2.threshold(binary_img, 0, 1, cv2.THRESH_BINARY)

    num_labels, labels = cv2.connectedComponents(bin01.astype(np.uint8))
    print(f"{name_prefix}: found {num_labels-1} components (excluding background)")

    # build color image
    out = np.zeros((h, w, 3), dtype=np.uint8)
    rng = np.random.default_rng(0)
    colors = rng.integers(0, 255, size=(num_labels, 3), dtype=np.uint8)
    colors[0] = np.array([0, 0, 0], dtype=np.uint8)  # background = black

    out[:] = colors[labels]

    out_path = os.path.join(OUT_DIR, f"{name_prefix}_components.png")
    cv2.imwrite(out_path, out)
    print("  saved:", out_path)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1) load mask
    img = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read {IMG_PATH}")

    # (optional) ensure it's binary
    _, img_bin = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)

    # 2) also save connected components on raw mask (no closing)
    color_connected_components(img_bin, "raw")

    # 3) try several morphological kernel sizes
    for k in KERNEL_SIZES:
        print(f"\n=== kernel size {k}x{k} ===")
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))

        # closing = dilate then erode: bridges small gaps
        closed = cv2.morphologyEx(img_bin, cv2.MORPH_CLOSE, kernel)

        # optional: save the closed mask itself
        closed_path = os.path.join(OUT_DIR, f"closed_k{k}.png")
        cv2.imwrite(closed_path, closed)
        print("saved closed mask:", closed_path)

        # 4) connected components on closed mask
        color_connected_components(closed, f"cc_k{k}")


if __name__ == "__main__":
    main()
