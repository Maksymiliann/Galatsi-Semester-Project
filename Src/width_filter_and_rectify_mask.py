
import cv2
import numpy as np
from pathlib import Path

# ============================================================
# USER PARAMETERS
# ============================================================
INPUT_PATH = r"C:\Users\makss\Git\Galatsi-Semester-Project\Results\occupancy_analysis\per_zones\mask_closed_cleaned.png"
OUTPUT_DIR = Path(r"C:\Users\makss\Git\Galatsi-Semester-Project\Results\Images\test\test_new_mask")

# Maximum allowed local width of white regions, in pixels.
# Anything locally thicker than this is removed.
MAX_WIDTH_PX = 40

# Remove very small leftovers after filtering
MIN_COMPONENT_AREA = 40

# Optional cleanup after width filtering
MORPH_OPEN_KERNEL = 0   # set to 0 to disable
MORPH_CLOSE_KERNEL = 0  # set to 0 to disable

# Rectangle approximation
REPLACE_BY_RECTANGLES = True

# Rectangle is shrunk a bit before intersecting with component.
# This helps stay conservative.
RECT_SHRINK = 0.92

# If the rectangle kept inside the component is too small compared
# to the component, keep the component instead.
MIN_RECT_COVERAGE = 0.25

# Save debug images
SAVE_DEBUG = True


# ============================================================
# HELPERS
# ============================================================
def ensure_binary(img):
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    return binary

def remove_small_components(binary, min_area):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    out = np.zeros_like(binary)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            out[labels == i] = 255
    return out

def filter_by_local_width(binary, max_width_px):
    """
    Keep only pixels whose local thickness is <= max_width_px.
    Thickness is approximated as 2 * distance to nearest background.
    """
    mask01 = (binary > 0).astype(np.uint8)
    dist = cv2.distanceTransform(mask01, cv2.DIST_L2, 5)
    keep = dist <= (max_width_px / 2.0)
    out = np.where(mask01 & keep, 255, 0).astype(np.uint8)
    return out, dist

def morph_cleanup(binary, open_k=0, close_k=0):
    out = binary.copy()
    if open_k and open_k > 1:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (open_k, open_k))
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, k)
    if close_k and close_k > 1:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (close_k, close_k))
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k)
    return out

def contour_mask(binary):
    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cnts

def draw_rotated_rect(shape, rect):
    box = cv2.boxPoints(rect)
    box = np.round(box).astype(np.int32)
    canvas = np.zeros(shape, dtype=np.uint8)
    cv2.fillConvexPoly(canvas, box, 255)
    return canvas

def conservative_rectangle_replace(binary, rect_shrink=0.92, min_rect_coverage=0.25):
    """
    For each component:
    - fit a min-area rotated rectangle
    - shrink it slightly
    - rasterize it
    - intersect with the original component
    This guarantees we never add pixels outside the component.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    out = np.zeros_like(binary)

    for i in range(1, num_labels):
        comp = np.where(labels == i, 255, 0).astype(np.uint8)
        area = stats[i, cv2.CC_STAT_AREA]
        if area == 0:
            continue

        cnts = contour_mask(comp)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)

        if len(cnt) < 3:
            out[comp > 0] = 255
            continue

        rect = cv2.minAreaRect(cnt)
        (cx, cy), (w, h), angle = rect

        if w < 1 or h < 1:
            out[comp > 0] = 255
            continue

        rect_shrunk = ((cx, cy), (max(1, w * rect_shrink), max(1, h * rect_shrink)), angle)
        rect_mask = draw_rotated_rect(binary.shape, rect_shrunk)

        # Conservative rule: never add outside the original component
        inside = cv2.bitwise_and(rect_mask, comp)

        inside_area = int(np.count_nonzero(inside))
        coverage = inside_area / max(area, 1)

        if coverage >= min_rect_coverage:
            out[inside > 0] = 255
        else:
            # Fallback to original component if rectangle is too destructive
            out[comp > 0] = 255

    return out

def save_image(path, img):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


# ============================================================
# MAIN
# ============================================================
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(INPUT_PATH, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {INPUT_PATH}")

    binary = ensure_binary(img)

    # 1) Remove regions that are too thick locally
    width_filtered, dist = filter_by_local_width(binary, MAX_WIDTH_PX)

    # 2) Morph cleanup + remove tiny pieces
    cleaned = morph_cleanup(width_filtered, MORPH_OPEN_KERNEL, MORPH_CLOSE_KERNEL)
    cleaned = remove_small_components(cleaned, MIN_COMPONENT_AREA)

    # 3) Optional rectangle replacement
    if REPLACE_BY_RECTANGLES:
        rectified = conservative_rectangle_replace(
            cleaned,
            rect_shrink=RECT_SHRINK,
            min_rect_coverage=MIN_RECT_COVERAGE,
        )
    else:
        rectified = cleaned.copy()

    # Save outputs
    save_image(OUTPUT_DIR / "01_binary.png", binary)
    save_image(OUTPUT_DIR / "02_width_filtered.png", width_filtered)
    save_image(OUTPUT_DIR / "03_cleaned.png", cleaned)
    save_image(OUTPUT_DIR / "04_rectified.png", rectified)

    if SAVE_DEBUG:
        # visualize local width estimate = 2 * distance
        local_width = np.clip(dist * 2.0, 0, 255).astype(np.uint8)
        save_image(OUTPUT_DIR / "debug_local_width.png", local_width)

        # overlay for quick inspection
        overlay = np.zeros((binary.shape[0], binary.shape[1], 3), dtype=np.uint8)
        overlay[:, :, 0] = binary                  # original in blue
        overlay[:, :, 1] = rectified              # result in green
        overlay[:, :, 2] = 0
        save_image(OUTPUT_DIR / "debug_overlay_original_vs_result.png", overlay)

    print("Done.")
    print(f"Saved outputs to: {OUTPUT_DIR}")
    print("Main result:", OUTPUT_DIR / "04_rectified.png")


if __name__ == "__main__":
    main()
