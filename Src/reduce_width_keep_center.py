
import cv2
import numpy as np
from pathlib import Path

# ============================================================
# USER PARAMETERS
# ============================================================
INPUT_PATH = r"C:\Users\makss\Git\Galatsi-Semester-Project\Results\occupancy_analysis\per_zones\mask_closed_cleaned.png"
OUTPUT_DIR = Path(r"C:\Users\makss\Git\Galatsi-Semester-Project\Results\Images\test\test_new_mask")

# Final maximum target width of white structures, in pixels
TARGET_MAX_WIDTH_PX = 20

# Remove tiny leftovers
MIN_COMPONENT_AREA = 40

# Optional cleanup
MORPH_OPEN_KERNEL = 3   # set 0 to disable
MORPH_CLOSE_KERNEL = 3  # set 0 to disable

# Optional rectangle replacement after width reduction
REPLACE_BY_RECTANGLES = False
RECT_SHRINK = 0.92
MIN_RECT_COVERAGE = 0.25

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

def morph_cleanup(binary, open_k=0, close_k=0):
    out = binary.copy()
    if open_k and open_k > 1:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (open_k, open_k))
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, k)
    if close_k and close_k > 1:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (close_k, close_k))
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k)
    return out

def make_disk_kernel(radius):
    radius = max(1, int(radius))
    size = 2 * radius + 1
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    mask = (x * x + y * y) <= radius * radius
    kernel = np.zeros((size, size), dtype=np.uint8)
    kernel[mask] = 1
    return kernel

def morphological_skeleton(binary):
    """
    Basic morphological skeletonization.
    Input must be 0/255 uint8.
    Output is 0/255 uint8.
    """
    img = (binary > 0).astype(np.uint8) * 255
    skel = np.zeros_like(img)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while True:
        eroded = cv2.erode(img, element)
        opened = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, opened)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:
            break

    return skel

def reduce_width_keep_center(binary, target_max_width_px):
    """
    Reduce thickness while keeping the centerline.

    Strategy:
    1) compute skeleton of original mask
    2) dilate skeleton to target half-width
    3) intersect with original mask so we NEVER add outside original

    This avoids the 'remove the center' problem from distance-thresholding.
    """
    skeleton = morphological_skeleton(binary)

    radius = max(1, int(round(target_max_width_px / 2.0)))
    kernel = make_disk_kernel(radius)

    rebuilt = cv2.dilate(skeleton, kernel)
    reduced = cv2.bitwise_and(rebuilt, binary)

    return reduced, skeleton, rebuilt

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

        inside = cv2.bitwise_and(rect_mask, comp)

        inside_area = int(np.count_nonzero(inside))
        coverage = inside_area / max(area, 1)

        if coverage >= min_rect_coverage:
            out[inside > 0] = 255
        else:
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

    reduced, skeleton, rebuilt = reduce_width_keep_center(binary, TARGET_MAX_WIDTH_PX)
    reduced = morph_cleanup(reduced, MORPH_OPEN_KERNEL, MORPH_CLOSE_KERNEL)
    reduced = remove_small_components(reduced, MIN_COMPONENT_AREA)

    if REPLACE_BY_RECTANGLES:
        final = conservative_rectangle_replace(
            reduced,
            rect_shrink=RECT_SHRINK,
            min_rect_coverage=MIN_RECT_COVERAGE
        )
    else:
        final = reduced.copy()

    save_image(OUTPUT_DIR / "01_binary.png", binary)
    save_image(OUTPUT_DIR / "02_skeleton.png", skeleton)
    save_image(OUTPUT_DIR / "03_rebuilt_from_skeleton.png", rebuilt)
    save_image(OUTPUT_DIR / "04_reduced_width.png", reduced)
    save_image(OUTPUT_DIR / "05_final.png", final)

    if SAVE_DEBUG:
        overlay = np.zeros((binary.shape[0], binary.shape[1], 3), dtype=np.uint8)
        overlay[:, :, 0] = binary   # original in blue
        overlay[:, :, 1] = final    # result in green
        save_image(OUTPUT_DIR / "debug_overlay_original_vs_final.png", overlay)

    print("Done.")
    print(f"Saved outputs to: {OUTPUT_DIR}")
    print("Main result:", OUTPUT_DIR / "05_final.png")


if __name__ == "__main__":
    main()
