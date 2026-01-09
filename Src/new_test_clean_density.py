import cv2
import numpy as np

"""
This script cleans a binary mask by keeping only white pixels that are supported by enough nearby white pixels.

It loads a grayscale mask image, binarizes it (0/255), and converts it to a 0/1 array where white pixels are 1.
Using an integral image, it efficiently counts how many white pixels fall inside a square neighborhood of size
(2*RADIUS + 1) around every pixel.

A white pixel is kept only if:
- it is white in the original mask, AND
- the number of white pixels in its neighborhood is at least MIN_WHITE_PIXELS

All other pixels are set to black. The result is a denoised mask that removes isolated/sparse white regions while
preserving dense areas, and the cleaned mask is saved to OUT_PATH.
"""



# -----------------------
# CONFIG
# -----------------------
IN_PATH  = r"C:/Users/makss/Git/Galatsi-Semester-Project/Results/parking_detection/dwell_mult_4/test5_thr_0.85/parking_dwell_state_MULTI_REG_zones_mask.png"
OUT_PATH = r"C:/Users/makss/Git/Galatsi-Semester-Project/Results/occupancy_analysis/per_zones/cleaned_bw_2.png"

RADIUS = 100              # voisinage = (2R+1)x(2R+1)
MIN_WHITE_PIXELS = 1300   # nombre minimal de pixels blancs requis
THRESH = 127

# -----------------------
# LOAD + BINARIZE
# -----------------------
img = cv2.imread(IN_PATH, cv2.IMREAD_GRAYSCALE)
assert img is not None, f"Cannot read {IN_PATH}"

_, bw = cv2.threshold(img, THRESH, 255, cv2.THRESH_BINARY)

# blanc = 1, noir = 0
white = (bw == 255).astype(np.uint8)

# -----------------------
# COUNT WHITE PIXELS (integral image)
# -----------------------
I = cv2.integral(white)
H, W = white.shape

y = np.arange(H)[:, None]
x = np.arange(W)[None, :]

y0 = np.clip(y - RADIUS, 0, H)
y1 = np.clip(y + RADIUS + 1, 0, H)
x0 = np.clip(x - RADIUS, 0, W)
x1 = np.clip(x + RADIUS + 1, 0, W)

# nombre de pixels blancs dans la fenêtre
white_count = I[y1, x1] - I[y0, x1] - I[y1, x0] + I[y0, x0]

# -----------------------
# FILTER RULE
# -----------------------
keep_white = (white == 1) & (white_count >= MIN_WHITE_PIXELS)

clean = np.zeros_like(bw)
clean[keep_white] = 255

cv2.imwrite(OUT_PATH, clean)
print(f"Saved: {OUT_PATH}")
