import cv2
import numpy as np

# -----------------
# PATHS
# -----------------
MASK_IDS_PATH = r"C:/Users/makss/Git/Galatsi-Semester-Project/Results/Ground_truth/mask_ids.png"
OUT_COLOR     = r"C:/Users/makss/Git/Galatsi-Semester-Project/Results/Ground_truth/mask_ids_color.png"

# -----------------
# LOAD
# -----------------
mask = cv2.imread(MASK_IDS_PATH, cv2.IMREAD_UNCHANGED)
assert mask is not None, "Impossible de lire mask_ids.png"

h, w = mask.shape
color = np.zeros((h, w, 3), dtype=np.uint8)

# -----------------
# COLORIZE
# -----------------
ids = np.unique(mask)
ids = ids[ids != 0]  # remove background

for i in ids:
    # pseudo-random but deterministic color
    c = (
        int((37 * i) % 255),
        int((17 * i) % 255),
        int((97 * i) % 255),
    )
    color[mask == i] = c

# -----------------
# SAVE
# -----------------
cv2.imwrite(OUT_COLOR, color)
print("Saved:", OUT_COLOR)
