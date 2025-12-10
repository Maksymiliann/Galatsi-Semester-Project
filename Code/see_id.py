
import cv2
import numpy as np
import random

# =========================
# CONFIG
# =========================
MASK_PATH   = r"C:/Users/makss/Git/Galatsi-Semester-Project/Results/occupancy_analysis/per_zones/mask_zones_id_cleaned.png"
OUTPUT_PATH = r"C:/Users/makss/Git/Galatsi-Semester-Project/Results/occupancy_analysis/per_zones/mask_zones_id_cleaned_annoted.png"

# =========================
# LOAD MASK
# =========================
mask_raw = cv2.imread(MASK_PATH, cv2.IMREAD_UNCHANGED)
if mask_raw.ndim == 3:
    mask = mask_raw[:, :, 0]
else:
    mask = mask_raw

h, w = mask.shape

# =========================
# GENERATE RANDOM COLORS FOR EACH ID
# =========================
unique_ids = np.unique(mask)
unique_ids = unique_ids[unique_ids != 0]  # ignore ID = 0

color_map = {}

for zone_id in unique_ids:
    # Generate bright random color (avoid dark grayscale-like colors)
    color_map[zone_id] = (
        random.randint(80, 255),
        random.randint(80, 255),
        random.randint(80, 255)
    )

# Background is black
color_img = np.zeros((h, w, 3), dtype=np.uint8)

# Color each pixel based on its ID
for zone_id in unique_ids:
    color_img[mask == zone_id] = color_map[zone_id]

# =========================
# DRAW CONTOURS + LABELS
# =========================
for zone_id in unique_ids:
    zone_mask = np.uint8(mask == zone_id) * 255
    contours, _ = cv2.findContours(zone_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contour in white
    cv2.drawContours(color_img, contours, -1, (255, 255, 255), 1)

    # Compute centroid
    M = cv2.moments(zone_mask)
    if M["m00"] == 0:
        continue
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    # BIGGER LABEL SIZE
    text = str(int(zone_id))
    font_scale = 1.4      # <-- increase this for even bigger numbers
    thickness_outline = 5
    thickness_text    = 2

    # Black outline
    cv2.putText(color_img, text, (cx - 15, cy + 10),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (0, 0, 0), thickness_outline, cv2.LINE_AA)

    # White foreground
    cv2.putText(color_img, text, (cx - 15, cy + 10),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (255, 255, 255), thickness_text, cv2.LINE_AA)

# =========================
# SAVE RESULT
# =========================
cv2.imwrite(OUTPUT_PATH, color_img)
print("Saved annotation:", OUTPUT_PATH)

