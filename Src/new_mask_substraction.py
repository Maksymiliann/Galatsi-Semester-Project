import cv2
import numpy as np

# ===============================
# PARAMETERS
# ===============================

img1_path = "Results/occupancy_analysis/per_zones/mask_closed_cleaned.png"
img2_path = "Results/Images/test/test4.png"
output_path = "Results/Images/test/new_mask2.png"

# ===============================
# LOAD IMAGES
# ===============================

img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

# ensure binary
_, img1 = cv2.threshold(img1, 127, 255, cv2.THRESH_BINARY)
_, img2 = cv2.threshold(img2, 127, 255, cv2.THRESH_BINARY)

# ===============================
# SUBTRACT
# ===============================

result = cv2.bitwise_and(img1, cv2.bitwise_not(img2))

# ===============================
# SAVE
# ===============================

cv2.imwrite(output_path, result)

print("Saved:", output_path)