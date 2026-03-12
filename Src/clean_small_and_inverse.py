import cv2
import numpy as np

# ===============================
# PARAMETERS
# ===============================

input_path = "Results/parking_detection/dwell_mult_4/test8_thr_0.85/parking_dwell_state_MULTI_REG_parking_location_mask_thr_0.800.png"
output_path = "Results/Images/test/test4.png"

kernel_size = 10   # try 3, 5, or 7

# ===============================
# FUNCTION
# ===============================

def remove_small_white_objects(img, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    cleaned = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    return cleaned

import cv2
import numpy as np

def remove_small_white_patches(img, min_area):
    """
    Remove white connected components smaller than min_area.

    Parameters
    ----------
    img : np.ndarray
        Binary image (white objects on black background).
        White should be 255, black should be 0.
    min_area : int
        Minimum area in pixels to keep.

    Returns
    -------
    cleaned : np.ndarray
        Binary image with small white patches removed.
    """
    
    # Make sure image is binary
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # Connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    # Output image
    cleaned = np.zeros_like(binary)

    # Start from 1 to skip background
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        
        if area >= min_area:
            cleaned[labels == label] = 255

    return cleaned

# ===============================
# LOAD IMAGE
# ===============================

img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
_, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# ===============================
# INVERT IMAGE
# ===============================

inverted = cv2.bitwise_not(img)

# ===============================
# REMOVE SMALL WHITE BLOBS
# ===============================

# cleaned = remove_small_white_objects(inverted, kernel_size)
cleaned = remove_small_white_patches(inverted, 1500)

# ===============================
# SAVE RESULT
# ===============================

cv2.imwrite(output_path, cleaned)
print("Saved:", output_path)