import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
img = cv2.imread(r"C:\Users\makss\Git\Galatsi-Semester-Project\Results\parking_detection\dwell_mult_4\test4_thr_0.85\parking_dwell_state_MULTI_REG_score_map_PRUNED.png")

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower = np.array([15, 10, 10])
upper_blue = np.array([140, 255, 255])

mask = cv2.inRange(hsv, lower, upper_blue)

plt.imshow(mask, cmap="gray")
plt.title("Blue threshold")
plt.axis("off")
plt.show()

