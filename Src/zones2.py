import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os

"""
Segments a binary parking mask into K spatial clusters using K-Means.

It loads a binary image, extracts the (x, y) coordinates of all white pixels, and runs K-Means for K in
[K_MIN, K_MAX]. For each K, pixels are colored by their assigned cluster and the resulting cluster map is
saved as an image in OUT_DIR.
"""

# -------- CONFIG ---------
IMG_PATH = r"C:/Users/makss/Git/Galatsi-Semester-Project/mask_closed.png"   # your binary image
K_MIN = 2                      # minimum number of clusters
K_MAX = 10                     # maximum number of clusters
OUT_DIR = r"C:/Users/makss/Git/Galatsi-Semester-Project/Results/parking_detection/zones2/test2/"
# --------------------------

def main():
    # Create output folder
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1) Load binary mask
    img = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read {IMG_PATH}")

    h, w = img.shape

    # 2) Extract (x,y) of white pixels
    ys, xs = np.where(img > 0)
    points = np.column_stack((xs, ys))
    print("Foreground pixels:", len(points))

    # 3) Loop through cluster counts
    for k in range(K_MIN, K_MAX + 1):
        print(f"\nClustering with K = {k} ...")

        km = KMeans(n_clusters=k, random_state=0, n_init=10)
        labels = km.fit_predict(points)

        # Create colored image
        out_img = np.zeros((h, w, 3), dtype=np.uint8)
        rng = np.random.default_rng(42)
        colors = rng.integers(0, 255, size=(k, 3), dtype=np.uint8)

        for (x, y), lab in zip(points, labels):
            out_img[y, x] = colors[lab]

        # Save image
        save_path = os.path.join(OUT_DIR, f"clusters_K{k}.png")
        cv2.imwrite(save_path, out_img)
        print("Saved:", save_path)

    print("\nFinished!")

if __name__ == "__main__":
    main()
