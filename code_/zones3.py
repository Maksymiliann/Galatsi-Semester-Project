import cv2
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import os

"""
Runs DBSCAN clustering on the white pixels of a binary mask to segment it into spatial zones.

It loads the mask, extracts (x, y) coordinates of all foreground pixels, optionally subsamples them for speed,
then sweeps over a grid of DBSCAN parameters (eps, min_samples). For each parameter pair, it clusters the points,
colors each cluster (noise stays black), and saves a visualization image to OUT_DIR. Optional Matplotlib display
can be enabled with SHOW_PLOTS.
"""


# -------- CONFIG ---------
IMG_PATH = r"C:/Users/makss/Git/Galatsi-Semester-Project/mask_closed.png"

OUT_DIR = r"C:/Users/makss/Git/Galatsi-Semester-Project/Results/parking_detection/zones3/test1/"

# Values to test
EPS_LIST = [10.0, 20, 30, 40, 50]          # try whatever you want here
MIN_SAMPLES_LIST = [20, 50, 100, 200, 500, 1000]

SUBSAMPLE_N = None                    # e.g. 50000 or None to use all
SHOW_PLOTS = False                    # True if you want pop-up windows
# --------------------------


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1) Load mask
    img = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read {IMG_PATH}")
    h, w = img.shape

    # 2) Get coordinates of white pixels
    ys, xs = np.where(img > 0)
    points = np.column_stack((xs, ys))  # shape (N, 2)
    print("Foreground pixels:", len(points))

    # Optional subsample to speed up DBSCAN
    if SUBSAMPLE_N is not None and SUBSAMPLE_N < len(points):
        idx = np.random.choice(len(points), SUBSAMPLE_N, replace=False)
        points_used = points[idx]
    else:
        points_used = points

    # 3) Loop over parameter grid
    for eps in EPS_LIST:
        for min_samples in MIN_SAMPLES_LIST:
            print(f"\nRunning DBSCAN with eps={eps}, min_samples={min_samples}...")

            db = DBSCAN(eps=eps, min_samples=min_samples).fit(points_used)
            labels = db.labels_

            unique_labels = sorted(set(labels))
            n_clusters = len([l for l in unique_labels if l != -1])
            print("Found clusters (excluding noise):", n_clusters)

            # 4) Build colored image
            cluster_img = np.zeros((h, w, 3), dtype=np.uint8)

            rng = np.random.default_rng(0)
            colors = {}
            for lab in unique_labels:
                if lab == -1:
                    # noise -> keep black
                    colors[lab] = np.array([0, 0, 0], dtype=np.uint8)
                else:
                    colors[lab] = rng.integers(0, 255, size=3, dtype=np.uint8)

            # Paint only the used points
            for (x, y), lab in zip(points_used, labels):
                cluster_img[y, x] = colors[lab]

            # 5) Save result
            eps_str = str(eps).replace('.', 'p')
            fname = f"dbscan_eps{eps_str}_min{min_samples}.png"
            out_path = os.path.join(OUT_DIR, fname)
            cv2.imwrite(out_path, cluster_img)
            print("Saved:", out_path)

            # Optional show
            if SHOW_PLOTS:
                plt.figure()
                plt.title(f"eps={eps}, min_samples={min_samples}, clusters={n_clusters}")
                plt.imshow(cv2.cvtColor(cluster_img, cv2.COLOR_BGR2RGB))
                plt.axis("off")
                plt.show()

    print("\nDone!")


if __name__ == "__main__":
    main()
