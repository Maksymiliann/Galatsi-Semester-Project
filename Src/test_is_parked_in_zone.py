import cv2
import numpy as np
import pandas as pd
import glob
import os
from pathlib import Path
from tqdm import tqdm

"""
This script filters detections to one specific parking zone and exports new TXT files.

For each vehicle in each TXT file:
1. The vehicle polygon is read in the SOURCE image coordinates.
2. It is reprojected into the REFERENCE image coordinates using a homography.
3. The vehicle is assigned to a target zone using polygon overlap with the zone labels.
4. If the vehicle belongs to the target zone:
      - label = "parked" if:
            state == "stop"
            AND overlap with global parking mask > OVERLAP_THR
      - otherwise label = "moving"
5. A new TXT file is written with only vehicles from the chosen zone.

The output TXT keeps the same columns as the original file, but updates/overwrites
the "state" column with "parked" or "moving".
"""


###########################################################
# CONFIG
###########################################################
MASK_PATH = r"C:/Users/makss/Git/Galatsi-Semester-Project/Results/occupancy_analysis/per_zones/mask_closed_cleaned.png"
ZONES_PATH = r"C:/Users/makss/Git/Galatsi-Semester-Project/Results/occupancy_analysis/per_zones/mask_zones_id_cleaned.png"
FOLDER = r"C:/Users/makss/Git/Galatsi-Semester-Project/Results/TXT_0314_D2_S4_S1_padded"

OUT_FOLDER = r"C:/Users/makss/Git/Galatsi-Semester-Project/Results/TXT_0314_filtered_zone_10"

REF_IMG_PATH = r"C:/Users/makss/Git/Galatsi-Semester-Project/Results/Images/needed for script/frame_1_0004.png"
SRC_IMG_PATH = r"C:/Users/makss/Git/Galatsi-Semester-Project/Results/Images/needed for script/frame_1_0314.png"

TARGET_ZONE_ID = 10
OVERLAP_THR = 0.7          # minimum overlap with parking mask to call it parked
ZONE_OVERLAP_THR = 0.3     # minimum overlap with target zone to consider the vehicle in that zone
MIN_ZONE_AREA = 5000


###########################################################
# HELPERS
###########################################################
def fast_bbox(poly_xy, W, H):
    x_coords = poly_xy[:, 0]
    y_coords = poly_xy[:, 1]

    xmin = int(np.floor(np.min(x_coords)))
    xmax = int(np.ceil(np.max(x_coords)))
    ymin = int(np.floor(np.min(y_coords)))
    ymax = int(np.ceil(np.max(y_coords)))

    xmin_c = max(xmin, 0)
    ymin_c = max(ymin, 0)
    xmax_c = min(xmax, W - 1)
    ymax_c = min(ymax, H - 1)

    return xmin_c, xmax_c, ymin_c, ymax_c


def rasterize_poly(poly_xy, W, H):
    """
    Rasterize polygon in REF frame.
    Returns:
        xmin_c, xmax_c, ymin_c, ymax_c, veh_local_mask
    or None if invalid.
    """
    xmin_c, xmax_c, ymin_c, ymax_c = fast_bbox(poly_xy, W, H)
    if xmax_c < xmin_c or ymax_c < ymin_c:
        return None

    shifted = poly_xy.copy()
    shifted[:, 0] -= xmin_c
    shifted[:, 1] -= ymin_c
    shifted = shifted.astype(np.int32)

    roi_w = xmax_c - xmin_c + 1
    roi_h = ymax_c - ymin_c + 1
    veh_local = np.zeros((roi_h, roi_w), dtype=np.uint8)
    cv2.fillPoly(veh_local, [shifted], 1)

    return xmin_c, xmax_c, ymin_c, ymax_c, veh_local


###########################################################
# HOMOGRAPHY
###########################################################
def estimate_homography(img_ref_bgr, img_bgr, ratio=0.75, ransac_thr=3.0, max_kpts=4000):
    """
    Returns H (img -> ref), nb_inliers, reproj_err.
    If failed, returns identity.
    """
    try:
        sift = cv2.SIFT_create(nfeatures=max_kpts)
    except Exception:
        sift = None

    if sift is None:
        orb = cv2.ORB_create(nfeatures=max_kpts)
        k1, d1 = orb.detectAndCompute(img_ref_bgr, None)
        k2, d2 = orb.detectAndCompute(img_bgr, None)
        norm = cv2.NORM_HAMMING
    else:
        k1, d1 = sift.detectAndCompute(img_ref_bgr, None)
        k2, d2 = sift.detectAndCompute(img_bgr, None)
        norm = cv2.NORM_L2

    if d1 is None or d2 is None or len(k1) < 4 or len(k2) < 4:
        return np.eye(3, dtype=np.float32), 0, float("inf")

    bf = cv2.BFMatcher(norm)
    knn = bf.knnMatch(d2, d1, k=2)  # src -> ref

    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)

    if len(good) < 4:
        return np.eye(3, dtype=np.float32), 0, float("inf")

    pts2 = np.float32([k2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)  # src
    pts1 = np.float32([k1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)  # ref

    H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, ransac_thr)
    if H is None:
        return np.eye(3, dtype=np.float32), 0, float("inf")

    inl = mask.ravel().astype(bool)
    err = float("inf")
    if np.any(inl):
        p2 = pts2[inl].reshape(-1, 2)
        p1 = pts1[inl].reshape(-1, 2)
        p2h = np.hstack([p2, np.ones((len(p2), 1), dtype=np.float32)])
        proj = (H @ p2h.T).T
        proj = proj[:, :2] / proj[:, 2:3]
        err = float(np.mean(np.linalg.norm(proj - p1, axis=1)))

    return H.astype(np.float32), int(np.sum(inl)), err


###########################################################
# CORE VEHICLE CLASSIFICATION
###########################################################
def classify_vehicle_in_target_zone(row, H_mat, mask_bin, labels, target_zone_id, overlap_thr, zone_overlap_thr):
    """
    Returns:
        in_target_zone (bool),
        new_label (str or None),
        parking_ratio (float),
        zone_ratio (float)
    """
    H_img, W_img = mask_bin.shape

    poly_src = np.array([
        (row["veh_bb_x1"], row["veh_bb_y1"]),
        (row["veh_bb_x2"], row["veh_bb_y2"]),
        (row["veh_bb_x3"], row["veh_bb_y3"]),
        (row["veh_bb_x4"], row["veh_bb_y4"]),
    ], dtype=np.float32)

    pts = poly_src.reshape(-1, 1, 2)
    pts_ref = cv2.perspectiveTransform(pts, H_mat).reshape(-1, 2).astype(np.float32)

    if not np.isfinite(pts_ref).all():
        return False, None, 0.0, 0.0

    res = rasterize_poly(pts_ref, W_img, H_img)
    if res is None:
        return False, None, 0.0, 0.0

    xmin_c, xmax_c, ymin_c, ymax_c, veh_local = res
    veh_px = int(np.sum(veh_local))
    if veh_px == 0:
        return False, None, 0.0, 0.0

    mask_roi = mask_bin[ymin_c:ymax_c+1, xmin_c:xmax_c+1]
    labels_roi = labels[ymin_c:ymax_c+1, xmin_c:xmax_c+1]

    parking_overlap_px = int(np.sum((veh_local == 1) & (mask_roi == 1)))
    zone_overlap_px = int(np.sum((veh_local == 1) & (labels_roi == target_zone_id)))

    parking_ratio = parking_overlap_px / veh_px
    zone_ratio = zone_overlap_px / veh_px

    # vehicle considered inside target zone only if enough overlap
    in_target_zone = zone_ratio > zone_overlap_thr

    if not in_target_zone:
        return False, None, parking_ratio, zone_ratio

    state_val = str(row.get("state", "")).strip().lower()

    if state_val == "stop" and parking_ratio > overlap_thr:
        new_label = "parked"
    else:
        new_label = "moving"

    return True, new_label, parking_ratio, zone_ratio


###########################################################
# MAIN
###########################################################
if __name__ == "__main__":
    os.makedirs(OUT_FOLDER, exist_ok=True)

    # ------------------------------------------------------
    # Load parking mask
    # ------------------------------------------------------
    mask_img = cv2.imread(MASK_PATH, cv2.IMREAD_GRAYSCALE)
    if mask_img is None:
        raise FileNotFoundError(f"Could not load mask image at {MASK_PATH}")

    mask_bin = (mask_img > 127).astype(np.uint8)
    H_img, W_img = mask_bin.shape

    # ------------------------------------------------------
    # Load reference/source images and estimate homography
    # ------------------------------------------------------
    img_ref = cv2.imread(REF_IMG_PATH, cv2.IMREAD_COLOR)
    if img_ref is None:
        raise FileNotFoundError(f"Could not load reference image at {REF_IMG_PATH}")

    img_src = cv2.imread(SRC_IMG_PATH, cv2.IMREAD_COLOR)
    if img_src is None:
        raise FileNotFoundError(f"Could not load source image at {SRC_IMG_PATH}")

    if os.path.abspath(REF_IMG_PATH) == os.path.abspath(SRC_IMG_PATH):
        H_mat = np.eye(3, dtype=np.float32)
        print("[H] REF == SRC -> identity homography.")
    else:
        H_mat, ninl, err = estimate_homography(img_ref, img_src)
        print(f"[H] Estimated SRC -> REF: inliers={ninl}, reproj_err={err:.3f}px")

    # ------------------------------------------------------
    # Load zone labels
    # ------------------------------------------------------
    if ZONES_PATH and Path(ZONES_PATH).exists():
        lab_img = cv2.imread(ZONES_PATH, cv2.IMREAD_GRAYSCALE)
        if lab_img is None:
            raise FileNotFoundError(f"Could not load zones image at {ZONES_PATH}")
        labels = lab_img.astype(np.int32)
        labels[mask_bin == 0] = 0
    else:
        num, lab = cv2.connectedComponents(mask_bin, connectivity=8)
        labels = lab.astype(np.int32)

    # ------------------------------------------------------
    # Remove tiny zones if desired
    # ------------------------------------------------------
    if MIN_ZONE_AREA and MIN_ZONE_AREA > 0:
        vals = labels.ravel()
        counts = np.bincount(vals)
        num = len(counts) - 1

        keep = np.zeros(num + 1, dtype=np.uint8)
        keep[0] = 1

        for z in range(1, num + 1):
            if counts[z] >= MIN_ZONE_AREA:
                keep[z] = 1

        remap = np.zeros(num + 1, dtype=np.int32)
        nid = 0
        for z in range(num + 1):
            if keep[z]:
                remap[z] = nid
                nid += 1

        labels = remap[labels]
        labels[mask_bin == 0] = 0

        print("[INFO] Zones remapped after small-zone filtering.")
        print("[INFO] Make sure TARGET_ZONE_ID still matches the new zone id.")

    # Check target zone exists
    unique_zones = set(np.unique(labels).tolist())
    if TARGET_ZONE_ID not in unique_zones:
        raise ValueError(
            f"TARGET_ZONE_ID={TARGET_ZONE_ID} not found in labels. "
            f"Available zones: {sorted([z for z in unique_zones if z != 0])}"
        )

    # ------------------------------------------------------
    # Process all TXT files
    # ------------------------------------------------------
    txt_files = sorted(glob.glob(os.path.join(FOLDER, "*.txt")))
    if len(txt_files) == 0:
        raise FileNotFoundError(f"No TXT files found in {FOLDER}")

    total_kept = 0
    total_parked = 0
    total_moving = 0

    for txt_path in tqdm(txt_files, desc="Processing TXT files"):
        df = pd.read_csv(txt_path, sep=";", engine="python")
        df.columns = [c.strip() for c in df.columns]

        kept_rows = []

        for _, row in df.iterrows():
            in_zone, new_label, parking_ratio, zone_ratio = classify_vehicle_in_target_zone(
                row=row,
                H_mat=H_mat,
                mask_bin=mask_bin,
                labels=labels,
                target_zone_id=TARGET_ZONE_ID,
                overlap_thr=OVERLAP_THR,
                zone_overlap_thr=ZONE_OVERLAP_THR
            )

            if not in_zone:
                continue

            row_out = row.copy()

            # overwrite state with new label
            row_out["state"] = new_label

            kept_rows.append(row_out)

            total_kept += 1
            if new_label == "parked":
                total_parked += 1
            else:
                total_moving += 1

        out_path = os.path.join(OUT_FOLDER, os.path.basename(txt_path))

        if len(kept_rows) == 0:
            # keep same columns if no rows
            empty_df = df.iloc[0:0].copy()
            empty_df.to_csv(out_path, sep=";", index=False)
        else:
            out_df = pd.DataFrame(kept_rows)
            out_df.to_csv(out_path, sep=";", index=False)

    print("\n================ DONE ================\n")
    print(f"Target zone id         : {TARGET_ZONE_ID}")
    print(f"Output folder          : {OUT_FOLDER}")
    print(f"Vehicles kept          : {total_kept}")
    print(f"Vehicles labeled parked: {total_parked}")
    print(f"Vehicles labeled moving: {total_moving}")