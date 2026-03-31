import cv2
import numpy as np
import pandas as pd
import glob
import os
import re
from pathlib import Path
from tqdm import tqdm

"""
This script reads a folder of TXT files and creates a new folder with the same TXT files,
but with 3 additional columns:

    - parking_zone
    - timestamp_in
    - timestamp_out

Logic:
1. Each vehicle polygon is read in SOURCE image coordinates.
2. It is projected into the REFERENCE image coordinates using a homography.
3. Polygon overlap is computed with:
      - the global parking mask
      - the zone labels image
4. A vehicle is considered PARKED if:
      - original state == "stop"
      - overlap with parking mask > OVERLAP_THR
      - overlap with best zone > ZONE_OVERLAP_THR
5. If parked:
      - parking_zone = best overlapping zone id
   otherwise:
      - parking_zone = -1
6. For each vehicle_id across time, contiguous parked sequences are detected.
   For each parked sequence:
      - timestamp_in = first timestamp of the sequence
      - timestamp_out = last timestamp of the sequence
   For non-parked rows:
      - timestamp_in = None
      - timestamp_out = None

Important assumptions:
- TXT files are processed in sorted order.
- Timestamps are extracted from the filename.
- If no number is found in the filename, the sorted file index is used instead.
- vehicle_id is assumed to be temporally consistent across files.
"""

###########################################################
# CONFIG
###########################################################
MASK_PATH = r"C:/Users/makss/Git/Galatsi-Semester-Project/Results/occupancy_analysis/per_zones/mask_closed_cleaned.png"
ZONES_PATH = r"C:/Users/makss/Git/Galatsi-Semester-Project/Results/occupancy_analysis/per_zones/mask_zones_id_cleaned.png"
FOLDER = r"C:/Users/makss/Git/Galatsi-Semester-Project/Results/TXT_0314_D2_S4_S1_padded_state_corrected"
OUT_FOLDER = r"C:/Users/makss/Git/Galatsi-Semester-Project/Results/TXT_0314_with_parking_zone_timestamps_gap_1_2"

REF_IMG_PATH = r"C:/Users/makss/Git/Galatsi-Semester-Project/Results/Images/needed for script/frame_1_0004.png"
SRC_IMG_PATH = r"C:/Users/makss/Git/Galatsi-Semester-Project/Results/Images/needed for script/frame_1_0314.png"

OVERLAP_THR = 0.7        # overlap with parking mask to call parked
ZONE_OVERLAP_THR = 0.3   # minimum overlap with a zone to accept that zone
MIN_ZONE_AREA = 5000
MIN_PARKED_FRAMES = 3000 # min amount of time in a parking zone to be considered as parked

# If True, parked sequence must also stay in the SAME zone to keep one episode.
BREAK_EPISODE_WHEN_ZONE_CHANGES = True

# If timestamps are numeric, this is the max allowed gap to still consider the parking
# episode continuous. Example: if files are frame numbers, GAP=1 means consecutive files only.
MAX_TIMESTAMP_GAP = 1


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


def extract_timestamp_from_filename(path, fallback_idx=None):
    """
    Example:
        0001.txt -> 1
        0234.txt -> 234
        2298.txt -> 2298
    """
    stem = Path(path).stem
    try:
        return int(stem)
    except ValueError:
        return fallback_idx


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
# VEHICLE -> ZONE CLASSIFICATION
###########################################################
def classify_vehicle_parking_zone(row, H_mat, mask_bin, labels, overlap_thr, zone_overlap_thr):
    """
    Returns a dict with:
        parked (bool)
        parking_zone (int)   -> -1 if not parked
        parking_ratio (float)
        zone_ratio (float)   -> ratio for chosen/best zone
        best_zone_id (int)   -> 0 if none
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
        return {
            "parked": False,
            "parking_zone": -1,
            "parking_ratio": 0.0,
            "zone_ratio": 0.0,
            "best_zone_id": 0
        }

    res = rasterize_poly(pts_ref, W_img, H_img)
    if res is None:
        return {
            "parked": False,
            "parking_zone": -1,
            "parking_ratio": 0.0,
            "zone_ratio": 0.0,
            "best_zone_id": 0
        }

    xmin_c, xmax_c, ymin_c, ymax_c, veh_local = res
    veh_px = int(np.sum(veh_local))
    if veh_px == 0:
        return {
            "parked": False,
            "parking_zone": -1,
            "parking_ratio": 0.0,
            "zone_ratio": 0.0,
            "best_zone_id": 0
        }

    mask_roi = mask_bin[ymin_c:ymax_c+1, xmin_c:xmax_c+1]
    labels_roi = labels[ymin_c:ymax_c+1, xmin_c:xmax_c+1]

    parking_overlap_px = int(np.sum((veh_local == 1) & (mask_roi == 1)))
    parking_ratio = parking_overlap_px / veh_px

    # Count overlap with each zone inside the polygon
    zone_pixels = labels_roi[veh_local == 1]
    zone_pixels = zone_pixels[zone_pixels > 0]

    if len(zone_pixels) == 0:
        best_zone_id = 0
        zone_ratio = 0.0
    else:
        counts = np.bincount(zone_pixels)
        best_zone_id = int(np.argmax(counts))
        zone_overlap_px = int(counts[best_zone_id])
        zone_ratio = zone_overlap_px / veh_px

    state_val = str(row.get("state", "")).strip().lower()

    parked = (
        state_val == "stop"
        and parking_ratio > overlap_thr
        and best_zone_id > 0
        and zone_ratio > zone_overlap_thr
    )

    parking_zone = best_zone_id if parked else -1

    return {
        "parked": parked,
        "parking_zone": int(parking_zone),
        "parking_ratio": float(parking_ratio),
        "zone_ratio": float(zone_ratio),
        "best_zone_id": int(best_zone_id)
    }


###########################################################
# PARKING EPISODES
###########################################################
def is_numeric_timestamp(x):
    return isinstance(x, (int, np.integer, float, np.floating)) and pd.notna(x)


def same_or_close_timestamp(prev_t, cur_t, max_gap=1):
    if is_numeric_timestamp(prev_t) and is_numeric_timestamp(cur_t):
        return (cur_t - prev_t) <= max_gap
    return False


def assign_timestamp_in_out(df_all):
    """
    Adds timestamp_in and timestamp_out based on contiguous parked sequences
    per vehicle_id.

    NEW CONDITION:
    A vehicle is considered truly parked only if it stayed in the same
    parking place (here: same parking_zone) consecutively for at least
    MIN_PARKED_FRAMES frames.

    If a candidate parked segment is shorter than MIN_PARKED_FRAMES:
        - parking_zone is reset to -1
        - timestamp_in remains None
        - timestamp_out remains None
    """
    df_all = df_all.copy()
    df_all["timestamp_in"] = None
    df_all["timestamp_out"] = None

    for veh_id, g in df_all.groupby("vehicle_id", sort=False):
        g = g.sort_values(["time_order"]).copy()
        idxs = g.index.tolist()

        current_segment = []

        prev_time = None
        prev_zone = None
        prev_candidate_parked = False

        def close_segment(segment_idxs):
            """
            Validate the candidate parked segment.
            Keep it only if it lasts at least MIN_PARKED_FRAMES.
            """
            if len(segment_idxs) == 0:
                return

            seg_times = df_all.loc[segment_idxs, "timestamp"]
            t_in = int(seg_times.iloc[0])
            t_out = int(seg_times.iloc[-1])

            # Because timestamps are frame numbers and should be consecutive,
            # duration in frames is:
            duration_frames = t_out - t_in + 1

            if duration_frames >= MIN_PARKED_FRAMES:
                df_all.loc[segment_idxs, "timestamp_in"] = t_in
                df_all.loc[segment_idxs, "timestamp_out"] = t_out
            else:
                # Reject this parked segment
                df_all.loc[segment_idxs, "parking_zone"] = -1
                df_all.loc[segment_idxs, "timestamp_in"] = None
                df_all.loc[segment_idxs, "timestamp_out"] = None

        for idx in idxs:
            row = df_all.loc[idx]
            cur_candidate_parked = int(row["parking_zone"]) != -1
            cur_zone = int(row["parking_zone"])
            cur_time = row["timestamp"]

            start_new_segment = False

            if not cur_candidate_parked:
                start_new_segment = True
            else:
                if not prev_candidate_parked:
                    start_new_segment = True
                else:
                    contiguous = same_or_close_timestamp(
                        prev_time, cur_time, max_gap=MAX_TIMESTAMP_GAP
                    )

                    if not contiguous:
                        start_new_segment = True

                    if BREAK_EPISODE_WHEN_ZONE_CHANGES and cur_zone != prev_zone:
                        start_new_segment = True

            # If we must start a new segment, close the previous one
            if start_new_segment and len(current_segment) > 0:
                close_segment(current_segment)
                current_segment = []

            # Add current row if it is a candidate parked row
            if cur_candidate_parked:
                current_segment.append(idx)

            prev_time = cur_time
            prev_zone = cur_zone
            prev_candidate_parked = cur_candidate_parked

        # close final segment
        if len(current_segment) > 0:
            close_segment(current_segment)

    return df_all


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

    # ------------------------------------------------------
    # List TXT files
    # ------------------------------------------------------
    txt_files = sorted(glob.glob(os.path.join(FOLDER, "*.txt")))
    if len(txt_files) == 0:
        raise FileNotFoundError(f"No TXT files found in {FOLDER}")

    print(f"[INFO] Found {len(txt_files)} TXT files.")

    # ------------------------------------------------------
    # PASS 1: read all files, classify parking zone row by row
    # ------------------------------------------------------
    all_rows = []

    for file_order, txt_path in enumerate(tqdm(txt_files, desc="Pass 1 / classify")):
        df = pd.read_csv(txt_path, sep=";", engine="python")
        df.columns = [c.strip() for c in df.columns]

        timestamp_val = extract_timestamp_from_filename(txt_path, fallback_idx=file_order)

        for row_idx_in_file, (_, row) in enumerate(df.iterrows()):
            out = row.to_dict()

            cls = classify_vehicle_parking_zone(
                row=row,
                H_mat=H_mat,
                mask_bin=mask_bin,
                labels=labels,
                overlap_thr=OVERLAP_THR,
                zone_overlap_thr=ZONE_OVERLAP_THR
            )

            out["parking_zone"] = int(cls["parking_zone"])
            out["timestamp_in"] = None
            out["timestamp_out"] = None

            # internal bookkeeping columns
            out["_src_file"] = txt_path
            out["_src_filename"] = os.path.basename(txt_path)
            out["_row_in_file"] = row_idx_in_file
            out["timestamp"] = timestamp_val
            out["time_order"] = file_order

            all_rows.append(out)

    df_all = pd.DataFrame(all_rows)

    # ensure vehicle_id numeric if possible
    if "vehicle_id" in df_all.columns:
        try:
            df_all["vehicle_id"] = pd.to_numeric(df_all["vehicle_id"])
        except Exception:
            pass

    # ------------------------------------------------------
    # PASS 2: assign timestamp_in / timestamp_out per parked episode
    # ------------------------------------------------------
    df_all = assign_timestamp_in_out(df_all)

    # ------------------------------------------------------
    # PASS 3: save one TXT per original file
    # ------------------------------------------------------
    output_columns = [
        "vehicle_id",
        "veh_bb_x1", "veh_bb_y1",
        "veh_bb_x2", "veh_bb_y2",
        "veh_bb_x3", "veh_bb_y3",
        "veh_bb_x4", "veh_bb_y4",
        "det_class",
        "conf_score",
        "state",
        "parking_zone",
        "timestamp_in",
        "timestamp_out",
    ]

    total_rows = 0
    total_parked_rows = 0

    for txt_path in tqdm(txt_files, desc="Pass 3 / save"):
        filename = os.path.basename(txt_path)
        out_path = os.path.join(OUT_FOLDER, filename)

        sub = df_all[df_all["_src_filename"] == filename].copy()
        sub = sub.sort_values("_row_in_file")

        # Keep exactly the requested output columns that exist
        cols_to_save = [c for c in output_columns if c in sub.columns]
        sub_out = sub[cols_to_save].copy()

        # Optional: represent non-parked timestamps as None-like text
        sub_out["timestamp_in"] = sub_out["timestamp_in"].where(pd.notna(sub_out["timestamp_in"]), None)
        sub_out["timestamp_out"] = sub_out["timestamp_out"].where(pd.notna(sub_out["timestamp_out"]), None)

        sub_out.to_csv(out_path, sep=";", index=False)

        total_rows += len(sub_out)
        total_parked_rows += int((sub_out["parking_zone"] != -1).sum())

    print("\n================ DONE ================\n")
    print(f"Input folder               : {FOLDER}")
    print(f"Output folder              : {OUT_FOLDER}")
    print(f"TXT files processed        : {len(txt_files)}")
    print(f"Rows processed             : {total_rows}")
    print(f"Rows classified as parked  : {total_parked_rows}")