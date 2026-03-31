import cv2
import numpy as np
import pandas as pd
import glob, os
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt

"""
Compute global and per-zone parking occupancy in REF coordinates (mask/zones),
WITHOUT saving CSVs.

"Parked candidates" = detections with:
  - state == "stop"
  - overlap(parking mask) > OVERLAP_THR

This version filters "true parked" vehicles by DURATION:
  - keep only vehicle_ids present in parked-candidates for >= MIN_FRAMES_PRESENT frames

Then it recomputes global % and per-zone time series using only those long-present vehicles
and plots results (optionally with rolling-mean smoothing).
"""


###########################################################
# CONFIG
###########################################################
MASK_PATH   = r"C:/Users/makss/Git/Galatsi-Semester-Project/Results/occupancy_analysis/per_zones/mask_closed_cleaned.png"
FOLDER      = r"C:/Users/makss/Git/Galatsi-Semester-Project/Results/TXT_0314_D2_S4_S1_padded_state_corrected"
ZONES_PATH  = r"C:/Users/makss/Git/Galatsi-Semester-Project/Results/occupancy_analysis/per_zones/mask_zones_id_cleaned.png"

OVERLAP_THR = 0.6
MIN_ZONE_AREA = 5000

USE_PARALLEL = True      # start False, then True if you want
MAX_WORKERS  = 4

REF_IMG_PATH = r"C:/Users/makss/Git/Galatsi-Semester-Project//Results/Images/needed for script/frame_1_0004.png"
SRC_IMG_PATH = r"C:/Users/makss/Git/Galatsi-Semester-Project//Results/Images/needed for script/frame_1_0314.png"

# --- NEW FILTER ---
MIN_FRAMES_PRESENT = 3500  # keep vehicle_id if it appears in >= this many frames (among parked-candidates)

# --- Plot smoothing ---
SMOOTH_WINDOWS = (50, 150)   # set () to disable smoothing


###########################################################
# GLOBALS FOR WORKERS
###########################################################
mask_bin_global = None
H_global = None
W_global = None
overlap_thr_global = None

labels_global = None
zone_ids_global = None
zone_areas_global = None

H_mat_global = None


def init_worker(mask_bin, labels, zone_ids, zone_areas, overlap_thr, H_mat):
    global mask_bin_global, labels_global, zone_ids_global, zone_areas_global
    global H_global, W_global, overlap_thr_global, H_mat_global

    mask_bin_global = mask_bin.astype(np.uint8)
    labels_global = labels.astype(np.int32)
    zone_ids_global = list(zone_ids)
    zone_areas_global = dict(zone_areas)
    H_global, W_global = mask_bin_global.shape
    overlap_thr_global = overlap_thr
    H_mat_global = H_mat.astype(np.float32) if H_mat is not None else np.eye(3, dtype=np.float32)


###########################################################
# HELPERS
###########################################################
def infer_vehicle_id_column(df):
    candidates = ["vehicle_id", "veh_id", "track_id", "id", "obj_id", "vehicleid", "trackid"]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        f"No vehicle id column found. Tried {candidates}. Found columns: {list(df.columns)}"
    )

def fast_bbox(poly_xy):
    x_coords = poly_xy[:, 0]
    y_coords = poly_xy[:, 1]
    xmin = int(np.floor(np.min(x_coords)))
    xmax = int(np.ceil(np.max(x_coords)))
    ymin = int(np.floor(np.min(y_coords)))
    ymax = int(np.ceil(np.max(y_coords)))
    xmin_c = max(xmin, 0)
    ymin_c = max(ymin, 0)
    xmax_c = min(xmax, W_global - 1)
    ymax_c = min(ymax, H_global - 1)
    return xmin_c, xmax_c, ymin_c, ymax_c

def quick_candidate_check(poly_xy):
    cx = float(np.mean(poly_xy[:, 0]))
    cy = float(np.mean(poly_xy[:, 1]))
    cx_i = int(round(cx))
    cy_i = int(round(cy))
    if cx_i < 0 or cx_i >= W_global or cy_i < 0 or cy_i >= H_global:
        return False
    x0 = max(cx_i - 2, 0)
    y0 = max(cy_i - 2, 0)
    x1 = min(cx_i + 2, W_global - 1)
    y1 = min(cy_i + 2, H_global - 1)
    local_mask = mask_bin_global[y0:y1+1, x0:x1+1]
    return np.any(local_mask == 1)

def rasterize_poly(poly_xy):
    xmin_c, xmax_c, ymin_c, ymax_c = fast_bbox(poly_xy)
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
        return np.eye(3, dtype=np.float32), 0, float('inf')

    bf = cv2.BFMatcher(norm)
    knn = bf.knnMatch(d2, d1, k=2)
    good = [m for m, n in knn if n is not None and m.distance < ratio * n.distance]
    if len(good) < 4:
        return np.eye(3, dtype=np.float32), 0, float('inf')

    pts2 = np.float32([k2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts1 = np.float32([k1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, ransac_thr)
    if H is None:
        return np.eye(3, dtype=np.float32), 0, float('inf')

    inl = mask.ravel().astype(bool)
    err = float('inf')
    if inl.any():
        p2 = pts2[inl].reshape(-1, 2)
        p1 = pts1[inl].reshape(-1, 2)
        p2h = np.hstack([p2, np.ones((len(p2), 1), np.float32)])
        proj = (H @ p2h.T).T
        proj = proj[:, :2] / proj[:, 2:3]
        err = float(np.mean(np.linalg.norm(proj - p1, axis=1)))

    return H.astype(np.float32), int(inl.sum()), err


###########################################################
# PER-FRAME WORKER
###########################################################
def process_one_frame(txt_path):
    df = pd.read_csv(txt_path, sep=';', engine='python')
    df.columns = [c.strip() for c in df.columns]

    veh_id_col = infer_vehicle_id_column(df)
    frame_name = os.path.basename(txt_path)

    total_count_all = 0
    parked_records = []  # candidates for duration filter

    for _, r in df.iterrows():
        total_count_all += 1

        state_val = str(r.get("state", "")).strip().lower()
        if state_val != "stop":
            continue

        poly_src = np.array([
            (r["veh_bb_x1"], r["veh_bb_y1"]),
            (r["veh_bb_x2"], r["veh_bb_y2"]),
            (r["veh_bb_x3"], r["veh_bb_y3"]),
            (r["veh_bb_x4"], r["veh_bb_y4"]),
        ], dtype=np.float32)

        pts = poly_src.reshape(-1, 1, 2)
        pts_ref = cv2.perspectiveTransform(pts, H_mat_global).reshape(-1, 2).astype(np.float32)
        if not np.isfinite(pts_ref).all():
            continue

        poly_xy = pts_ref
        if not quick_candidate_check(poly_xy):
            continue

        res = rasterize_poly(poly_xy)
        if res is None:
            continue
        xmin_c, xmax_c, ymin_c, ymax_c, veh_local = res

        mask_roi = mask_bin_global[ymin_c:ymax_c+1, xmin_c:xmax_c+1]
        veh_px = int(np.sum(veh_local))
        if veh_px == 0:
            continue

        overlap_px = int(np.sum((veh_local == 1) & (mask_roi == 1)))
        ratio = overlap_px / veh_px

        if ratio > overlap_thr_global:
            # zone assignment (majority vote)
            zid_assigned = 0
            lab_roi = labels_global[ymin_c:ymax_c+1, xmin_c:xmax_c+1]
            values = lab_roi[veh_local == 1]
            if values.size > 0:
                counts = np.bincount(values, minlength=labels_global.max()+1)
                counts[0] = 0
                if counts.sum() > 0:
                    zid_assigned = int(np.argmax(counts))

            vid = str(r.get(veh_id_col, "")).strip()
            frame_idx = int(Path(frame_name).stem)

            parked_records.append({
                "frame": frame_name,
                "frame_idx": frame_idx,
                "vehicle_id": vid,
                "zone_id": int(zid_assigned),
            })

    return {
        "frame": frame_name,
        "total_vehicles": int(total_count_all),
        "parked_records": parked_records,
    }


###########################################################
# PLOTTING
###########################################################
def add_rolling(df, col, windows):
    for w in windows:
        df[f"{col}_rm{w}"] = df[col].rolling(window=w, min_periods=1).mean()
    return df

def plot_global(df_global, windows=SMOOTH_WINDOWS):
    df = df_global.sort_values("frame_idx").copy()

    plt.figure()
    plt.plot(df["frame_idx"], df["perc_in_parking_truly"], linewidth=1, label="true parked %")

    if windows:
        df = add_rolling(df, "perc_in_parking_truly", windows)
        for w in windows:
            plt.plot(df["frame_idx"], df[f"perc_in_parking_truly_rm{w}"], linewidth=2, label=f"RM{w}")

    plt.xlabel("Frame index")
    plt.ylabel("Parked vehicles (%)")
    plt.title(f"Global % parked (vehicle present ≥ {MIN_FRAMES_PRESENT} frames)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_zone(dz_tp, zone_id, windows=SMOOTH_WINDOWS):
    z = dz_tp[dz_tp["zone_id"] == zone_id].sort_values("frame_idx").copy()
    if z.empty:
        print(f"No data for zone {zone_id}")
        return

    plt.figure()
    plt.plot(z["frame_idx"], z["vehicles_in_zone"], linewidth=1, label="raw")

    if windows:
        z = z.set_index("frame_idx")
        for w in windows:
            sm = z["vehicles_in_zone"].rolling(window=w, min_periods=1).mean()
            plt.plot(sm.index, sm.values, linewidth=2, label=f"RM{w}")

    plt.xlabel("Frame index")
    plt.ylabel("Parked vehicles in zone")
    plt.title(f"Zone {zone_id}: parked vehicles over time (present ≥ {MIN_FRAMES_PRESENT} frames)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_all_zones(dz_tp):
    pivot = dz_tp.pivot(index="frame_idx", columns="zone_id", values="vehicles_in_zone").sort_index()

    plt.figure()
    for zid in pivot.columns:
        plt.plot(pivot.index, pivot[zid], linewidth=1, label=f"Zone {zid}")

    plt.xlabel("Frame index")
    plt.ylabel("Parked vehicles")
    plt.title(f"All zones: parked vehicles over time (present ≥ {MIN_FRAMES_PRESENT} frames)")
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()


###########################################################
# MAIN
###########################################################
if __name__ == "__main__":
    # Load mask
    mask_img = cv2.imread(MASK_PATH, cv2.IMREAD_GRAYSCALE)
    if mask_img is None:
        raise FileNotFoundError(f"Could not load mask image at {MASK_PATH}")
    mask_bin = (mask_img > 127).astype(np.uint8)

    # Homography SRC -> REF
    img_ref = cv2.imread(REF_IMG_PATH, cv2.IMREAD_COLOR)
    if img_ref is None:
        raise FileNotFoundError(f"Could not load reference image at {REF_IMG_PATH}")

    img_src = cv2.imread(SRC_IMG_PATH, cv2.IMREAD_COLOR)
    if img_src is None:
        raise FileNotFoundError(f"Could not load source image at {SRC_IMG_PATH}")

    if os.path.abspath(REF_IMG_PATH) == os.path.abspath(SRC_IMG_PATH):
        H_mat = np.eye(3, dtype=np.float32)
        print("[H] REF == SRC -> homography identity.")
    else:
        H_mat, ninl, err = estimate_homography(img_ref, img_src)
        print(f"[H] Estimated SRC -> REF: inliers={ninl}, reproj_err={err:.3f}px")

    # Zones labels
    if ZONES_PATH and Path(ZONES_PATH).exists():
        lab_img = cv2.imread(ZONES_PATH, cv2.IMREAD_GRAYSCALE)
        if lab_img is None:
            raise FileNotFoundError(ZONES_PATH)
        labels = lab_img.astype(np.int32)
        labels[mask_bin == 0] = 0
    else:
        num, lab = cv2.connectedComponents(mask_bin, connectivity=8)
        labels = lab.astype(np.int32)

    # Filter small zones
    if MIN_ZONE_AREA and MIN_ZONE_AREA > 0:
        vals = labels.ravel()
        counts = np.bincount(vals)
        num = len(counts) - 1

        keep = np.zeros(num + 1, np.uint8)
        keep[0] = 1
        for z in range(1, num + 1):
            if counts[z] >= MIN_ZONE_AREA:
                keep[z] = 1

        remap = np.zeros(num + 1, np.int32)
        nid = 0
        for z in range(num + 1):
            if keep[z]:
                remap[z] = nid
                nid += 1

        labels = remap[labels]
        labels[mask_bin == 0] = 0

    # Zones meta
    vals = labels.ravel()
    counts = np.bincount(vals)
    zone_ids = [z for z in range(1, len(counts)) if counts[z] > 0]
    zone_areas = {int(z): int(counts[z]) for z in zone_ids}

    frames = sorted(glob.glob(os.path.join(FOLDER, "*.txt")))
    if len(frames) == 0:
        raise FileNotFoundError(f"No .txt files found in {FOLDER}")

    results = []

    if USE_PARALLEL:
        with ProcessPoolExecutor(
            max_workers=MAX_WORKERS,
            initializer=init_worker,
            initargs=(mask_bin, labels, zone_ids, zone_areas, OVERLAP_THR, H_mat)
        ) as pool:
            for frame_stats in tqdm(
                pool.map(process_one_frame, frames),
                total=len(frames),
                desc="Processing frames in parallel"
            ):
                results.append(frame_stats)
    else:
        init_worker(mask_bin, labels, zone_ids, zone_areas, OVERLAP_THR, H_mat)
        for fp in tqdm(frames, total=len(frames), desc="Processing frames sequentially"):
            results.append(process_one_frame(fp))

    # Build candidates dataframe
    cand_rows = []
    for r in results:
        cand_rows.extend(r["parked_records"])
    cand = pd.DataFrame(cand_rows)

    if cand.empty:
        raise RuntimeError("No parked candidates found (STOP+overlap). Check OVERLAP_THR / state / homography.")

    # Count how many frames each vehicle appears in (among parked candidates)
    counts = cand.groupby("vehicle_id")["frame_idx"].nunique().reset_index(name="n_frames")
    keep_ids = set(counts.loc[counts["n_frames"] >= MIN_FRAMES_PRESENT, "vehicle_id"].astype(str))

    print(f"\nParked candidates vehicle_ids: {cand['vehicle_id'].nunique()}")
    print(f"Keeping vehicle_ids with n_frames >= {MIN_FRAMES_PRESENT}: {len(keep_ids)}")

    cand_tp = cand[cand["vehicle_id"].isin(keep_ids)].copy()

    # Global per frame
    df_global = pd.DataFrame([{
        "frame": r["frame"],
        "total_vehicles": r["total_vehicles"],
    } for r in results])

    tp_per_frame = cand_tp.groupby("frame")["vehicle_id"].nunique().reset_index()
    tp_per_frame = tp_per_frame.rename(columns={"vehicle_id": "vehicles_in_parking_truly"})

    df_global = df_global.merge(tp_per_frame, on="frame", how="left")
    df_global["vehicles_in_parking_truly"] = df_global["vehicles_in_parking_truly"].fillna(0).astype(int)

    df_global["perc_in_parking_truly"] = np.where(
        df_global["total_vehicles"] > 0,
        100.0 * df_global["vehicles_in_parking_truly"] / df_global["total_vehicles"],
        0.0
    )

    df_global["frame_idx"] = df_global["frame"].astype(str).str.replace(".txt", "", regex=False).astype(int)
    df_global = df_global.sort_values("frame_idx")

    # Per-zone per-frame
    dz_tp = (
        cand_tp.groupby(["frame_idx", "zone_id"])["vehicle_id"]
        .nunique()
        .reset_index()
        .rename(columns={"vehicle_id": "vehicles_in_zone"})
        .sort_values(["frame_idx", "zone_id"])
    )

    # Plots
    plot_global(df_global, windows=SMOOTH_WINDOWS)
    plot_zone(dz_tp, zone_id=1, windows=SMOOTH_WINDOWS)
    plot_all_zones(dz_tp)