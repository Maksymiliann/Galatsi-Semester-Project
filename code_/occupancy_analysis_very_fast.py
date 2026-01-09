import cv2
import numpy as np
import pandas as pd
import glob, os
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

"""
This script computes global parking occupancy metrics over time from per-frame vehicle detection polygons.

It loads a binary parking mask and processes all detection files (.txt) in a folder. For each vehicle polygon,
it quickly checks whether the vehicle center falls near the parking area, then computes the polygon overlap ratio
with the mask using a bounding-box ROI. A vehicle is counted as “parked” if overlap > OVERLAP_THR.

For each frame, the script outputs:
- total detected vehicles
- number and percentage of vehicles counted as parked
- parking surface usage, estimated by rasterizing all parked polygons into a union mask and counting covered
  parking pixels (pixels and % of total parking mask area)

Frames are processed in parallel with ProcessPoolExecutor, and a summary is printed including average occupancy
and the most/least occupied frames.
"""



###########################################################
# CONFIG (EDIT THESE PATHS)
###########################################################
MASK_PATH = r"C:/Users/makss/Git/Galatsi-Semester-Project/Results/parking_detection/dwell_mult/test2_thr_0.85/parking_dwell_state_MULTI_REG_parking_location_mask.png"
FOLDER    = r"C:/Users/makss/Git/Galatsi-Semester-Project/Results/TXT_0004"
OVERLAP_THR = 0.7  # % of vehicle polygon inside parking required to count as "parked" # 0.3

###########################################################
# GLOBALS FOR WORKERS
###########################################################
mask_bin_global = None
parking_area_global = None
H_global = None
W_global = None
overlap_thr_global = None


def init_worker(mask_bin, parking_area, overlap_thr):
    global mask_bin_global, parking_area_global, H_global, W_global, overlap_thr_global
    mask_bin_global = mask_bin.astype(np.uint8)
    parking_area_global = int(parking_area)
    H_global, W_global = mask_bin_global.shape
    overlap_thr_global = overlap_thr

###########################################################
# HELPER FUNCTIONS
###########################################################

def fast_bbox(poly_xy):
    x_coords = poly_xy[:, 0]
    y_coords = poly_xy[:, 1]
    xmin = int(np.floor(np.min(x_coords)))
    xmax = int(np.ceil (np.max(x_coords)))
    ymin = int(np.floor(np.min(y_coords)))
    ymax = int(np.ceil (np.max(y_coords)))
    xmin_c = max(xmin, 0)
    ymin_c = max(ymin, 0)
    xmax_c = min(xmax, W_global - 1)
    ymax_c = min(ymax, H_global - 1)
    return xmin_c, xmax_c, ymin_c, ymax_c


def quick_candidate_check(poly_xy):
    cx = float(np.mean(poly_xy[:,0]))
    cy = float(np.mean(poly_xy[:,1]))
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


def compute_overlap_ratio(poly_xy):
    xmin_c, xmax_c, ymin_c, ymax_c = fast_bbox(poly_xy)
    if xmax_c < xmin_c or ymax_c < ymin_c:
        return 0.0
    shifted_poly = poly_xy.copy()
    shifted_poly[:, 0] -= xmin_c
    shifted_poly[:, 1] -= ymin_c
    shifted_poly = shifted_poly.astype(np.int32)
    roi_w = xmax_c - xmin_c + 1
    roi_h = ymax_c - ymin_c + 1
    veh_local = np.zeros((roi_h, roi_w), dtype=np.uint8)
    cv2.fillPoly(veh_local, [shifted_poly], 1)
    mask_roi = mask_bin_global[ymin_c:ymax_c+1, xmin_c:xmax_c+1]
    veh_px = int(np.sum(veh_local))
    if veh_px == 0:
        return 0.0
    overlap_px = int(np.sum((veh_local == 1) & (mask_roi == 1)))
    return overlap_px / veh_px


def rasterize_into_union(poly_xy, union_mask):
    xmin_c, xmax_c, ymin_c, ymax_c = fast_bbox(poly_xy)
    if xmax_c < xmin_c or ymax_c < ymin_c:
        return
    shifted_poly = poly_xy.copy()
    shifted_poly[:, 0] -= xmin_c
    shifted_poly[:, 1] -= ymin_c
    shifted_poly = shifted_poly.astype(np.int32)
    roi_w = xmax_c - xmin_c + 1
    roi_h = ymax_c - ymin_c + 1
    temp = np.zeros((roi_h, roi_w), dtype=np.uint8)
    cv2.fillPoly(temp, [shifted_poly], 1)
    union_mask[ymin_c:ymax_c+1, xmin_c:xmax_c+1] |= temp

###########################################################
# PER-FRAME WORKER
###########################################################

def process_one_frame(txt_path):
    df = pd.read_csv(txt_path, sep=';', engine='python')
    df.columns = [c.strip() for c in df.columns]

    union_total = np.zeros((H_global, W_global), dtype=np.uint8)
    total_count = 0
    parked_count = 0

    for _, r in df.iterrows():
        poly_xy = np.array([
            (r["veh_bb_x1"], r["veh_bb_y1"]),
            (r["veh_bb_x2"], r["veh_bb_y2"]),
            (r["veh_bb_x3"], r["veh_bb_y3"]),
            (r["veh_bb_x4"], r["veh_bb_y4"]),
        ], dtype=np.float32)

        total_count += 1

        if not quick_candidate_check(poly_xy):
            continue

        ratio = compute_overlap_ratio(poly_xy)
        if ratio > overlap_thr_global:
            parked_count += 1
            rasterize_into_union(poly_xy, union_total)

    # Compute stats
    frame_name = os.path.basename(txt_path)
    perc_in_parking = 100.0 * parked_count / total_count if total_count > 0 else 0.0
    covered_pixels = int(np.sum((union_total == 1) & (mask_bin_global == 1)))
    parking_area_used_pct = 100.0 * covered_pixels / parking_area_global if parking_area_global > 0 else 0.0

    return {
        "frame": frame_name,
        "total_vehicles": int(total_count),
        "vehicles_in_parking": int(parked_count),
        "perc_in_parking": perc_in_parking,
        "parking_area_used_%": parking_area_used_pct,
        "parking_area_used_pixels": covered_pixels,
    }

###########################################################
# MAIN
###########################################################

if __name__ == "__main__":
    # Load mask once
    mask_img = cv2.imread(MASK_PATH, cv2.IMREAD_GRAYSCALE)
    if mask_img is None:
        raise FileNotFoundError(f"Could not load mask image at {MASK_PATH}")

    mask_bin = (mask_img > 127).astype(np.uint8)
    H, W = mask_bin.shape
    parking_area = int(np.sum(mask_bin))

    # Collect frames
    frames = sorted(glob.glob(os.path.join(FOLDER, "*.txt")))

    # Uncomment below for dev mode (process every 5th frame)
    # frames = frames[::5]

    # Parallel processing
    results = []
    with ProcessPoolExecutor() as pool:
        pool._initializer = init_worker
        pool._initargs = (mask_bin, parking_area, OVERLAP_THR)
        for frame_stats in tqdm(pool.map(process_one_frame, frames),
                                total=len(frames),
                                desc="Processing frames in parallel"):
            results.append(frame_stats)

    df = pd.DataFrame(results)

    # Compute global stats
    idx_max = df["parking_area_used_%"].idxmax()
    idx_min = df["parking_area_used_%"].idxmin()
    row_max = df.loc[idx_max]
    row_min = df.loc[idx_min]

    avg_parking_use = df["perc_in_parking"].mean()
    avg_area_use = df["parking_area_used_%"].mean()

    print("\n================ SUMMARY ================\n")
    print(f"Average % vehicles parked      : {avg_parking_use:.2f}%")
    print(f"Average % parking area covered : {avg_area_use:.4f}%")
    print(f"Most occupied frame            : {row_max['frame']} ({row_max['parking_area_used_%']:.4f}% area used)")
    print(f"Least occupied frame           : {row_min['frame']} ({row_min['parking_area_used_%']:.4f}% area used)\n")

    print("============ DETAILS ============\n")
    print(f"[ MOST OCCUPIED FRAME ]")
    print(f"Frame name                     : {row_max['frame']}")
    print(f"Vehicles detected              : {row_max['total_vehicles']}")
    print(f"Vehicles parked                : {row_max['vehicles_in_parking']}")
    print(f"% vehicles parked              : {row_max['perc_in_parking']:.2f}%")
    print(f"% parking surface used         : {row_max['parking_area_used_%']:.4f}%")
    print(f"Parking pixels covered         : {row_max['parking_area_used_pixels']} px\n")

    print(f"[ LEAST OCCUPIED FRAME ]")
    print(f"Frame name                     : {row_min['frame']}")
    print(f"Vehicles detected              : {row_min['total_vehicles']}")
    print(f"Vehicles parked                : {row_min['vehicles_in_parking']}")
    print(f"% vehicles parked              : {row_min['perc_in_parking']:.2f}%")
    print(f"% parking surface used         : {row_min['parking_area_used_%']:.4f}%")
    print(f"Parking pixels covered         : {row_min['parking_area_used_pixels']} px\n")
