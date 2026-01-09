import cv2
import numpy as np
import pandas as pd
import glob, os
from tqdm import tqdm

"""
This script computes parking occupancy over time from per-frame vehicle polygons and a binary parking mask.

For each detection file (.txt), it builds a vehicle polygon, computes its overlap ratio with the parking mask,
and counts the vehicle as “parked” if the ratio exceeds overlap_thr. The same logic is applied to three groups:
- total: all vehicles
- cars: det_class == 10
- big:  det_class == 9

To speed up processing, polygons are rasterized only inside their bounding-box ROI (instead of the full image),
both for overlap computation and for updating a per-class union mask of parked vehicles. The union mask is used
to estimate how much of the parking area is covered (pixels and % of the total parking mask).

After processing all frames, results are stored in a DataFrame and the script prints, per group:
average % of parked vehicles, average % of parking area covered, and the most/least occupied frames with details.
"""



# === CONFIG ===
mask_path = r"C:/Users/makss/Git/Galatsi-Semester-Project/Results/parking_detection/dwell_mult/test2_thr_0.8_2/parking_dwell_state_MULTI_REG_parking_location_mask.png"
folder    = r"C:/Users/makss/Git/Galatsi-Semester-Project/Results/TXT_0004"
overlap_thr = 0.3  # % vehicle area that must lie on parking pixels to consider it "parked"

# === LOAD MASK ===
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
if mask is None:
    raise FileNotFoundError(f"Could not load mask image at {mask_path}")

mask_bin = (mask > 127).astype(np.uint8)  # 1 where parking is valid
H, W = mask_bin.shape
parking_area = int(np.sum(mask_bin))


def rasterize_polygon_into_roi(poly_xy, canvas_mask, value=1):
    """
    poly_xy: np.array([[x1,y1],[x2,y2],...]], float or int
    canvas_mask: (H,W) uint8 full-frame union mask for that class.
    value: what to fill.

    We'll:
    - get the tight bbox of the polygon
    - create a small temp mask of that bbox size
    - draw the polygon shifted into that local coords
    - OR that temp mask back into the canvas_mask's ROI
    """

    # compute bbox in integer pixel coords
    x_coords = poly_xy[:,0]
    y_coords = poly_xy[:,1]

    xmin = int(np.floor(np.min(x_coords)))
    xmax = int(np.ceil (np.max(x_coords)))
    ymin = int(np.floor(np.min(y_coords)))
    ymax = int(np.ceil (np.max(y_coords)))

    # clamp to image bounds to avoid out-of-range
    xmin_clamped = max(xmin, 0)
    ymin_clamped = max(ymin, 0)
    xmax_clamped = min(xmax, W-1)
    ymax_clamped = min(ymax, H-1)

    if xmax_clamped < xmin_clamped or ymax_clamped < ymin_clamped:
        return  # polygon is fully out of frame? skip safely

    # shift polygon to local ROI coords
    shifted_poly = poly_xy.copy()
    shifted_poly[:,0] -= xmin_clamped
    shifted_poly[:,1] -= ymin_clamped
    shifted_poly = shifted_poly.astype(np.int32)

    roi_w = xmax_clamped - xmin_clamped + 1
    roi_h = ymax_clamped - ymin_clamped + 1

    # temp mask just for this polygon
    temp = np.zeros((roi_h, roi_w), dtype=np.uint8)
    cv2.fillPoly(temp, [shifted_poly], value)

    # OR it into the canvas union mask ROI
    canvas_mask[ymin_clamped:ymax_clamped+1,
                xmin_clamped:xmax_clamped+1] |= temp


def compute_overlap_ratio(poly_xy, mask_bin):
    """
    Decide if a vehicle counts as 'parked':
    ratio = (# pixels of vehicle that fall on parking) / (# pixels of vehicle)

    We'll do this with ROI logic too (fast).
    """

    x_coords = poly_xy[:,0]
    y_coords = poly_xy[:,1]

    xmin = int(np.floor(np.min(x_coords)))
    xmax = int(np.ceil (np.max(x_coords)))
    ymin = int(np.floor(np.min(y_coords)))
    ymax = int(np.ceil (np.max(y_coords)))

    xmin_c = max(xmin, 0)
    ymin_c = max(ymin, 0)
    xmax_c = min(xmax, mask_bin.shape[1]-1)
    ymax_c = min(ymax, mask_bin.shape[0]-1)

    if xmax_c < xmin_c or ymax_c < ymin_c:
        return 0.0  # completely off screen

    shifted_poly = poly_xy.copy()
    shifted_poly[:,0] -= xmin_c
    shifted_poly[:,1] -= ymin_c
    shifted_poly = shifted_poly.astype(np.int32)

    roi_w = xmax_c - xmin_c + 1
    roi_h = ymax_c - ymin_c + 1

    veh_local = np.zeros((roi_h, roi_w), dtype=np.uint8)
    cv2.fillPoly(veh_local, [shifted_poly], 1)

    # compare only in that roi
    mask_roi = mask_bin[ymin_c:ymax_c+1, xmin_c:xmax_c+1]

    veh_px = int(np.sum(veh_local))
    if veh_px == 0:
        return 0.0

    overlap_px = int(np.sum((veh_local == 1) & (mask_roi == 1)))

    return overlap_px / veh_px


def process_frame_fast(txt_path):
    """
    One pass for the frame.
    We build 3 union masks: total / cars / big.
    We do ROI-based rasterization to reduce cost.
    """

    df = pd.read_csv(txt_path, sep=';', engine='python')
    df.columns = [c.strip() for c in df.columns]

    classes = {
        "total": {
            "check": lambda det_class: True,
            "union_mask": np.zeros((H,W), dtype=np.uint8),
            "veh_total": 0,
            "veh_parked": 0,
        },
        "cars": {
            "check": lambda det_class: det_class == 10,
            "union_mask": np.zeros((H,W), dtype=np.uint8),
            "veh_total": 0,
            "veh_parked": 0,
        },
        "big": {
            "check": lambda det_class: det_class == 9,
            "union_mask": np.zeros((H,W), dtype=np.uint8),
            "veh_total": 0,
            "veh_parked": 0,
        },
    }

    # iterate vehicles
    for _, r in df.iterrows():
        det_class = int(r["det_class"])

        poly_xy = np.array([
            (r["veh_bb_x1"], r["veh_bb_y1"]),
            (r["veh_bb_x2"], r["veh_bb_y2"]),
            (r["veh_bb_x3"], r["veh_bb_y3"]),
            (r["veh_bb_x4"], r["veh_bb_y4"]),
        ], dtype=np.float32)

        ratio = compute_overlap_ratio(poly_xy, mask_bin)
        is_parked = ratio > overlap_thr

        # update the 3 buckets
        for bucket in classes.values():
            if bucket["check"](det_class):
                bucket["veh_total"] += 1
                if is_parked:
                    bucket["veh_parked"] += 1
                    # paint into union mask using ROI only
                    rasterize_polygon_into_roi(poly_xy, bucket["union_mask"], value=1)

    # after all vehicles, compute stats for each bucket
    out_rows = []
    frame_name = os.path.basename(txt_path)

    for cls_name, bucket in classes.items():
        total_veh = bucket["veh_total"]
        parked_veh = bucket["veh_parked"]
        union_mask = bucket["union_mask"]

        # area actually covered in parking area
        covered_pixels = int(np.sum((union_mask == 1) & (mask_bin == 1)))

        if total_veh > 0:
            perc_in_parking = 100.0 * parked_veh / total_veh
        else:
            perc_in_parking = 0.0

        if parking_area > 0:
            parking_area_used_pct = 100.0 * covered_pixels / parking_area
        else:
            parking_area_used_pct = 0.0

        out_rows.append({
            "frame": frame_name,
            "type": cls_name,
            "total_vehicles": int(total_veh),
            "vehicles_in_parking": int(parked_veh),
            "perc_in_parking": float(perc_in_parking),
            "parking_area_used_%": float(parking_area_used_pct),
            "parking_area_used_pixels": int(covered_pixels),
        })

    return out_rows


# === PROCESS ALL FRAMES ===
frames = sorted(glob.glob(os.path.join(folder, "*.txt")))
all_stats = []

for fpath in tqdm(frames, desc="Processing frames"):
    frame_stats = process_frame_fast(fpath)
    all_stats.extend(frame_stats)

df = pd.DataFrame(all_stats)

# === SUMMARY / EXTREMES ===
summary = {}
details = {}

for cls in ["total", "cars", "big"]:
    sub = df[df["type"] == cls]

    if len(sub) == 0:
        summary[cls] = {}
        details[cls] = {}
        continue

    idx_max = sub["parking_area_used_%"].idxmax()
    idx_min = sub["parking_area_used_%"].idxmin()

    row_max = sub.loc[idx_max]
    row_min = sub.loc[idx_min]

    summary[cls] = {
        "avg_parking_use_%": sub["perc_in_parking"].mean(),
        "avg_area_use_%": sub["parking_area_used_%"].mean(),
        "most_occupied_frame": row_max["frame"],
        "max_area_%": row_max["parking_area_used_%"],
        "least_occupied_frame": row_min["frame"],
        "min_area_%": row_min["parking_area_used_%"],
    }

    details[cls] = {
        "MOST_OCCUPIED": {
            "frame": row_max["frame"],
            "total_vehicles": int(row_max["total_vehicles"]),
            "vehicles_in_parking": int(row_max["vehicles_in_parking"]),
            "%_vehicles_in_parking": float(row_max["perc_in_parking"]),
            "%_parking_surface_used": float(row_max["parking_area_used_%"]),
            "parking_surface_pixels_used": int(row_max["parking_area_used_pixels"]),
        },
        "LEAST_OCCUPIED": {
            "frame": row_min["frame"],
            "total_vehicles": int(row_min["total_vehicles"]),
            "vehicles_in_parking": int(row_min["vehicles_in_parking"]),
            "%_vehicles_in_parking": float(row_min["perc_in_parking"]),
            "%_parking_surface_used": float(row_min["parking_area_used_%"]),
            "parking_surface_pixels_used": int(row_min["parking_area_used_pixels"]),
        }
    }

print("\n================ SUMMARY ================\n")
for t, vals in summary.items():
    if not vals:
        print(f"--- {t.upper()} ---")
        print("No vehicles of this type found.\n")
        continue

    print(f"--- {t.upper()} ---")
    print(f"Average % vehicles parked      : {vals['avg_parking_use_%']:.2f}%")
    print(f"Average % parking area covered : {vals['avg_area_use_%']:.4f}%")
    print(f"Most occupied frame            : {vals['most_occupied_frame']} ({vals['max_area_%']:.4f}% area used)")
    print(f"Least occupied frame           : {vals['least_occupied_frame']} ({vals['min_area_%']:.4f}% area used)")
    print()

print("\n============ DETAILS PER EXTREME FRAME ============\n")
for t, d in details.items():
    if not d:
        print(f"### {t.upper()} ###")
        print("No data.\n")
        continue

    print(f"### {t.upper()} ###")

    most = d["MOST_OCCUPIED"]
    print("\n[ MOST OCCUPIED FRAME ]")
    print(f"Frame name                     : {most['frame']}")
    print(f"Vehicles detected              : {most['total_vehicles']}")
    print(f"Vehicles counted as parked     : {most['vehicles_in_parking']}")
    print(f"% vehicles parked              : {most['%_vehicles_in_parking']:.2f}%")
    print(f"% parking surface used         : {most['%_parking_surface_used']:.4f}%")
    print(f"Parking pixels covered         : {most['parking_surface_pixels_used']} px")

    least = d["LEAST_OCCUPIED"]
    print("\n[ LEAST OCCUPIED FRAME ]")
    print(f"Frame name                     : {least['frame']}")
    print(f"Vehicles detected              : {least['total_vehicles']}")
    print(f"Vehicles counted as parked     : {least['vehicles_in_parking']}")
    print(f"% vehicles parked              : {least['%_vehicles_in_parking']:.2f}%")
    print(f"% parking surface used         : {least['%_parking_surface_used']:.4f}%")
    print(f"Parking pixels covered         : {least['parking_surface_pixels_used']} px")

    print("\n---------------------------------------------------\n")
