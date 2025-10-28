import cv2
import numpy as np
import pandas as pd
import glob, os
from tqdm import tqdm

# === CONFIG ===
mask_path = r"C:/Users/makss/Git/Galatsi-Semester-Project/Results/parking_detection/dwell_mult/test2_thr_0.8_2/parking_dwell_state_MULTI_REG_parking_location_mask.png"
folder    = r"C:/Users/makss/Git/Galatsi-Semester-Project/Results/TXT_0004"
overlap_thr = 0.3  # % of vehicle polygon inside parking area to call it "parked"

# === LOAD MASK ===
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
if mask is None:
    raise FileNotFoundError(f"Could not load mask image at {mask_path}")

mask_bin = (mask > 127).astype(np.uint8)
parking_area = int(np.sum(mask_bin))
H, W = mask_bin.shape

def process_frame_fast(txt_path):
    """
    returns stats rows (dicts) for: total / cars / big
    using one pass over the vehicles
    """

    df = pd.read_csv(txt_path, sep=';', engine='python')
    df.columns = [c.strip() for c in df.columns]  # strip column names

    # per-class bookkeeping
    classes = {
        "total": {
            "filter": lambda det_class: True,
            "union_mask": np.zeros_like(mask_bin, dtype=np.uint8),
            "veh_count_total": 0,
            "veh_count_parked": 0,
        },
        "cars": {
            "filter": lambda det_class: det_class == 10,
            "union_mask": np.zeros_like(mask_bin, dtype=np.uint8),
            "veh_count_total": 0,
            "veh_count_parked": 0,
        },
        "big": {
            "filter": lambda det_class: det_class == 9,
            "union_mask": np.zeros_like(mask_bin, dtype=np.uint8),
            "veh_count_total": 0,
            "veh_count_parked": 0,
        },
    }

    # loop over vehicles once
    for _, r in df.iterrows():
        det_class = int(r["det_class"])

        # polygon coords as int32
        poly = np.array(
            [(r[f"veh_bb_x{i}"], r[f"veh_bb_y{i}"]) for i in range(1,5)],
            dtype=np.int32
        )

        # make a tiny mask for THIS vehicle just to evaluate ratio
        veh_mask = np.zeros_like(mask_bin, dtype=np.uint8)
        cv2.fillPoly(veh_mask, [poly], 1)

        overlap_px = np.sum((veh_mask == 1) & (mask_bin == 1))
        veh_px     = np.sum(veh_mask)

        if veh_px == 0:
            parked_ratio = 0.0
        else:
            parked_ratio = overlap_px / veh_px

        # update each class bucket
        for bucket in classes.values():
            if bucket["filter"](det_class):
                bucket["veh_count_total"] += 1

                if parked_ratio > overlap_thr:
                    bucket["veh_count_parked"] += 1
                    # draw this vehicle ONCE into that class' union mask
                    cv2.fillPoly(bucket["union_mask"], [poly], 1)

    # after looping all vehicles, compute per-class stats
    out_rows = []
    frame_name = os.path.basename(txt_path)

    for cls_name, bucket in classes.items():
        total_veh   = bucket["veh_count_total"]
        parked_veh  = bucket["veh_count_parked"]

        # area covered for that class (all parked vehicles merged)
        covered_pixels = int(np.sum((bucket["union_mask"] == 1) & (mask_bin == 1)))

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
            "perc_in_parking": perc_in_parking,
            "parking_area_used_%": parking_area_used_pct,
            "parking_area_used_pixels": covered_pixels,
        })

    return out_rows


# === PROCESS ALL FRAMES WITH TQDM ===
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

    # guard in case a class doesn't appear at all
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

# === PRINT RESULT ===
print("\n================ SUMMARY ================\n")
for t, vals in summary.items():
    if not vals:
        print(f"--- {t.upper()} ---")
        print("No vehicles of this type found.\n")
        continue

    print(f"--- {t.upper()} ---")
    print(f"Average % vehicles parked      : {vals['avg_parking_use_%']:.2f}%")
    print(f"Average % parking area covered : {vals['avg_area_use_%']:.2f}%")
    print(f"Most occupied frame            : {vals['most_occupied_frame']} ({vals['max_area_%']:.2f}% area used)")
    print(f"Least occupied frame           : {vals['least_occupied_frame']} ({vals['min_area_%']:.2f}% area used)")
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
    print(f"% parking surface used         : {most['%_parking_surface_used']:.2f}%")
    print(f"Parking pixels covered         : {most['parking_surface_pixels_used']} px")

    least = d["LEAST_OCCUPIED"]
    print("\n[ LEAST OCCUPIED FRAME ]")
    print(f"Frame name                     : {least['frame']}")
    print(f"Vehicles detected              : {least['total_vehicles']}")
    print(f"Vehicles counted as parked     : {least['vehicles_in_parking']}")
    print(f"% vehicles parked              : {least['%_vehicles_in_parking']:.2f}%")
    print(f"% parking surface used         : {least['%_parking_surface_used']:.2f}%")
    print(f"Parking pixels covered         : {least['parking_surface_pixels_used']} px")

    print("\n---------------------------------------------------\n")
