import cv2
import numpy as np
import pandas as pd
from shapely.geometry import Polygon
import glob, os
from tqdm import tqdm   # <- make sure you have `pip install tqdm`

"""
This script estimates parking occupancy from per-frame detection polygons and a binary parking mask.

It loads a grayscale parking mask, binarizes it, and treats white pixels as valid parking area. For each frame
(.txt file), it reads vehicle detections and builds a 4-corner polygon for each vehicle. The polygon is rasterized
into a temporary mask to measure how many vehicle pixels overlap the parking mask. A vehicle is counted as “parked”
if the overlap ratio (overlap_pixels / vehicle_pixels) exceeds overlap_thr.

The same computation is repeated for three detection groups:
- total: all detections
- cars: det_class == 10
- big:  det_class == 9

For each group, a union mask of all parked vehicles is built to estimate the total parking surface covered
(pixels and % of total parking mask area), avoiding double-counting overlaps between vehicles.

After processing all frames with a progress bar (tqdm), the script summarizes average occupancy and reports the
most/least occupied frames (based on % parking area used) with detailed metrics.
"""



# === CONFIGURATION ===
mask_path = r"C:/Users/makss/Git/Galatsi-Semester-Project/Results/parking_detection/dwell_mult/test2_thr_0.8_2/parking_dwell_state_MULTI_REG_parking_location_mask.png"
folder = r"C:/Users/makss/Git/Galatsi-Semester-Project/Results/TXT_0004"
overlap_thr = 0.3  # % of vehicle area inside mask to consider it "parked"

# === LOAD PARKING MASK ===
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
if mask is None:
    raise FileNotFoundError(f"Could not load mask image at {mask_path}")

mask_bin = (mask > 127).astype(np.uint8)
parking_area = np.sum(mask_bin)  # total white pixels of valid parking area

H, W = mask.shape

# --- Helper: compute overlap between vehicle polygon and parking mask ---
def vehicle_overlap_area(coords, mask_bin):
    poly = np.array(coords, dtype=np.int32)

    # blank mask for this vehicle
    veh_mask = np.zeros_like(mask_bin, dtype=np.uint8)

    # draw the oriented box polygon
    cv2.fillPoly(veh_mask, [poly], 1)

    # pixels where the vehicle is on a valid parking pixel
    overlap = np.sum((veh_mask == 1) & (mask_bin == 1))

    # total pixel area of that vehicle polygon
    total = np.sum(veh_mask)

    if total == 0:
        return 0, 0, 0.0

    return overlap, total, overlap / total  # (overlap px, veh px, % veh in parking)

def process_frame(txt_path):
    """
    Returns a list of dicts (one for each vehicle class variant: total / cars / big)
    with stats for this frame.
    """
    df = pd.read_csv(txt_path, sep=';', engine='python')
    df.columns = [c.strip() for c in df.columns]  # strip whitespace from headers

    stats = []

    for cls_name, cls_code in [("total", None), ("cars", 10), ("big", 9)]:
        subset = df if cls_code is None else df[df["det_class"] == cls_code]

        inside_count = 0          # how many vehicles counted as parked
        union_mask = np.zeros_like(mask_bin, dtype=np.uint8)  # to avoid double count of pixels

        for _, r in subset.iterrows():
            # build the polygon from the 4 bounding box corners
            coords = [(r[f"veh_bb_x{i}"], r[f"veh_bb_y{i}"]) for i in range(1, 5)]

            overlap, total, ratio = vehicle_overlap_area(coords, mask_bin)

            # consider it "parked" if enough of the vehicle lies inside parking area
            if ratio > overlap_thr:
                inside_count += 1
                # paint that vehicle onto union_mask
                cv2.fillPoly(union_mask, [np.array(coords, dtype=np.int32)], 1)

        # now compute how much parking surface is covered in total by this class
        area_covered_pixels = np.sum((union_mask == 1) & (mask_bin == 1))

        total_veh = len(subset)
        perc_inside = (inside_count / total_veh * 100) if total_veh > 0 else 0.0
        perc_area = (area_covered_pixels / parking_area * 100) if parking_area > 0 else 0.0

        stats.append({
            "frame": os.path.basename(txt_path),
            "type": cls_name,
            "total_vehicles": int(total_veh),
            "vehicles_in_parking": int(inside_count),
            "perc_in_parking": perc_inside,
            "parking_area_used_%": perc_area,
            "parking_area_used_pixels": int(area_covered_pixels),
        })

    return stats


# === PROCESS ALL FRAMES WITH TQDM ===
frames = sorted(glob.glob(os.path.join(folder, "*.txt")))
all_stats = []

for fpath in tqdm(frames, desc="Processing frames"):
    frame_stats = process_frame(fpath)
    all_stats.extend(frame_stats)

df = pd.DataFrame(all_stats)

# === SUMMARY PER TYPE ===
summary = {}
details = {}  # we’ll store full details for best and worst frames

for cls in ["total", "cars", "big"]:
    sub = df[df["type"] == cls]

    # identify max/min occupancy by AREA USED %
    idx_max = sub["parking_area_used_%"].idxmax()
    idx_min = sub["parking_area_used_%"].idxmin()

    row_max = sub.loc[idx_max]
    row_min = sub.loc[idx_min]

    summary[cls] = {
        "avg_parking_use_%": sub["perc_in_parking"].mean(),         # avg % of vehicles that are parked
        "avg_area_use_%": sub["parking_area_used_%"].mean(),        # avg % of parking surface covered
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
            "%_vehicles_in_parking": row_max["perc_in_parking"],
            "%_parking_surface_used": row_max["parking_area_used_%"],
            "parking_surface_pixels_used": int(row_max["parking_area_used_pixels"]),
        },
        "LEAST_OCCUPIED": {
            "frame": row_min["frame"],
            "total_vehicles": int(row_min["total_vehicles"]),
            "vehicles_in_parking": int(row_min["vehicles_in_parking"]),
            "%_vehicles_in_parking": row_min["perc_in_parking"],
            "%_parking_surface_used": row_min["parking_area_used_%"],
            "parking_surface_pixels_used": int(row_min["parking_area_used_pixels"]),
        }
    }

# === CLEAN PRINT ===
print("\n================ SUMMARY ================\n")
for t, vals in summary.items():
    print(f"--- {t.upper()} ---")
    print(f"Average % vehicles parked      : {vals['avg_parking_use_%']:.2f}%")
    print(f"Average % parking area covered : {vals['avg_area_use_%']:.2f}%")
    print(f"Most occupied frame            : {vals['most_occupied_frame']} ({vals['max_area_%']:.2f}% area used)")
    print(f"Least occupied frame           : {vals['least_occupied_frame']} ({vals['min_area_%']:.2f}% area used)")
    print()

print("\n============ DETAILS PER EXTREME FRAME ============\n")
for t, d in details.items():
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
