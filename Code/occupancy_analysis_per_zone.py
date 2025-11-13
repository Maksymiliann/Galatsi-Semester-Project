import cv2
import numpy as np
import pandas as pd
import glob, os
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

###########################################################
# CONFIG
###########################################################
MASK_PATH   = r"C:/Users/makss/Git/Galatsi-Semester-Project/Results/occupancy_analysis/per_zones/mask_closed.png"
FOLDER      = r"C:/Users/makss/Git/Galatsi-Semester-Project/Results/TXT_0004"
ZONES_PATH  = r"C:/Users/makss/Git/Galatsi-Semester-Project/Results/occupancy_analysis/per_zones/mask_zones_id.png"

out_dir = r"C:/Users/makss/Git/Galatsi-Semester-Project/Results/occupancy_analysis/per_zones/test1"

OVERLAP_THR = 0.7  # % du polygone véhicule dans parking pour compter “garé”

# (optionnel) filtrer les toutes petites zones
MIN_ZONE_AREA = 4000

# --- Execution mode ---
USE_PARALLEL = True      # True = ProcessPoolExecutor, False = séquentiel
MAX_WORKERS  = None        # None -> os.cpu_count(); sinon mets un int (ex: 4)

###########################################################
# GLOBALS FOR WORKERS
###########################################################
mask_bin_global = None
H_global = None
W_global = None
overlap_thr_global = None

labels_global = None           # int32, 0=bg, 1..K=zones
zone_ids_global = None         # liste des ids >0
zone_areas_global = None       # dict id->aire en px

def init_worker(mask_bin, labels, zone_ids, zone_areas, overlap_thr):
    global mask_bin_global, labels_global, zone_ids_global, zone_areas_global
    global H_global, W_global, overlap_thr_global
    mask_bin_global = mask_bin.astype(np.uint8)
    labels_global = labels.astype(np.int32)
    zone_ids_global = list(zone_ids)
    zone_areas_global = dict(zone_areas)
    H_global, W_global = mask_bin_global.shape
    overlap_thr_global = overlap_thr

###########################################################
# HELPERS
###########################################################
def fast_bbox(poly_xy):
    x_coords = poly_xy[:, 0]
    y_coords = poly_xy[:, 1]
    xmin = int(np.floor(np.min(x_coords))); xmax = int(np.ceil(np.max(x_coords)))
    ymin = int(np.floor(np.min(y_coords))); ymax = int(np.ceil(np.max(y_coords)))
    xmin_c = max(xmin, 0); ymin_c = max(ymin, 0)
    xmax_c = min(xmax, W_global - 1); ymax_c = min(ymax, H_global - 1)
    return xmin_c, xmax_c, ymin_c, ymax_c

def quick_candidate_check(poly_xy):
    cx = float(np.mean(poly_xy[:,0])); cy = float(np.mean(poly_xy[:,1]))
    cx_i = int(round(cx)); cy_i = int(round(cy))
    if cx_i < 0 or cx_i >= W_global or cy_i < 0 or cy_i >= H_global:
        return False
    x0 = max(cx_i - 2, 0); y0 = max(cy_i - 2, 0)
    x1 = min(cx_i + 2, W_global - 1); y1 = min(cy_i + 2, H_global - 1)
    local_mask = mask_bin_global[y0:y1+1, x0:x1+1]
    return np.any(local_mask == 1)

def rasterize_poly(poly_xy):
    """
    Rasterise le polygone UNE SEULE FOIS et renvoie:
    (xmin_c, xmax_c, ymin_c, ymax_c, veh_local_mask)
    """
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
# PER-FRAME WORKER
###########################################################
def process_one_frame(txt_path):
    df = pd.read_csv(txt_path, sep=';', engine='python')
    df.columns = [c.strip() for c in df.columns]

    union_total = np.zeros((H_global, W_global), dtype=np.uint8)
    total_count = 0
    parked_count = 0

    # per-zone accumulators (int)
    z_vehicle_counts = {z: 0 for z in zone_ids_global}
    z_area_px = {z: 0 for z in zone_ids_global}

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

        # --- RASTERISATION UNIQUE DU POLYGONE ---
        res = rasterize_poly(poly_xy)
        if res is None:
            continue
        xmin_c, xmax_c, ymin_c, ymax_c, veh_local = res

        # --- CALCUL DU RATIO D'OVERLAP ---
        mask_roi = mask_bin_global[ymin_c:ymax_c+1, xmin_c:xmax_c+1]
        veh_px = int(np.sum(veh_local))
        if veh_px == 0:
            continue

        overlap_px = int(np.sum((veh_local == 1) & (mask_roi == 1)))
        ratio = overlap_px / veh_px

        if ratio > overlap_thr_global:
            parked_count += 1

            # --- MISE À JOUR DU UNION MASK ---
            union_total[ymin_c:ymax_c+1, xmin_c:xmax_c+1] |= veh_local

            # --- ASSIGNATION DE ZONE AVEC LE MÊME MASK ---
            lab_roi = labels_global[ymin_c:ymax_c+1, xmin_c:xmax_c+1]
            values = lab_roi[veh_local == 1]
            if values.size > 0:
                counts = np.bincount(values, minlength=labels_global.max()+1)
                counts[0] = 0
                if counts.sum() > 0:
                    zid = int(np.argmax(counts))
                    if zid in z_vehicle_counts:
                        z_vehicle_counts[zid] += 1

    # area per zone: use labels on union_total∩mask
    covered = (union_total == 1) & (mask_bin_global == 1)
    if np.any(covered):
        labs = labels_global[covered]
        counts = np.bincount(labs, minlength=labels_global.max()+1)
        for z in zone_ids_global:
            z_area_px[z] = int(counts[z])

    # frame-level stats
    frame_name = os.path.basename(txt_path)
    perc_in_parking = 100.0 * parked_count / total_count if total_count > 0 else 0.0
    covered_pixels = int(np.sum(covered))
    # global pct (toutes zones confondues)
    parking_area_global = int(np.sum(mask_bin_global))
    parking_area_used_pct = 100.0 * covered_pixels / parking_area_global if parking_area_global > 0 else 0.0

    # per-zone pct
    z_area_pct = {
        z: (100.0 * z_area_px[z] / zone_areas_global[z] if zone_areas_global[z] > 0 else 0.0)
        for z in zone_ids_global
    }

    return {
        "frame": frame_name,
        "total_vehicles": int(total_count),
        "vehicles_in_parking": int(parked_count),
        "perc_in_parking": float(perc_in_parking),
        "parking_area_used_%": float(parking_area_used_pct),
        "parking_area_used_pixels": int(covered_pixels),
        "z_vehicle_counts": z_vehicle_counts,
        "z_area_px": z_area_px,
        "z_area_pct": z_area_pct,
    }

###########################################################
# MAIN
###########################################################
if __name__ == "__main__":
    # Load mask (global parking region)
    mask_img = cv2.imread(MASK_PATH, cv2.IMREAD_GRAYSCALE)
    if mask_img is None:
        raise FileNotFoundError(f"Could not load mask image at {MASK_PATH}")
    mask_bin = (mask_img > 127).astype(np.uint8)
    H, W = mask_bin.shape

    # Load / build zones labels
    if ZONES_PATH and Path(ZONES_PATH).exists():
        lab_img = cv2.imread(ZONES_PATH, cv2.IMREAD_GRAYSCALE)
        if lab_img is None:
            raise FileNotFoundError(ZONES_PATH)
        labels = lab_img.astype(np.int32)
        # clamp hors parking
        labels[mask_bin == 0] = 0
    else:
        # fallback: composantes connexes du mask
        num, lab = cv2.connectedComponents(mask_bin, connectivity=8)
        labels = lab.astype(np.int32)

    # --------- FILTRE DES PETITES ZONES (bincount au lieu de (labels==z).sum()) ----------
    if MIN_ZONE_AREA and MIN_ZONE_AREA > 0:
        vals = labels.ravel()
        counts = np.bincount(vals)  # counts[z] = nb de pixels dans la zone z
        num = len(counts) - 1

        keep = np.zeros(num + 1, np.uint8)
        keep[0] = 1  # background toujours gardé

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

    # --------- ZONES META (bincount de nouveau après remap) ----------
    vals = labels.ravel()
    counts = np.bincount(vals)
    zone_ids = [z for z in range(1, len(counts)) if counts[z] > 0]
    zone_areas = {int(z): int(counts[z]) for z in zone_ids}

    # Collect frames
    frames = sorted(glob.glob(os.path.join(FOLDER, "*.txt")))
    # frames = frames[::5]  # dev mode si tu veux

    results = []

    if USE_PARALLEL:
        with ProcessPoolExecutor(
            max_workers=MAX_WORKERS,
            initializer=init_worker,
            initargs=(mask_bin, labels, zone_ids, zone_areas, OVERLAP_THR)
        ) as pool:
            for frame_stats in tqdm(
                pool.map(process_one_frame, frames),
                total=len(frames),
                desc="Processing frames in parallel"
            ):
                results.append(frame_stats)
    else:
        init_worker(mask_bin, labels, zone_ids, zone_areas, OVERLAP_THR)
        for fp in tqdm(frames, total=len(frames), desc="Processing frames sequentially"):
            results.append(process_one_frame(fp))

    # -------- Global dataframe --------
    df = pd.DataFrame([{
        "frame": r["frame"],
        "total_vehicles": r["total_vehicles"],
        "vehicles_in_parking": r["vehicles_in_parking"],
        "perc_in_parking": r["perc_in_parking"],
        "parking_area_used_%": r["parking_area_used_%"],
        "parking_area_used_pixels": r["parking_area_used_pixels"],
    } for r in results])

    idx_max = df["parking_area_used_%"].idxmax()
    idx_min = df["parking_area_used_%"].idxmin()
    row_max = df.loc[idx_max]; row_min = df.loc[idx_min]
    avg_parking_use = df["perc_in_parking"].mean()
    avg_area_use = df["parking_area_used_%"].mean()

    print("\n================ SUMMARY (GLOBAL) ================\n")
    print(f"Average % vehicles parked      : {avg_parking_use:.2f}%")
    print(f"Average % parking area covered : {avg_area_use:.4f}%")
    print(f"Most occupied frame            : {row_max['frame']} ({row_max['parking_area_used_%']:.4f}% area used)")
    print(f"Least occupied frame           : {row_min['frame']} ({row_min['parking_area_used_%']:.4f}% area used)\n")

    print("============ DETAILS (GLOBAL) ============\n")
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

    # -------- Per-zone per-frame (long format) --------
    rows = []
    for r in results:
        f = r["frame"]
        for z in zone_ids:
            rows.append(dict(
                frame=f, zone_id=z,
                vehicles_in_zone=r["z_vehicle_counts"].get(z, 0),
                area_px_in_zone=r["z_area_px"].get(z, 0),
                area_pct_in_zone=r["z_area_pct"].get(z, 0.0),
            ))
    dz = pd.DataFrame(rows).sort_values(["frame","zone_id"])

    # -------- Per-zone summary over all frames --------
    dz_summary = dz.groupby("zone_id").agg(
        avg_veh=("vehicles_in_zone","mean"),
        max_veh=("vehicles_in_zone","max"),
        avg_area_pct=("area_pct_in_zone","mean"),
        max_area_pct=("area_pct_in_zone","max"),
    ).reset_index()
    dz_summary["zone_area_px"] = dz_summary["zone_id"].map(zone_areas)

    # -------- Exports (fix f-strings + dossier) --------
    os.makedirs(out_dir, exist_ok=True)

    df.to_csv(os.path.join(out_dir, "global_results.csv"), index=False)
    dz.to_csv(os.path.join(out_dir, "per_zone_per_frame.csv"), index=False)
    dz_summary.to_csv(os.path.join(out_dir, "per_zone_summary.csv"), index=False)

    print("Saved:")
    print(os.path.join(out_dir, "global_results.csv"))
    print(os.path.join(out_dir, "per_zone_per_frame.csv"))
    print(os.path.join(out_dir, "per_zone_summary.csv"))
