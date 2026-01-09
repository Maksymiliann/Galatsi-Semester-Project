import cv2
import numpy as np
import pandas as pd
import glob, os
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

"""
This script computes parking occupancy for a detection dataset whose frames are not in the same viewpoint as the
reference parking mask.

It loads a reference image (IMG_REF) and an “active” image (IMG_ACT), estimates a homography ACT→REF using
SIFT (ORB fallback) with RANSAC, then inverts it to warp the reference parking mask into the active image
coordinate system (nearest-neighbor). Optional debug images are saved to visually verify the warp.

With the warped mask (now matching the TXT detections), the script processes all per-frame detection files in
parallel. For each vehicle polygon, it uses ROI rasterization to compute the overlap ratio with parking pixels
and counts the vehicle as “parked” if overlap > OVERLAP_THR. A union footprint of parked polygons is used to
estimate parking surface usage (pixels and %).

Finally, it prints average occupancy metrics and the most/least occupied frames based on % parking area used.
"""



###########################################################
# CONFIG (EDIT THESE PATHS)
###########################################################
# 1) Image de référence (celle qui correspond au mask)
IMG_REF   = r"C:/Users/makss/Git/Galatsi-Semester-Project/frame_1_0004.png"
# 2) Image actuelle (une frame typique du dossier FOLDER)
IMG_ACT   = r"C:/Users/makss/Git/Galatsi-Semester-Project/frame_1_0312.png"

MASK_PATH = r"C:/Users/makss/Git/Galatsi-Semester-Project/Results/parking_detection/zone/mask_closed.png"
FOLDER    = r"C:/Users/makss/Git/Galatsi-Semester-Project/Results/TXT_0312_D2_S3_S1"
OVERLAP_THR = 0.7  # % de l'aire du véhicule sur parking pour le compter "parked"

out_prefix = "Results/occupancy_analysis/w_feature_tracking/test2/"

SAVE_DEBUG = True  # exporte des images de debug du warping

###########################################################
# GLOBALS POUR LES WORKERS
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
# HOMOGRAPHY (SIFT/ORB + RANSAC) : ACT -> REF
###########################################################
def estimate_homography(img_ref_bgr, img_act_bgr, ratio=0.75, ransac_thr=3.0, max_kpts=4000):
    """
    Retourne H (ACT -> REF), nb_inliers, reproj_err.
    Si échec, retourne (I, 0, inf).
    """
    # Try SIFT, fallback to ORB if unavailable
    sift = None
    try:
        sift = cv2.SIFT_create(nfeatures=max_kpts)
    except Exception:
        pass

    if sift is not None:
        k1, d1 = sift.detectAndCompute(img_ref_bgr, None)
        k2, d2 = sift.detectAndCompute(img_act_bgr, None)
        norm = cv2.NORM_L2
    else:
        orb = cv2.ORB_create(nfeatures=max_kpts)
        k1, d1 = orb.detectAndCompute(img_ref_bgr, None)
        k2, d2 = orb.detectAndCompute(img_act_bgr, None)
        norm = cv2.NORM_HAMMING

    if d1 is None or d2 is None or len(k1) < 4 or len(k2) < 4:
        return np.eye(3, dtype=np.float32), 0, float('inf')

    bf = cv2.BFMatcher(norm)
    knn = bf.knnMatch(d2, d1, k=2)  # (act -> ref)
    good = [m for m,n in knn if n is not None and m.distance < ratio*n.distance]
    if len(good) < 4:
        return np.eye(3, dtype=np.float32), 0, float('inf')

    pts_act = np.float32([k2[m.queryIdx].pt for m in good]).reshape(-1,1,2)  # act
    pts_ref = np.float32([k1[m.trainIdx].pt for m in good]).reshape(-1,1,2)  # ref

    H, inl_mask = cv2.findHomography(pts_act, pts_ref, cv2.RANSAC, ransac_thr)
    if H is None:
        return np.eye(3, dtype=np.float32), 0, float('inf')

    inl = inl_mask.ravel().astype(bool)
    err = float('inf')
    if inl.any():
        p2 = pts_act[inl].reshape(-1,2); p1 = pts_ref[inl].reshape(-1,2)
        p2h = np.hstack([p2, np.ones((len(p2),1), np.float32)])
        proj = (H @ p2h.T).T; proj = proj[:,:2]/proj[:,2:3]
        err = float(np.mean(np.linalg.norm(proj - p1, axis=1)))

    return H.astype(np.float32), int(inl.sum()), err

###########################################################
# HELPERS ROI / RASTER
###########################################################
def fast_bbox(poly_xy):
    x = poly_xy[:, 0]; y = poly_xy[:, 1]
    xmin = int(np.floor(np.min(x))); xmax = int(np.ceil(np.max(x)))
    ymin = int(np.floor(np.min(y))); ymax = int(np.ceil(np.max(y)))
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

def compute_overlap_ratio(poly_xy):
    xmin_c, xmax_c, ymin_c, ymax_c = fast_bbox(poly_xy)
    if xmax_c < xmin_c or ymax_c < ymin_c:
        return 0.0
    shifted = poly_xy.copy()
    shifted[:, 0] -= xmin_c; shifted[:, 1] -= ymin_c
    shifted = shifted.astype(np.int32)
    roi_w = xmax_c - xmin_c + 1; roi_h = ymax_c - ymin_c + 1
    veh_local = np.zeros((roi_h, roi_w), dtype=np.uint8)
    cv2.fillPoly(veh_local, [shifted], 1)
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
    shifted = poly_xy.copy()
    shifted[:, 0] -= xmin_c; shifted[:, 1] -= ymin_c
    shifted = shifted.astype(np.int32)
    roi_w = xmax_c - xmin_c + 1; roi_h = ymax_c - ymin_c + 1
    temp = np.zeros((roi_h, roi_w), dtype=np.uint8)
    cv2.fillPoly(temp, [shifted], 1)
    union_mask[ymin_c:ymax_c+1, xmin_c:xmax_c+1] |= temp

###########################################################
# WORKER (par frame)
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

    frame_name = os.path.basename(txt_path)
    perc_in_parking = 100.0 * parked_count / total_count if total_count > 0 else 0.0
    covered_pixels = int(np.sum((union_total == 1) & (mask_bin_global == 1)))
    parking_area_used_pct = 100.0 * covered_pixels / parking_area_global if parking_area_global > 0 else 0.0

    return {
        "frame": frame_name,
        "total_vehicles": int(total_count),
        "vehicles_in_parking": int(parked_count),
        "perc_in_parking": float(perc_in_parking),
        "parking_area_used_%": float(parking_area_used_pct),
        "parking_area_used_pixels": int(covered_pixels),
    }

###########################################################
# MAIN
###########################################################
if __name__ == "__main__":
    # 0) Charger images ref/act
    img_ref = cv2.imread(IMG_REF, cv2.IMREAD_COLOR)
    img_act = cv2.imread(IMG_ACT, cv2.IMREAD_COLOR)
    if img_ref is None:
        raise FileNotFoundError(f"Cannot read IMG_REF: {IMG_REF}")
    if img_act is None:
        raise FileNotFoundError(f"Cannot read IMG_ACT: {IMG_ACT}")

    H_ref, W_ref = img_ref.shape[:2]
    H_act, W_act = img_act.shape[:2]

    # 1) Charger le mask (référence)
    mask_ref = cv2.imread(MASK_PATH, cv2.IMREAD_GRAYSCALE)
    if mask_ref is None:
        raise FileNotFoundError(f"Could not load mask image at {MASK_PATH}")

    # Ajuste le mask à la taille de IMG_REF si besoin (au cas où)
    if mask_ref.shape[:2] != (H_ref, W_ref):
        mask_ref = cv2.resize(mask_ref, (W_ref, H_ref), interpolation=cv2.INTER_NEAREST)

    # 2) Estimer homographie ACT -> REF
    H_act_to_ref, ninl, err = estimate_homography(img_ref, img_act)
    print(f"[H] inliers={ninl}, reproj_err={err:.3f}px")

    # 3) Inverse: REF -> ACT (pour warper le mask dans l'image actuelle)
    try:
        H_ref_to_act = np.linalg.inv(H_act_to_ref)
    except np.linalg.LinAlgError:
        print("[WARN] Homography inverse failed. Falling back to identity.")
        H_ref_to_act = np.eye(3, dtype=np.float32)

    # 4) Warper le mask de référence vers la taille de l'image ACTUELLE
    mask_act = cv2.warpPerspective(mask_ref, H_ref_to_act, (W_act, H_act),
                                   flags=cv2.INTER_NEAREST,
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=0)

    if SAVE_DEBUG:
        warped_ref_on_act = cv2.warpPerspective(img_ref, H_ref_to_act, (W_act, H_act))
        overlay = cv2.addWeighted(img_act, 0.5, warped_ref_on_act, 0.5, 0)
        Path(Path(out_prefix).parent).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(f"{out_prefix}mask_warped_to_ACT.png", mask_act)
        cv2.imwrite(f"{out_prefix}ref_warped_on_ACT.png", warped_ref_on_act)
        cv2.imwrite(f"{out_prefix}overlay_ref_ACT.png", overlay)
        print(f"[OK] Saved debug.")

    # 5) Binariser le mask ACT et calculer l'aire de parking
    mask_bin = (mask_act > 127).astype(np.uint8)
    H, W = mask_bin.shape
    parking_area = int(np.sum(mask_bin))
    print(f"[INFO] Mask warped to ACT size: {W}x{H}, parking_area_px={parking_area}")

    # 6) Lister les frames TXT à analyser (dans le repère ACTUEL)
    frames = sorted(glob.glob(os.path.join(FOLDER, "*.txt")))
    if not frames:
        raise FileNotFoundError(f"No .txt files found in {FOLDER}")

    # 7) Lancer le multiprocess avec le mask ACT
    results = []
    with ProcessPoolExecutor() as pool:
        pool._initializer = init_worker
        pool._initargs = (mask_bin, parking_area, OVERLAP_THR)

        for frame_stats in tqdm(pool.map(process_one_frame, frames),
                                total=len(frames),
                                desc="Processing frames in parallel"):
            results.append(frame_stats)

    df = pd.DataFrame(results)

    # 8) Stats globales & extrêmes (basées sur % surface)
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
