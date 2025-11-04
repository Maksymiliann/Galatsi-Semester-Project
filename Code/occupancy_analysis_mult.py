import cv2
import numpy as np
import pandas as pd
import glob, os
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import matplotlib.pyplot as plt

###########################################################
# CONFIGURATION
###########################################################
IMG_REF   = r"C:/Users/makss/Git/Galatsi-Semester-Project/frame_1_0004.png"
MASK_PATH = r"C:/Users/makss/Git/Galatsi-Semester-Project/Results/parking_detection/zone/mask_closed.png"

TXT_DIRS = [
    r"C:/Users/makss/Git/Galatsi-Semester-Project/Results/TXT_0004",
    r"C:/Users/makss/Git/Galatsi-Semester-Project/Results/TXT_0005",
    r"C:/Users/makss/Galatsi-Semester-Project/Results/TXT_0006",
    r"C:/Users/makss/Git/Galatsi-Semester-Project/Results/TXT_0312_D2_S3_S1",
    r"C:/Users/makss/Git/Galatsi-Semester-Project/Results/TXT_0314_D2_S4_S1",
    r"C:/Users/makss/Git/Galatsi-Semester-Project/Results/TXT_0319_D2_S5_S1",
]
IMG_PATHS = [
    r"C:/Users/makss/Git/Galatsi-Semester-Project/frame_1_0004.png",  # REF
    r"C:/Users/makss/Git/Galatsi-Semester-Project/frame_1_0005.png",
    r"C:/Users/makss/Git/Galatsi-Semester-Project/frame_1_0006.png",
    r"C:/Users/makss/Git/Galatsi-Semester-Project/frame_1_0312.png",
    r"C:/Users/makss/Git/Galatsi-Semester-Project/frame_1_0314.png",
    r"C:/Users/makss/Git/Galatsi-Semester-Project/frame_1_0319.png",
]

OVERLAP_THR = 0.7
OUT_DIR = Path("Results/occupancy_analysis/mult_vid/test3/")
OUT_DIR.mkdir(parents=True, exist_ok=True)
SAVE_DEBUG = False

###########################################################
# GLOBALS POUR WORKERS
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
# HOMOGRAPHY (SIFT/ORB)
###########################################################
def estimate_homography(img_ref_bgr, img_act_bgr, ratio=0.75, ransac_thr=3.0, max_kpts=4000):
    try:
        sift = cv2.SIFT_create(nfeatures=max_kpts)
    except Exception:
        sift = None

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
    knn = bf.knnMatch(d2, d1, k=2)
    good = [m for m,n in knn if n is not None and m.distance < ratio*n.distance]
    if len(good) < 4:
        return np.eye(3, dtype=np.float32), 0, float('inf')

    pts_act = np.float32([k2[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    pts_ref = np.float32([k1[m.trainIdx].pt for m in good]).reshape(-1,1,2)

    H, inl_mask = cv2.findHomography(pts_act, pts_ref, cv2.RANSAC, ransac_thr)
    if H is None:
        return np.eye(3, dtype=np.float32), 0, float('inf')
    return H.astype(np.float32), int(np.sum(inl_mask)), 0.0

###########################################################
# ANALYSE D'UN FOLDER (une vidéo)
###########################################################
def process_one_frame(txt_path):
    df = pd.read_csv(txt_path, sep=';', engine='python')
    df.columns = [c.strip() for c in df.columns]
    union_total = np.zeros((H_global, W_global), dtype=np.uint8)
    total_count, parked_count = 0, 0
    for _, r in df.iterrows():
        poly = np.array([
            (r["veh_bb_x1"], r["veh_bb_y1"]),
            (r["veh_bb_x2"], r["veh_bb_y2"]),
            (r["veh_bb_x3"], r["veh_bb_y3"]),
            (r["veh_bb_x4"], r["veh_bb_y4"]),
        ], dtype=np.float32)
        total_count += 1
        cx, cy = np.mean(poly[:,0]), np.mean(poly[:,1])
        cx_i, cy_i = int(cx), int(cy)
        if not (0 <= cx_i < W_global and 0 <= cy_i < H_global):
            continue
        if mask_bin_global[max(0, cy_i-2):cy_i+3, max(0, cx_i-2):cx_i+3].sum() == 0:
            continue
        xmin, ymin = np.min(poly[:,0]).astype(int), np.min(poly[:,1]).astype(int)
        xmax, ymax = np.max(poly[:,0]).astype(int), np.max(poly[:,1]).astype(int)
        xmin, xmax = max(xmin,0), min(xmax,W_global-1)
        ymin, ymax = max(ymin,0), min(ymax,H_global-1)
        veh_mask = np.zeros((ymax-ymin+1, xmax-xmin+1), np.uint8)
        shifted = (poly - [xmin, ymin]).astype(np.int32)
        cv2.fillPoly(veh_mask, [shifted], 1)
        mask_roi = mask_bin_global[ymin:ymax+1, xmin:xmax+1]
        overlap_ratio = np.sum((veh_mask==1) & (mask_roi==1)) / max(1, np.sum(veh_mask))
        if overlap_ratio > overlap_thr_global:
            parked_count += 1
            union_total[ymin:ymax+1, xmin:xmax+1] |= veh_mask
    covered_pixels = int(np.sum((union_total==1)&(mask_bin_global==1)))
    parking_area_used_pct = 100.0*covered_pixels/parking_area_global
    perc_in_parking = 100.0*parked_count/max(1,total_count)
    return {
        "frame": os.path.basename(txt_path),
        "total_vehicles": total_count,
        "vehicles_in_parking": parked_count,
        "perc_in_parking": perc_in_parking,
        "parking_area_used_%": parking_area_used_pct,
        "parking_area_used_pixels": covered_pixels,
    }

def analyze_video(txt_dir, img_act, img_ref, mask_ref, out_prefix):
    print(f"\n=== Processing {txt_dir} ===")
    img_ref_bgr = cv2.imread(img_ref, cv2.IMREAD_COLOR)
    img_act_bgr = cv2.imread(img_act, cv2.IMREAD_COLOR)
    if img_ref_bgr is None or img_act_bgr is None:
        raise FileNotFoundError("Cannot load reference or active image.")

    H_act_to_ref, ninl, err = estimate_homography(img_ref_bgr, img_act_bgr)
    try:
        H_ref_to_act = np.linalg.inv(H_act_to_ref)
    except np.linalg.LinAlgError:
        H_ref_to_act = np.eye(3, np.float32)
    H_act, W_act = img_act_bgr.shape[:2]
    mask_act = cv2.warpPerspective(mask_ref, H_ref_to_act, (W_act, H_act), flags=cv2.INTER_NEAREST)
    mask_bin = (mask_act > 127).astype(np.uint8)
    parking_area = int(np.sum(mask_bin))
    frames = sorted(glob.glob(os.path.join(txt_dir, "*.txt")))
    if not frames:
        return None
    with ProcessPoolExecutor() as pool:
        pool._initializer = init_worker
        pool._initargs = (mask_bin, parking_area, OVERLAP_THR)
        res = list(tqdm(pool.map(process_one_frame, frames), total=len(frames),
                        desc=f"Analyzing {Path(txt_dir).name}"))
    df = pd.DataFrame(res)
    df["video_name"] = Path(txt_dir).name
    df.to_csv(OUT_DIR / f"{Path(txt_dir).name}_results.csv", index=False)
    return df

###########################################################
# MAIN
###########################################################
if __name__ == "__main__":
    img_ref_bgr = cv2.imread(IMG_REF, cv2.IMREAD_COLOR)
    mask_ref = cv2.imread(MASK_PATH, cv2.IMREAD_GRAYSCALE)
    if mask_ref is None:
        raise FileNotFoundError(f"Could not read {MASK_PATH}")

    all_dfs = []
    for txt_dir, img_act in zip(TXT_DIRS, IMG_PATHS):
        df = analyze_video(txt_dir, img_act, IMG_REF, mask_ref, OUT_DIR)
        if df is not None:
            all_dfs.append(df)

    if not all_dfs:
        raise RuntimeError("No results generated.")

    df_all = pd.concat(all_dfs, ignore_index=True)
    summary = (
        df_all.groupby("video_name")
        .agg({
            "perc_in_parking": "mean",
            "parking_area_used_%": "mean"
        })
        .rename(columns={
            "perc_in_parking": "Avg % parked",
            "parking_area_used_%": "Avg % parking area used"
        })
    )

    summary.to_csv(OUT_DIR / "summary_all_videos.csv")
    print("\n=== Summary Across Videos ===")
    print(summary)

    plt.figure(figsize=(8,4))
    summary["Avg % parking area used"].plot(kind='bar', color='skyblue')
    plt.ylabel("% Parking Surface Used (avg)")
    plt.title("Average Parking Surface Usage per Video")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "avg_usage_per_video.png")
    plt.show()


    # =====================  MIN/MAX (fenêtre glissante 5 min)  =====================
    import re
    import matplotlib.pyplot as plt

    # --- Config FPS ---
    FPS_DEFAULT = 25.0
    # Optionnel: associer un FPS spécifique par dossier (clé = nom de dossier)
    FPS_MAP = {
        # "TXT_0004": 25.0,
        # "TXT_0005": 30.0,
    }

    def frame_index_from_name(name: str) -> int:
        # Essaie d'extraire un numéro de frame depuis "000123.txt", "frame_123.txt", etc.
        m = re.search(r'(\d+)', name)
        return int(m.group(1)) if m else 0

    # Assure un index triable
    df_all["frame_idx"] = df_all["frame"].apply(frame_index_from_name)

    # Fenêtre glissante de 5 minutes => N = fps * 60 * 5
    def window_size_for(video_name: str) -> int:
        fps = FPS_MAP.get(video_name, FPS_DEFAULT)
        return max(1, int(round(fps * 60 * 5)))

    # On calcule une moyenne glissante par vidéo (triée par frame)
    df_all = df_all.sort_values(["video_name", "frame_idx"]).copy()
    df_all["rolling_mean_%"] = (
        df_all.groupby("video_name", group_keys=False)
            .apply(lambda g: g["parking_area_used_%"]
                    .rolling(window=window_size_for(g.name), min_periods=1)
                    .mean())
    )

    # Trouver min/max de la courbe lissée par vidéo
    ext_rows = []
    for vid, g in df_all.groupby("video_name", sort=False):
        g = g.reset_index(drop=True)
        i_min = g["rolling_mean_%"].idxmin()
        i_max = g["rolling_mean_%"].idxmax()
        row_min = df_all.loc[i_min]
        row_max = df_all.loc[i_max]
        ext_rows.append({
            "video_name": vid,
            "min_%": row_min["rolling_mean_%"],
            "min_frame": row_min["frame"],
            "min_frame_idx": int(row_min["frame_idx"]),
            "max_%": row_max["rolling_mean_%"],
            "max_frame": row_max["frame"],
            "max_frame_idx": int(row_max["frame_idx"]),
            "avg_%": g["rolling_mean_%"].mean()
        })

    df_ext = pd.DataFrame(ext_rows).sort_values("video_name")
    df_ext.to_csv(OUT_DIR / "min_max_by_video_5min_window.csv", index=False)
    print("\n=== Min/Max (rolling 5 min) by video ===")
    print(df_ext)

    # --------------------  Bar plot Min/Max par vidéo  --------------------
    labels = df_ext["video_name"].tolist()
    mins   = df_ext["min_%"].tolist()
    maxs   = df_ext["max_%"].tolist()

    x = np.arange(len(labels))
    width = 0.38

    plt.figure(figsize=(10,5))
    plt.bar(x - width/2, mins, width, label="Min (5 min window)")
    plt.bar(x + width/2, maxs, width, label="Max (5 min window)")
    plt.ylabel("% Parking surface used")
    plt.title("Min/Max parking occupancy per video (5-min rolling)")
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "min_max_per_video_5min_window.png", dpi=150)
    plt.show()

    # --------------------  (Optionnel) Boxplot “style vignette”  --------------------
    # Si tu veux un visuel façon boxplot (comme ta capture), décommente :
    plt.figure(figsize=(10,5))
    data = [grp["rolling_mean_%"].values for _, grp in df_all.groupby("video_name")]
    plt.boxplot(data, labels=[n for n,_ in df_all.groupby("video_name")])
    plt.ylabel("% Parking surface used (5-min rolling)")
    plt.title("Distribution of occupancy per video")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "boxplot_per_video_5min_rolling.png", dpi=150)
    plt.show()

    # --------------------  Détails lisibles pour rapport  --------------------
    # On garde aussi un petit résumé “lisible”:
    summary_minmax = df_ext[[
        "video_name", "min_%", "min_frame", "max_%", "max_frame", "avg_%"
    ]].rename(columns={
        "min_%": "Min % (5min)",
        "max_%": "Max % (5min)",
        "avg_%": "Avg % (5min mean)"
    })
    summary_minmax.to_csv(OUT_DIR / "summary_minmax_readable.csv", index=False)
    print("\n=== Summary (readable) ===")
    print(summary_minmax)
