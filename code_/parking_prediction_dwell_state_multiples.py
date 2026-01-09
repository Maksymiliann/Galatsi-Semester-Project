import re
from pathlib import Path
import numpy as np
import cv2

try:
    from tqdm import tqdm
except:
    def tqdm(x, **k): return x

"""
This script builds a single parking “dwell vs move” heatmap by aggregating multiple video/detection folders
after registering them into one common reference view.

Inputs:
- Several TXT directories containing per-frame detections (vehicle id, 4-point OBB polygon, confidence, state)
- One representative image per directory, where image_paths[0] is the reference view

Pipeline:
1) Estimate a homography H_k (image_k → image_ref) for each non-reference view using SIFT (ORB fallback) + RANSAC.
   Optional debug outputs save each warped image and an overlay with the reference to visually verify registration.
2) For every detection file in each directory:
   - read detections and filter by conf_min
   - warp each OBB polygon into the reference coordinate system with cv2.perspectiveTransform(H_k)
   - rasterize the warped polygon onto a coarse grid (cell pixels per cell)
   - accumulate STOP as dwell time (+dt seconds) and MOVE as activity (+1 hit, or +dt if desired)
3) Compute a signed score grid: score = w_stop * dwell_stop − w_move * move_hits, then normalize it to [0,1]
   using robust percentiles (2–98%) for visualization.
4) Upsample to full image resolution, optionally smooth, convert to a JET heatmap, and overlay on the reference.
   A thresholded version is also saved, and a binary parking_location_mask is exported from score_img >= thr.

Outputs (with out_prefix):
- *_score_map.png, *_overlay.png
- *_score_map_thresh.png, *_overlay_thresh.png
- *_parking_location_mask.png
The function also returns the final score image, overlays, the list of homographies, and the raw (unnormalized) score.
"""


# -----------------------------
# Parsing utils (inchangé)
# -----------------------------
def parse_line_semicolon(line):
    parts = [p.strip() for p in line.strip().split(';') if p.strip() != ""]
    if len(parts) < 11:
        return None
    veh_id = None
    for k in range(min(3, len(parts)-10)):
        try:
            veh_id = int(float(parts[k])); parts = parts[k:]; break
        except: continue
    if veh_id is None: return None

    floats=[]
    for p in parts[1:]:
        try: floats.append(float(p))
        except: break
    if len(floats) < 8: return None
    x1,y1,x2,y2,x3,y3,x4,y4 = floats[:8]

    state, conf = None, None
    for p in reversed(parts):
        if state is None and re.match(r'[A-Za-z]+', p):
            state = p.lower(); continue
        if conf is None:
            try: conf=float(p); break
            except: continue
    if state is None: state='stop'
    if conf  is None: conf=1.0

    poly = np.array([[x1,y1],[x2,y2],[x3,y3],[x4,y4]], dtype=np.float32)
    return veh_id, poly, conf, state


def read_txt_file(fp):
    dets=[]
    with open(fp, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line=line.strip()
            if not line: continue
            p = parse_line_semicolon(line)
            if p: dets.append(p)
    return dets

# -----------------------------
# Raster utils
# -----------------------------
def poly_to_cells(poly_xy, H, W, cell):
    """
    Remplit l'OBB (déjà dans le repère de l'image de référence) sur une grille.
    poly_xy: (4,2) float32 en coordonnées image REF
    """
    h_grid, w_grid = int(np.ceil(H/cell)), int(np.ceil(W/cell))
    poly_grid = (poly_xy / cell).astype(np.float32)
    mask = np.zeros((h_grid, w_grid), dtype=np.uint8)
    cv2.fillPoly(mask, [poly_grid.astype(np.int32)], 1)
    return mask.astype(bool)

# -----------------------------
# Registration (SIFT + RANSAC)
# -----------------------------
def estimate_homography(img_ref_bgr, img_bgr,
                        ratio=0.75, ransac_thr=3.0, max_kpts=4000):
    """
    Retourne H (img -> ref), nb_inliers, reproj_err.
    Si échec, retourne (I, 0, inf).
    """
    # SIFT
    try:
        sift = cv2.SIFT_create(nfeatures=max_kpts)
    except Exception:
        # fallback ORB
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
    knn = bf.knnMatch(d2, d1, k=2)  # img -> ref
    good = [m for m,n in knn if n is not None and m.distance < ratio*n.distance]
    if len(good) < 4:
        return np.eye(3, dtype=np.float32), 0, float('inf')

    pts2 = np.float32([k2[m.queryIdx].pt for m in good]).reshape(-1,1,2)  # img
    pts1 = np.float32([k1[m.trainIdx].pt for m in good]).reshape(-1,1,2)  # ref

    H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, ransac_thr)
    if H is None:
        return np.eye(3, dtype=np.float32), 0, float('inf')

    inl = mask.ravel().astype(bool)
    err = float('inf')
    if inl.any():
        p2 = pts2[inl].reshape(-1,2); p1 = pts1[inl].reshape(-1,2)
        p2h = np.hstack([p2, np.ones((len(p2),1), np.float32)])
        proj = (H @ p2h.T).T; proj = proj[:,:2]/proj[:,2:3]
        err = float(np.mean(np.linalg.norm(proj - p1, axis=1)))

    return H.astype(np.float32), int(inl.sum()), err

# -----------------------------
# Core (multi-dossiers + registration)
# -----------------------------
def run_parking_dwell_state_multi_registered(
    txt_dirs,              # liste des dossiers TXT (len = K)
    image_paths,           # liste des chemins images correspondantes (len = K), image_paths[0] = REF
    fps=25.0,              # float OU liste par dossier (len = K)
    cell=4,
    conf_min=0.0,
    w_stop=1.0,
    w_move=1.0,
    gaussian_sigma=1.5,
    alpha_overlay=0.6,
    thr=0.70,
    out_prefix="parking_dwell_state_MULTI",
    save_reg_debug=True,     # sauve les overlays warp/ref pour debug
):
    """
    - Calcule H_k : image_k -> image_ref (k=1..K-1) via SIFT+RANSAC
    - Reprojette toutes les OBB dans le repère REF
    - Agrège dwell_stop & move_hits et génère la prédiction unique
    """
    # 0) checks
    assert len(txt_dirs) == len(image_paths), "txt_dirs et image_paths doivent avoir même longueur."
    K = len(txt_dirs)

    # fps peut être scalaire ou liste
    if isinstance(fps, (int, float)):
        fps_list = [float(fps)] * K
    else:
        assert len(fps) == K, "fps liste doit avoir la même longueur que txt_dirs."
        fps_list = [float(x) for x in fps]

    # 1) image de référence
    img_ref = cv2.imread(str(image_paths[0]), cv2.IMREAD_COLOR)
    if img_ref is None:
        raise FileNotFoundError(f"Cannot read reference image: {image_paths[0]}")
    H_img, W_img = img_ref.shape[:2]
    hG, wG = int(np.ceil(H_img/cell)), int(np.ceil(W_img/cell))

    # 2) estimer les homographies vers la REF
    H_list = [np.eye(3, dtype=np.float32)]
    for k in range(1, K):
        img_k = cv2.imread(str(image_paths[k]), cv2.IMREAD_COLOR)
        if img_k is None:
            raise FileNotFoundError(f"Cannot read image: {image_paths[k]}")
        Hk, ninl, err = estimate_homography(img_ref, img_k)
        H_list.append(Hk)
        print(f"[H] {k}: inliers={ninl}, reproj_err={err:.3f}px")

        if save_reg_debug:
            warped = cv2.warpPerspective(img_k, Hk, (W_img, H_img))
            overlay = cv2.addWeighted(img_ref, 0.5, warped, 0.5, 0)
            Path(Path(out_prefix).parent).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(f"{out_prefix}_reg_warp_{k}.png", warped)
            cv2.imwrite(f"{out_prefix}_reg_overlay_{k}.png", overlay)

    # 3) accumulateurs globaux (dans repère REF)
    dwell_stop = np.zeros((hG, wG), dtype=np.float32)
    move_hits  = np.zeros((hG, wG), dtype=np.float32)

    # 4) boucle sur dossiers
    for k in range(K):
        d = Path(txt_dirs[k])
        files = sorted(d.glob("*.txt"))
        if not files:
            print(f"[WARN] No .txt in {d}")
            continue
        dt = 1.0 / max(1e-6, fps_list[k])
        Hk = H_list[k]

        for fp in tqdm(files, desc=f"Accumulating (set {k})", unit="file"):
            dets = read_txt_file(fp)
            if not dets: 
                continue
            for _, poly, conf, state in dets:
                if conf < conf_min:
                    continue
                # projeter l'OBB (4,2) vers l'image REF
                pts = poly.reshape(-1,1,2)  # (4,1,2)
                pts_warp = cv2.perspectiveTransform(pts, Hk).reshape(-1,2).astype(np.float32)
                # filtre: points valides dans l'image (optionnel)
                if not np.isfinite(pts_warp).all():
                    continue

                mask = poly_to_cells(pts_warp, H_img, W_img, cell)
                if state.startswith("stop"):
                    dwell_stop[mask] += dt
                else:
                    move_hits[mask]  += 1.0  # mets += dt pour unitaire secondes

    # 5) score brut -> visu normalisée (percentiles) pour heatmap
    score_grid_brut = w_stop * dwell_stop - w_move * move_hits
    flat = score_grid_brut.ravel()
    lo = np.percentile(flat, 2)
    hi = np.percentile(flat, 98)
    score_grid = np.clip((score_grid_brut - lo) / max(1e-6, (hi - lo)), 0, 1)

    # 6) upsample visu + lissage + colormap
    score_img = cv2.resize(score_grid.astype(np.float32), (W_img, H_img), interpolation=cv2.INTER_CUBIC)
    if gaussian_sigma and gaussian_sigma > 0:
        ksize = int(max(3, 2*int(3*gaussian_sigma)+1))
        score_img = cv2.GaussianBlur(score_img, (ksize, ksize), gaussian_sigma)

    score_uint8 = (np.clip(score_img, 0, 1) * 255).astype(np.uint8)
    heatmap_bgr  = cv2.applyColorMap(score_uint8, cv2.COLORMAP_JET)
    overlay_bgr  = cv2.addWeighted(img_ref, 1.0, heatmap_bgr, alpha_overlay, 0)

    Path(Path(out_prefix).parent).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(f"{out_prefix}_score_map.png",  heatmap_bgr)
    cv2.imwrite(f"{out_prefix}_overlay.png",    overlay_bgr)
    print(f"[OK] Saved score map & overlay.")

    # 7) threshold sur heatmap normalisée (comme avant, pour visu)
    mask = (score_img >= float(thr)).astype(np.uint8)
    mask3 = np.dstack([mask]*3)
    heatmap_thresh = (heatmap_bgr * mask3).astype(np.uint8)
    overlay_thresh = cv2.addWeighted(img_ref, 1.0, heatmap_thresh, alpha_overlay, 0)
    cv2.imwrite(f"{out_prefix}_score_map_thresh.png", heatmap_thresh)
    cv2.imwrite(f"{out_prefix}_overlay_thresh.png",   overlay_thresh)
    print(f"[OK] Saved thresholded map & overlay.")

    # 8) mask for parking location
    mask_img = (score_img >= thr).astype(np.uint8)
    cv2.imwrite(f"{out_prefix}_parking_location_mask.png", (mask_img*255).astype(np.uint8))

    return score_img.astype(np.float32), overlay_bgr, heatmap_bgr, H_list, score_grid_brut


# -----------------------------
# Exemple d’utilisation
# -----------------------------
if __name__ == "__main__":
    TXT_DIRS = [
        r"C:/Users/makss/Git/Galatsi-Semester-Project/Results/TXT_0004",
        r"C:/Users/makss/Git/Galatsi-Semester-Project/Results/TXT_0005",
        r"C:/Users/makss/Git/Galatsi-Semester-Project/Results/TXT_0006",
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
    FPS = [25.0, 25.0, 25.0, 25.0, 25.0, 25.0]  # ou une valeur unique

    run_parking_dwell_state_multi_registered(
        txt_dirs=TXT_DIRS,
        image_paths=IMG_PATHS,
        fps=FPS,
        cell=10,    #10
        conf_min=0.0,   #0.0
        w_stop=1.0,     #1.0
        w_move=1.0,      # add +=dt pour move pour un score en secondes #1.0
        gaussian_sigma=1.5, #1.5
        alpha_overlay=0.6,  #0.6
        thr=0.85,   #0.9
        out_prefix="Results/parking_detection/dwell_mult/test2_thr_0.85/parking_dwell_state_MULTI_REG",
        save_reg_debug=True
    )
