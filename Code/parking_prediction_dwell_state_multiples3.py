import re
import math
from pathlib import Path
import numpy as np
import cv2

try:
    from tqdm import tqdm
except:
    def tqdm(x, **k): return x

"""
This script builds a single parking heatmap from multiple detection folders by registering every view to a
reference image, then applying a weighted “gap filling” boost along likely parking-line directions.

Main steps:
1) Multi-view registration:
   - image_paths[0] is the reference view.
   - For each other view, a homography H_k (image_k → image_ref) is estimated with SIFT (ORB fallback) + RANSAC.
   - Optional debug outputs save warped images and overlays to verify alignment.

2) Stop/move accumulation in the reference frame:
   - Each TXT detection is parsed into (vehicle_id, 4-corner OBB polygon, confidence, state).
   - Polygons are warped into the reference frame with cv2.perspectiveTransform(H_k), rasterized on a coarse grid
     (cell pixels per cell), and accumulated:
       STOP  → dwell_stop += dt seconds
       MOVE  → move_hits  += 1 hit (can be changed to += dt if needed)

3) Scoring and visualization:
   - A signed score is computed: score = w_stop * dwell_stop − w_move * move_hits.
   - The score is normalized to [0,1] using 2–98% percentiles, upsampled to full resolution, optionally smoothed,
     and saved as a JET heatmap + overlay on the reference image.

4) Gap-weighted boosting (parking-line completion):
   - A permissive seed mask is created from the score (score >= thr - 0.10).
   - HoughLinesP detects line segments on the seed edges; segments are grouped by angle and merged with a PCA fit.
   - For each merged segment, a “bridge” boost is computed using:
       • context_stop: mean score near the segment endpoints
       • move_penalty: mean move activity along the segment (discourages roads)
       • dist_weight: depends on how much seed support exists along the segment
   - The resulting boost map is blurred, normalized, and added to the score to fill gaps between aligned segments.
   - Debug images for bridges and boost maps are saved.

5) Final outputs:
   - Thresholded heatmap/overlay (score >= thr)
   - A binary parking_location_mask.png derived from the boosted score map
   - Additional debug files from registration and gap boosting
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


def gap_weighted_boost(score_img,           # float32 [0..1], taille image
                       mask_seed,            # uint8 {0,1}, seed permissif (ex: score_img >= thr*0.8)
                       ref_img_bgr=None,     # pour overlays (optionnel)
                       move_hits_grid=None,  # (hG,wG) float32, facultatif (pénalise routes)
                       cell=4,               # pour upsampler move_hits_grid -> image
                       # --- détection segments ---
                       canny1=30, canny2=90,
                       rho=1, theta_deg=1.0, votes=40,
                       min_len=60, max_gap=25,
                       angle_bin_deg=10, merge_dist_px=25,
                       # --- scoring / géométrie des ponts ---
                       band_width_px=6,         # demi-largeur pour tester/peindre le pont
                       endpoint_win_px=15,      # voisinage des extrémités pour "context_stop"
                       blur_sigma=5.0,
                       gain=0.25,               # gain global max du boost
                       alpha_overlay=0.6,
                       out_prefix=None):
    """
    Crée des ponts entre segments colinéaires et attribue un boost pondéré :
       boost_line = gain * context_stop * (1 - move_penalty) * dist_weight
    où:
      - context_stop : moyenne(score_img) autour des extrémités du pont
      - move_penalty : moyenne(move_map_norm) le long de la bande (routes -> élevé)
      - dist_weight  : décroît avec la longueur quand pas assez de support
    Retourne: boosted_score, debug_img (lignes), boost_map
    """

    H, W = score_img.shape
    # 0) carte de trafic (si fournie)
    move_map = None
    if move_hits_grid is not None:
        # upsample -> image + normalisation robuste [0..1]
        move_map = cv2.resize(move_hits_grid.astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
        flat = move_map.ravel()
        lo = np.percentile(flat, 5)
        hi = np.percentile(flat, 95)
        if hi > lo:
            move_map = np.clip((move_map - lo) / (hi - lo), 0, 1)
        else:
            move_map = np.zeros_like(move_map, np.float32)

    # 1) edges sur seed dilaté (connectivité)
    seed_u8 = (mask_seed.astype(np.uint8) * 255)
    dil = cv2.dilate(seed_u8, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)), 1)
    edges = cv2.Canny(dil, canny1, canny2)

    # 2) HoughP -> segments
    lines = cv2.HoughLinesP(edges, rho=rho, theta=np.deg2rad(theta_deg),
                            threshold=votes, minLineLength=min_len, maxLineGap=max_gap)
    if lines is None:
        # rien à booster
        return score_img, np.zeros((H,W,3), np.uint8), np.zeros((H,W), np.float32)

    segs = [(l[0][0], l[0][1], l[0][2], l[0][3]) for l in lines]

    # 3) regrouper par angle grossier
    def seg_angle(s):
        x1,y1,x2,y2 = s
        return (math.degrees(math.atan2(y2-y1, x2-x1)) + 180.0) % 180.0

    bins = {}
    for s in segs:
        a = seg_angle(s)
        key = int(round(a / angle_bin_deg))
        bins.setdefault(key, []).append(s)

    boost = np.zeros((H, W), np.float32)
    dbg   = np.zeros((H, W, 3), np.uint8)

    # utilitaires
    def line_points(p1, p2, thickness):
        """ Génère des pixels de la bande épaisse entre p1 et p2 (rectangle dilaté). """
        x1,y1 = map(int, p1); x2,y2 = map(int, p2)
        band = np.zeros((H,W), np.uint8)
        cv2.line(band, (x1,y1), (x2,y2), 1, thickness=thickness)
        ys, xs = np.where(band > 0)
        return xs, ys

    def local_mean(img, x, y, r):
        x0, x1 = max(0, x-r), min(W, x+r+1)
        y0, y1 = max(0, y-r), min(H, y+r+1)
        if x1<=x0 or y1<=y0: return 0.0
        roi = img[y0:y1, x0:x1]
        return float(roi.mean()) if roi.size else 0.0

    # 4) pour chaque bin, clusteriser par proximité et fitter une ligne unique
    for _, group in bins.items():
        if len(group) < 2:
            continue

        # points d'extrémités
        pts = []
        for (x1,y1,x2,y2) in group:
            pts.append((x1,y1)); pts.append((x2,y2))
        pts = np.array(pts, np.float32)

        used = np.zeros(len(pts), bool)
        for i in range(len(pts)):
            if used[i]: continue
            cluster_idx = [i]; used[i] = True
            changed = True
            while changed:
                changed = False
                for j in range(len(pts)):
                    if used[j]: continue
                    if np.min(np.linalg.norm(pts[cluster_idx] - pts[j], axis=1)) <= merge_dist_px:
                        cluster_idx.append(j); used[j] = True; changed = True

            cl = pts[cluster_idx]
            if len(cl) < 2: 
                continue

            # fit PCA 2D
            m = cl.mean(axis=0)
            U, S, Vt = cv2.SVDecomp((cl - m).astype(np.float32))
            dirv = Vt[0,:]
            dirv = dirv / (np.linalg.norm(dirv) + 1e-9)

            t = (cl - m) @ dirv
            tmin, tmax = t.min(), t.max()
            p1 = (m + tmin*dirv).astype(int)
            p2 = (m + tmax*dirv).astype(int)

            # 5) calcul du poids de pont
            #    - contexte "stop" aux extrémités (moyenne score_img)
            c1 = local_mean(score_img, int(p1[0]), int(p1[1]), endpoint_win_px)
            c2 = local_mean(score_img, int(p2[0]), int(p2[1]), endpoint_win_px)
            context_stop = max(0.0, (c1 + c2) * 0.5)  # [0..1]

            #    - pénalité trafic (si move_map dispo) le long de la bande
            xs, ys = line_points(p1, p2, thickness=band_width_px)
            if xs.size == 0:
                continue
            if move_map is not None:
                move_penalty = float(move_map[ys, xs].mean())  # [0..1]
            else:
                move_penalty = 0.0

            #    - poids de distance: si pont très long, mais peu de contexte, on réduit
            length = max(1.0, float(np.hypot(*(p2 - p1))))
            # densité seed le long de la bande (proportion de pixels seed dans la bande)
            seed_density = float(mask_seed[ys, xs].mean())
            # plus la bande est "justifiée" par du seed, plus on accepte la longueur
            dist_weight = float(np.clip(seed_density * (min(300.0, length) / 300.0), 0.0, 1.0))

            bridge_weight = gain * context_stop * (1.0 - move_penalty) * dist_weight
            if bridge_weight <= 1e-4:
                continue

            # 6) peindre la bande pondérée dans boost
            tmp = np.zeros((H,W), np.float32)
            cv2.line(tmp, (int(p1[0]),int(p1[1])), (int(p2[0]),int(p2[1])), 1.0, thickness=band_width_px)
            boost = np.maximum(boost, tmp * float(bridge_weight))

            # debug
            cv2.line(dbg, (int(p1[0]),int(p1[1])), (int(p2[0]),int(p2[1])), (0,0,255), 2)

    # 7) lisser et normaliser la boost map
    if blur_sigma and blur_sigma > 0:
        k = int(max(3, 2*int(3*blur_sigma)+1))
        boost = cv2.GaussianBlur(boost, (k,k), blur_sigma)
    if boost.max() > 1e-6:
        boost = boost / boost.max()

    boosted_score = np.clip(score_img + boost, 0.0, 1.0)  # boost déjà borné par gain

    # 8) sorties debug
    if out_prefix:
        cv2.imwrite(f"{out_prefix}_bridges_dbg.png", dbg)
        cv2.imwrite(f"{out_prefix}_boostmap.png", (boost*255).astype(np.uint8))
        hm = cv2.applyColorMap((boosted_score*255).astype(np.uint8), cv2.COLORMAP_JET)
        cv2.imwrite(f"{out_prefix}_score_map_boosted.png", hm)
        if ref_img_bgr is not None:
            ov = cv2.addWeighted(ref_img_bgr, 1.0, hm, alpha_overlay, 0)
            cv2.imwrite(f"{out_prefix}_overlay_boosted.png", ov)

    return boosted_score, dbg, boost

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


    # seed permissif (un peu sous ton thr final)
    thr_seed = max(0.5, float(thr) - 0.10)
    mask_seed = (score_img >= thr_seed).astype(np.uint8)

    # si tu as move_hits (grille hG×wG) et cell disponibles ici, passe-les :
    # move_hits vient de ta boucle d'accumulation; garde-le côté image de ref.
    try:
        move_hits_grid_here = move_hits  # (hG,wG) float32
    except NameError:
        move_hits_grid_here = None

    score_img, dbg_lines, boost_map = gap_weighted_boost(
        score_img=score_img,
        mask_seed=mask_seed,
        ref_img_bgr=img_ref,
        move_hits_grid=move_hits_grid_here,
        cell=cell,
        min_len=80, max_gap=50,
        band_width_px=20,
        endpoint_win_px=25,
        blur_sigma=5.0,
        gain=0.25,  # ↑/↓ l’impact global du comblement pondéré
        out_prefix=f"{out_prefix}_gapboost",
        alpha_overlay=alpha_overlay
)

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
        out_prefix="Results/parking_detection/dwell_mult_3/test2_thr_0.85/parking_dwell_state_MULTI_REG",
        save_reg_debug=True
    )
