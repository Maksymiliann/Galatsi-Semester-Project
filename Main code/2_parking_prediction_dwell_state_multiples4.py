import re
from pathlib import Path
import numpy as np
import cv2

try:
    from tqdm import tqdm
except:
    def tqdm(x, **k): return x

"""
This script builds a parking heatmap from multiple detection folders by registering all views into a single
reference image, then post-processes the result to (1) fill small gaps between adjacent parking spots and
(2) produce a smoother “parking zones” mask.

Pipeline:
1) Multi-view registration:
   - image_paths[0] is the reference view.
   - For each other view, a homography H_k (image_k → image_ref) is estimated with SIFT (ORB fallback) + RANSAC.
   - Optional debug outputs save warped images and overlays to verify alignment.

2) Stop/move accumulation in the reference frame:
   - Each TXT line is parsed into (vehicle_id, 4-corner OBB polygon, confidence, state).
   - Polygons are warped into the reference frame with cv2.perspectiveTransform(H_k), rasterized on a coarse grid
     (cell pixels per cell), and accumulated:
       STOP → dwell_stop += dt seconds
       MOVE → move_hits  += 1 hit (can be changed to += dt if needed)
   - A signed score is computed: score = w_stop * dwell_stop − w_move * move_hits,
     then normalized to [0,1] using robust percentiles (2–98%), upsampled to image resolution, smoothed, and
     saved as a JET heatmap + overlay. A binary parking_location_mask (score >= thr) is also exported.

3) Gap filling between parking spots (seed-based boosting):
   - cand_band: permissive candidate band where parking is plausible (score >= low_thr).
   - seeds: strong detections from the thresholded mask (score >= thr).
   - seeds are dilated to cover nearby “inter-spot” gaps, but the dilation is done *per connected component*
     and *oriented along the component’s main axis* (minAreaRect), producing seeds_dil.
   - gaps are defined as pixels inside the candidate band and the oriented dilation, but not already in seeds.

4) Distance-weighted score boost:
   - A distance transform gives each gap pixel a weight based on how close it is to an existing seed
     (closer = larger weight, limited by gap_px).
   - score_img is boosted inside gaps by bonus_scale * weight, then clipped back to [0,1].

5) Zone mask generation:
   - The boosted score is thresholded again (>= thr), then morphologically closed (and slightly dilated) to create
     a smoother “zones” mask that merges nearby spots into continuous parking regions.

Outputs (with out_prefix):
- Heatmaps/overlays: *_score_map.png, *_overlay.png, *_score_map_thresh.png, *_overlay_thresh.png
- Binary masks: *_parking_location_mask.png, *_score_boost.png, *_zones_mask.png
- Debug images for oriented dilation and gaps: *_mask_dilated_oriented.png, *_gaps_oriented.png, *_overlay_*_debug.png
- *_overlay_zones.png: visualization of the final zones mask on top of the heatmap overlay
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
    # mask_img = (score_img >= thr).astype(np.uint8)
    # cv2.imwrite(f"{out_prefix}_parking_location_mask.png", (mask_img*255).astype(np.uint8))
    step = 0.005

    for thr in np.arange(0.0, 1.0 + step, step):
        
        mask_img = (score_img >= thr).astype(np.uint8)
        
        filename = f"{out_prefix}_parking_location_mask_thr_{thr:.3f}.png"
        
        cv2.imwrite(filename, (mask_img * 255).astype(np.uint8))

    # === 9) combler les "trous entre places" et créer des zones de parking ===
    # Idées clés:
    # - cand_band : bande candidate où des places sont plausibles (score moyen/haut)
    # - seeds     : graines => tes places fortes (mask_img)
    # - gaps      : pixels proches des graines, dans la bande, mais non détectés (les "trous")
    # - on booste le score dans gaps proportionnellement à la proximité des graines

    # Paramètres (à ajuster selon l'échelle de tes images)
    low_thr      = 0.45   # seuil bas pour définir la "bande routière" candidate
    gap_px       = 20     # taille max du trou à combler (en pixels image)
    bonus_scale  = 0.35   # combien on ajoute au score dans les gaps (0..1)
    close_len_px = 9      # fermeture morphologique pour lier les segments (px)

    # 9.1) bande candidate (zones de route/parking plausibles)
    cand_band = (score_img >= float(low_thr)).astype(np.uint8)

    # 9.2) graines = tes détections fortes
    seeds = (mask_img > 0).astype(np.uint8)

    # # 9.3) gaps: dilatation des graines pour attraper les inter-espaces proches
    # k_disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*gap_px+1, 2*gap_px+1))
    # seeds_dil = cv2.dilate(seeds, k_disk)
    # gaps = ((seeds_dil > 0) & (cand_band > 0) & (seeds == 0)).astype(np.uint8)

    # 9.3) dilatation ORIENTÉE par composante (suivant l'axe long de chaque seed)
    def line_kernel(L, angle_deg, thickness=1):
        k = np.zeros((L, L), np.uint8)
        c = L // 2
        rad = np.deg2rad(angle_deg)
        dx, dy = int(np.cos(rad)*c), int(np.sin(rad)*c)
        cv2.line(k, (c-dx, c-dy), (c+dx, c+dy), 1, thickness)
        return k

    def component_oriented_dilate(seeds_bin, gap_px, cand_band_bin, snap=None, thickness=1):
        """
        Dilate chaque composante connexe de 'seeds_bin' uniquement le long de son axe principal.
        snap: si donné (ex. 15, 30, 45, 90), l'angle est arrondi au multiple le plus proche.
        """
        L = 2*gap_px + 1
        # labels: 0 arrière-plan, 1..N composantes
        num, labels = cv2.connectedComponents(seeds_bin, connectivity=8)
        out = np.zeros_like(seeds_bin)

        for lbl in range(1, num):
            comp = (labels == lbl).astype(np.uint8)
            # contour principal
            cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            cnt = max(cnts, key=cv2.contourArea)
            (cx, cy), (w, h), ang = cv2.minAreaRect(cnt)  # ang in (-90,0]
            if w < 1 or h < 1:
                continue

            # orienter selon l'AXE LONG de la boîte
            if w < h:
                ang = ang + 90.0  # convertit l'angle OpenCV pour pointer l'axe long

            # option: arrondir l’angle (stabilise)
            if snap and snap > 0:
                ang = round(ang / snap) * snap

            # kernel linéaire aligné + dilatation seulement pour cette composante
            k_line = line_kernel(L, ang, thickness=thickness)
            dil = cv2.dilate(comp, k_line)

            # union des dilatations
            out = np.maximum(out, dil)

        out = (out > 0).astype(np.uint8)
        # contraindre à la bande candidate
        out = (out & cand_band_bin).astype(np.uint8)
        return out

    # --- appel : dilatation orientée + définition des gaps
    seeds_dil = component_oriented_dilate(seeds, gap_px=gap_px, cand_band_bin=cand_band,
                                          snap=15,   # mets None si tu veux l'angle exact
                                          thickness=1)
    gaps = ((seeds_dil > 0) & (cand_band > 0) & (seeds == 0)).astype(np.uint8)

    # === Debug visuel des gaps et du mask dilaté ORIENTÉ ===
    cv2.imwrite(f"{out_prefix}_mask_dilated_oriented.png", (seeds_dil * 255).astype(np.uint8))
    cv2.imwrite(f"{out_prefix}_gaps_oriented.png", (gaps * 255).astype(np.uint8))
    debug_overlay = img_ref.copy()
    debug_overlay[gaps > 0] = (0, 0, 255)          # gaps en rouge
    debug_overlay[seeds_dil > 0] = (0, 255, 0)     # dilatation orientée en vert
    cv2.imwrite(f"{out_prefix}_overlay_gaps_oriented.png", debug_overlay)

    # === Debug visuel des gaps et du mask dilaté ===
    cv2.imwrite(f"{out_prefix}_mask_dilated.png", (seeds_dil * 255).astype(np.uint8))
    cv2.imwrite(f"{out_prefix}_gaps.png", (gaps * 255).astype(np.uint8))

    debug_overlay = img_ref.copy()
    debug_overlay[gaps > 0] = (0, 0, 255)          # gaps en rouge
    debug_overlay[seeds_dil > 0] = (0, 255, 0)     # mask dilaté en vert
    cv2.imwrite(f"{out_prefix}_overlay_gaps_debug.png", debug_overlay)

    # 9.4) bonus proportionnel à la proximité (distance transform sur les NON-seeds)
    nonseeds = (seeds == 0).astype(np.uint8)
    # distance euclidienne jusqu'à la détection la plus proche
    dist = cv2.distanceTransform(nonseeds, distanceType=cv2.DIST_L2, maskSize=3)
    # pondération décroissante avec la distance, bornée à gap_px
    with np.errstate(divide='ignore', invalid='ignore'):
        w = np.clip((gap_px - dist) / max(1e-6, gap_px), 0.0, 1.0)
    gap_weight = (w * (gaps > 0)).astype(np.float32)

    # 9.5) boost du score (dans [0,1]) + re-seuillage optionnel
    score_img_boost = np.clip(score_img + bonus_scale * gap_weight, 0.0, 1.0)

    # 9.6) fermer / relier pour obtenir des "zones"
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (close_len_px, close_len_px))
    zone_mask = cv2.morphologyEx((score_img_boost >= thr).astype(np.uint8), cv2.MORPH_CLOSE, k_close)
    # option: un léger élargissement pour un rendu en zones
    zone_mask = cv2.dilate(zone_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)), iterations=1)

    # 9.7) exports
    cv2.imwrite(f"{out_prefix}_score_boost.png", (score_img_boost*255).astype(np.uint8))
    cv2.imwrite(f"{out_prefix}_zones_mask.png", (zone_mask*255).astype(np.uint8))

    # visu overlay zones
    zones_bgr = cv2.cvtColor((zone_mask*255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    zones_bgr[:, :, 1:] = 0  # rouge
    overlay_zones = cv2.addWeighted(overlay_bgr, 1.0, zones_bgr, 0.5, 0)
    cv2.imwrite(f"{out_prefix}_overlay_zones.png", overlay_zones)


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
        r"C:\Users\makss\Git\Galatsi-Semester-Project\Results\Images\needed for script\frame_1_0004.png",  # REF
        r"C:\Users\makss\Git\Galatsi-Semester-Project\Results\Images\needed for script\frame_1_0005.png",
        r"C:\Users\makss\Git\Galatsi-Semester-Project\Results\Images\needed for script\frame_1_0006.png",
        r"C:\Users\makss\Git\Galatsi-Semester-Project\Results\Images\needed for script\frame_1_0312.png",
        r"C:\Users\makss\Git\Galatsi-Semester-Project\Results\Images\needed for script\frame_1_0314.png",
        r"C:\Users\makss\Git\Galatsi-Semester-Project\Results\Images\needed for script\frame_1_0319.png",
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
        out_prefix="Results/parking_detection/dwell_mult_4/test8_thr_0.85/parking_dwell_state_MULTI_REG",
        save_reg_debug=True
    )
