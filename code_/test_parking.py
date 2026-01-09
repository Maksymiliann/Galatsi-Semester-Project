import os, re
from collections import deque
from pathlib import Path
import numpy as np
import cv2
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

"""
Parking heatmap estimation with an EM/GMM model, extended with “anti-traffic-light” features.

The script rasterizes STOP and MOVE detections onto a coarse grid and accumulates base signals:
- G: STOP presence count per cell
- R: MOVE presence count per cell (weighted by move_weight)
- D: STOP dwell time per cell (seconds)

It then adds extra features designed to reduce false positives from traffic lights / intersections:
- D_long: long continuous STOP dwell (run-length >= T_long)
- Burst: frames where many STOP pixels appear locally (box-filter neighborhood >= N_burst)
- S2M: STOP→MOVE transition indicator using a sliding window of recent STOP masks (W_move seconds)

These features (plus ratios/log terms) are standardized and fed to a 2-component GaussianMixture (EM). Posterior
probabilities are optionally smoothed with a temperature parameter, and the “parking” component is selected as the
cluster with higher average stop_ratio + long-dwell score.

The final parking probability map is written back to an image-sized heatmap (JET) and an overlay is saved.
"""



def parse_line_semicolon(line):
    parts = [p.strip() for p in line.strip().split(';') if p.strip() != ""]
    if len(parts) < 11: return None
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
        if state is None and re.match(r'[A-Za-z]+', p): state = p.lower(); continue
        if conf  is None:
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

def polys_to_grid(polys_px, cell):
    if not polys_px: return []
    out=[]
    for P in polys_px:
        G = (P / cell).astype(np.float32).astype(np.int32)
        out.append(G)
    return out

def draw_many(mask_uint8, polys_grid):
    if polys_grid:
        cv2.fillPoly(mask_uint8, polys_grid, 1)

def run_parking_em_gmm_fast(
    txt_dir, image_path,
    fps=25.0, cell=4, conf_min=0.0,
    alpha=0.6, out_prefix="parking_result_fast",
    compute_U="none",
    move_weight=3.0,                 # pénalité move
    exclude_stop_if_move=True,       # move annule le stop même frame
    # --- nouveautés anti-feux ---
    T_long=90.0,                     # seuil long-dwell (s)
    N_burst=3,                       # nb véhicules stop simultanés (local)
    burst_kernel_cells=7,            # taille voisinage (grille) pour le burst
    W_move=8.0,                      # fenêtre (s) "stop->move rapide"
    covariance_type='tied',          # GMM plus "large"
    reg_covar=1e-3,                  # gonfle les covariances
    temp=2.0                         # adoucit les proba
):
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None: raise FileNotFoundError(f"Cannot read image: {image_path}")
    H, W = img.shape[:2]
    hG, wG = int(np.ceil(H/cell)), int(np.ceil(W/cell))

    # Accumulateurs de base
    G = np.zeros((hG, wG), dtype=np.float32)   # frames stop
    R = np.zeros((hG, wG), dtype=np.float32)   # frames move (pondérées)
    D = np.zeros((hG, wG), dtype=np.float32)   # dwell total stop (s)

    # Nouveaux accumulateurs anti-feux
    D_long = np.zeros((hG, wG), dtype=np.float32)   # dwell long (>= T_long)
    Burst  = np.zeros((hG, wG), dtype=np.float32)   # frames avec burst simultané
    S2M    = np.zeros((hG, wG), dtype=np.float32)   # stop->move rapide

    # run-length de stop (contigu) pour détecter long-dwell
    run_stop_sec = np.zeros((hG, wG), dtype=np.float32)

    # mémoire pour "stop récent" (fenêtre glissante)
    W_frames = max(1, int(round(W_move * fps)))
    past_stop_sum = np.zeros((hG, wG), dtype=np.int16)
    ring = deque(maxlen=W_frames)

    # U approx (centroid) si demandé
    id2cells = {} if compute_U == "centroid" else None

    txt_files = sorted(Path(txt_dir).glob("*.txt"))
    if not txt_files: raise FileNotFoundError(f"No .txt files in {txt_dir}")

    dt = 1.0 / float(max(1e-6, fps))
    k = burst_kernel_cells
    k = k + 1 if k % 2 == 0 else k  # noyau impair

    stop_mask = np.zeros((hG, wG), dtype=np.uint8)
    move_mask = np.zeros((hG, wG), dtype=np.uint8)

    for fp in tqdm(txt_files, desc="Rasterizing + anti-light features", unit="file"):
        dets = read_txt_file(fp)
        if not dets: 
            # même si vide, fais avancer la fenêtre stop->move
            ring.append(np.zeros_like(stop_mask))
            if len(ring) == ring.maxlen:
                oldest = ring[0]
                past_stop_sum -= oldest
            past_stop_sum += ring[-1]
            continue

        stop_polys_px, move_polys_px = [], []

        if compute_U == "centroid":
            for veh_id, poly, conf, state in dets:
                if conf < conf_min: continue
                if state.startswith("stop"):
                    cx = float(poly[:,0].mean()); cy = float(poly[:,1].mean())
                    i = int(np.floor(cy / cell)); j = int(np.floor(cx / cell))
                    if 0 <= i < hG and 0 <= j < wG:
                        s = id2cells.get(veh_id)
                        if s is None: s=set(); id2cells[veh_id]=s
                        s.add(i*wG + j)

        for _, poly, conf, state in dets:
            if conf < conf_min: continue
            (stop_polys_px if state.startswith("stop") else move_polys_px).append(poly)

        stop_mask.fill(0); move_mask.fill(0)
        draw_many(stop_mask, polys_to_grid(stop_polys_px, cell))
        draw_many(move_mask, polys_to_grid(move_polys_px, cell))

        if exclude_stop_if_move:
            stop_mask[move_mask == 1] = 0

        # ---- Accumulations de base
        G += stop_mask
        R += move_mask.astype(np.float32) * move_weight
        D += stop_mask.astype(np.float32) * dt

        # ---- Long-dwell (run-length contigu)
        run_stop_sec = np.where(stop_mask == 1, run_stop_sec + dt, 0.0)
        D_long += (run_stop_sec >= T_long) * dt

        # ---- Burst simultané: comptage local des stops par convolution
        #     (compte nb de pixels stop dans un voisinage kxk ; surrogate simple et rapide)
        local_cnt = cv2.boxFilter(stop_mask, ddepth=-1, ksize=(k, k), normalize=False)
        Burst += (local_cnt >= N_burst).astype(np.float32)

        # ---- Stop->Move rapide: somme glissante de stops récents vs move courant
        # update fenêtre
        if len(ring) == ring.maxlen:
            oldest = ring[0]      # avant append (deque maxlen écrase après append)
            past_stop_sum -= oldest
        ring.append(stop_mask.copy())
        past_stop_sum += ring[-1]
        S2M += ((move_mask == 1) & (past_stop_sum > 0)).astype(np.float32)

    # U final (approx centroid)
    if compute_U == "centroid":
        U = np.zeros((hG, wG), dtype=np.float32)
        for _, cellset in id2cells.items():
            idxs = np.fromiter(cellset, dtype=np.int64)
            np.add.at(U.ravel(), idxs, 1.0)
        U = U.reshape(hG, wG)
    else:
        U = np.zeros((hG, wG), dtype=np.float32)

    # -------- Features pour GMM --------
    visited = (G + R) > 0
    ii, jj = np.where(visited)
    if ii.size == 0:
        raise RuntimeError("No visited cells (G+R==0).")

    X, coords = [], []
    eps = 1e-6
    for i, j in zip(ii, jj):
        g, r, d, u = G[i,j], R[i,j], D[i,j], U[i,j]
        dl, b, s2m   = D_long[i,j], Burst[i,j], S2M[i,j]
        stop_ratio   = g / (g + r + eps)
        move_ratio   = r / (g + r + eps)
        # On met les signaux anti-feu pour que le GMM voie une seconde dimension claire
        X.append([g, r, d, u, stop_ratio, np.log1p(d), np.log1p(u), move_ratio,
                  np.log1p(dl), b, s2m])
        coords.append((i, j))
    X = np.asarray(X, dtype=np.float32)

    # -------- GMM + adoucissement --------
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)
    gmm = GaussianMixture(
        n_components=2, covariance_type=covariance_type, reg_covar=reg_covar,
        init_params='kmeans', n_init=3, random_state=0
    )
    gmm.fit(Xz)
    # temp-softmax pour lisser
    probs = gmm.predict_proba(Xz)
    probs = probs ** (1.0 / float(temp))
    probs /= probs.sum(axis=1, keepdims=True)

    # choisir composante "parking" (favorise stop_ratio + long-dwell)
    labels = np.argmax(probs, axis=1)
    sc = []
    for c in range(2):
        m = labels == c
        if m.sum() == 0: sc.append(-1e9); continue
        sc.append(float((X[m, 4] + X[m, 8]).mean()))  # stop_ratio + log1p(D_long)
    parking_comp = int(np.argmax(sc))

    # -------- Carte proba --------
    P_grid = np.zeros((hG, wG), dtype=np.float32)
    for (i, j), p in zip(coords, probs[:, parking_comp]):
        P_grid[i, j] = float(p)
    P_img = cv2.resize(P_grid, (W, H), interpolation=cv2.INTER_CUBIC)
    P_img = np.clip(P_img, 0, 1).astype(np.float32)

    P_uint8 = (P_img * 255).astype(np.uint8)
    cmap_img = cv2.applyColorMap(P_uint8, cv2.COLORMAP_JET)
    overlay_img = cv2.addWeighted(img, 1.0, cmap_img, alpha, 0)

    Path(os.path.dirname(out_prefix) or ".").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(f"{out_prefix}_prediction_map.png", cmap_img)
    cv2.imwrite(f"{out_prefix}_overlay.png", overlay_img)
    print(f"[OK] Saved prediction map: {out_prefix}_prediction_map.png")
    print(f"[OK] Saved overlay:        {out_prefix}_overlay.png")

    return P_img, overlay_img, cmap_img

if __name__ == "__main__":
    txt_folder = r"C:/Users/makss/Git/Galatsi-Semester-Project/Results/TXT12"
    image_path = r"C:/Users/makss/Git/Galatsi-Semester-Project/first_frame_static.png"

    run_parking_em_gmm_fast(
        txt_dir=txt_folder,
        image_path=image_path,
        fps=25.0,
        cell=4,
        conf_min=0.0,
        alpha=0.6,
        out_prefix="Results/parking_detection/GMM_EM_test/test3/parking_fast_plus",
        compute_U="none",
        move_weight=4.0,           # pénalise plus la circulation
        exclude_stop_if_move=True,
        T_long=90.0,               # augmente si les feux durent longtemps
        N_burst=7,                 # >=3 véhicules stoppés en même temps
        burst_kernel_cells=9,      # voisinage (9 cellules grille)
        W_move=8.0,                # "stop->move" dans les 8 secondes
        covariance_type='tied',
        reg_covar=1e-1,
        temp=2.0
    )
