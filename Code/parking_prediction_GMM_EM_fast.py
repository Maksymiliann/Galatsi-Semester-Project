import os, re
from pathlib import Path
import numpy as np
import cv2
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# ---------- parsing identique ----------
def parse_line_semicolon(line):
    parts = [p.strip() for p in line.strip().split(';') if p.strip() != ""]
    if len(parts) < 11:
        return None
    veh_id = None
    for k in range(min(3, len(parts)-10)):
        try:
            veh_id = int(float(parts[k])); parts = parts[k:]; break
        except:
            continue
    if veh_id is None:
        return None

    floats = []
    for p in parts[1:]:
        try: floats.append(float(p))
        except: break
    if len(floats) < 8:
        return None
    x1,y1,x2,y2,x3,y3,x4,y4 = floats[:8]

    state, conf = None, None
    for p in reversed(parts):
        if state is None and re.match(r'[A-Za-z]+', p):
            state = p.lower(); continue
        if conf is None:
            try: conf = float(p); break
            except: continue
    if state is None: state = 'stop'
    if conf is None:  conf = 1.0
    poly = np.array([[x1,y1],[x2,y2],[x3,y3],[x4,y4]], dtype=np.float32)
    return veh_id, poly, conf, state

def read_txt_file(fp):
    dets = []
    with open(fp, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line=line.strip()
            if not line: continue
            parsed = parse_line_semicolon(line)
            if parsed: dets.append(parsed)
    return dets

# ---------- utils ----------
def polys_to_grid(polys_px, cell):
    """ Convertit une liste de polygones (en px image) en liste de polygones grille int32. """
    if not polys_px:
        return []
    polys_grid = []
    for P in polys_px:
        G = (P / cell).astype(np.float32)
        polys_grid.append(G.astype(np.int32))
    return polys_grid

def draw_many_fillPoly(mask_uint8, polys_grid):
    """ Remplit tous les polygones dans mask_uint8 (0/1). """
    if polys_grid:
        cv2.fillPoly(mask_uint8, polys_grid, 1)

# ---------- main rapide ----------
def run_parking_em_gmm_fast(
    txt_dir, image_path, fps=25.0, cell=4, conf_min=0.0,
    alpha=0.6, stop_weight=0.25, out_prefix="parking_result_fast",
    compute_U="none"  # "none" | "centroid"
):
    """
    Version optimisée :
      - 1 seul fillPoly par état et par frame (stop/move)
      - plus de np.where/zip par pixel
      - U optionnel (approx 'centroid' rapide)

    Retourne (P_img_float [0..1], overlay_bgr, heatmap_bgr)
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    H, W = img.shape[:2]
    hG, wG = int(np.ceil(H/cell)), int(np.ceil(W/cell))

    # Accumulateurs
    G = np.zeros((hG, wG), dtype=np.float32)   # compte stop
    R = np.zeros((hG, wG), dtype=np.float32)   # compte move
    D = np.zeros((hG, wG), dtype=np.float32)   # dwell stop (s)

    # U rapide (approx) par centroid: id -> set(linear_idx)
    id2cells = {} if compute_U == "centroid" else None

    txt_files = sorted(Path(txt_dir).glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {txt_dir}")

    dt = 1.0 / float(max(1e-6, fps))

    # Masques temporaires réutilisés (évite réalloc)
    stop_mask = np.zeros((hG, wG), dtype=np.uint8)
    move_mask = np.zeros((hG, wG), dtype=np.uint8)

    for fp in tqdm(txt_files, desc="Rasterizing frames (fast)", unit="file"):
        dets = read_txt_file(fp)
        if not dets:
            continue

        stop_polys_px = []
        move_polys_px = []

        if compute_U == "centroid":
            # on ne compte U que pour les arrêts
            for veh_id, poly, conf, state in dets:
                if conf < conf_min: 
                    continue
                if state.startswith("stop"):
                    cx = float(poly[:,0].mean()); cy = float(poly[:,1].mean())
                    i = int(np.floor(cy / cell)); j = int(np.floor(cx / cell))
                    if 0 <= i < hG and 0 <= j < wG:
                        s = id2cells.get(veh_id)
                        if s is None:
                            s = set(); id2cells[veh_id] = s
                        s.add(i*wG + j)

        # sépare listes pour dessin groupé
        for veh_id, poly, conf, state in dets:
            if conf < conf_min: 
                continue
            if state.startswith("stop"):
                stop_polys_px.append(poly)
            else:
                move_polys_px.append(poly)

        # remet à zéro les masques temp
        stop_mask.fill(0)
        move_mask.fill(0)

        # dessine tous les polygones d'un coup (grille)
        stop_polys_grid = polys_to_grid(stop_polys_px, cell)
        move_polys_grid = polys_to_grid(move_polys_px, cell)
        draw_many_fillPoly(stop_mask, stop_polys_grid)
        draw_many_fillPoly(move_mask, move_polys_grid)

        # accumulation vectorisée
        # cast en float32 une seule fois (évite where + boucles)
        G += (stop_mask * stop_weight)
        R += move_mask
        D += (stop_mask.astype(np.float32) * dt)

    # U final
    if compute_U == "centroid":
        U = np.zeros((hG, wG), dtype=np.float32)
        for _, cellset in id2cells.items():
            idxs = np.fromiter(cellset, dtype=np.int64)
            np.add.at(U.ravel(), idxs, 1.0)
        U = U.reshape(hG, wG)
    else:
        U = np.zeros((hG, wG), dtype=np.float32)

    # Features
    visited = (G + R) > 0
    idxs = np.where(visited)
    if idxs[0].size == 0:
        raise RuntimeError("No visited cells (G+R==0 everywhere).")

    X_feats, coords = [], []
    eps = 1e-6
    for i, j in zip(*idxs):
        g = G[i, j]; r = R[i, j]; d = D[i, j]; u = U[i, j]
        stop_ratio = g / (g + r + eps)
        move_ratio = r / (g + r + eps)
        X_feats.append([g, r, d, u, stop_ratio, np.log1p(d), np.log1p(u), move_ratio])
        coords.append((i, j))
    X_feats = np.asarray(X_feats, dtype=np.float32)

    # GMM
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X_feats)
    gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=0, init_params='kmeans', n_init=3)
    gmm.fit(Xz)
    probs = gmm.predict_proba(Xz)

    # choisir la composante "parking"
    means = []
    labels = np.argmax(probs, axis=1)
    for c in range(2):
        m = labels == c
        if m.sum() == 0:
            means.append(-1e9)
        else:
            sr = X_feats[m, 4]; logD = X_feats[m, 5]
            means.append(float((sr + logD).mean()))
    parking_comp = int(np.argmax(means))

    # carte proba grille -> image
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

    print("G median:", np.median(G[G>0]))
    print("R median:", np.median(R[R>0]))
    print("D median(s):", np.median(D[D>0]))

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
        stop_weight=0.25,
        out_prefix="Results/parking_detection/GMM_EM_fast/test4/parking_result_fast",
        compute_U="centroid"   # "none" (plus rapide) ou "centroid" (approx U)
    )

