import os
import re
from pathlib import Path
import numpy as np
import cv2
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm  # <-- ajoute ceci en haut du fichier (si pas déjà)

# -----------------------------
# Parsing utils
# -----------------------------
def parse_line_semicolon(line):
    parts = [p.strip() for p in line.strip().split(';') if p.strip() != ""]
    if len(parts) < 11:
        return None
    try:
        veh_id = int(float(parts[0]))
    except:
        for k in range(min(3, len(parts)-10)):
            try:
                veh_id = int(float(parts[k]))
                parts = parts[k:]
                break
            except:
                continue
        else:
            return None

    floats = []
    for p in parts[1:]:
        try:
            floats.append(float(p))
        except:
            break

    if len(floats) < 8:
        return None

    x1,y1,x2,y2,x3,y3,x4,y4 = floats[:8]

    state = None
    conf = None
    for p in reversed(parts):
        if state is None and re.match(r'[A-Za-z]+', p):
            state = p.lower()
            continue
        if conf is None:
            try:
                conf = float(p)
                break
            except:
                continue
    if state is None or conf is None:
        state = 'stop'
        conf = 1.0

    poly = np.array([[x1,y1],[x2,y2],[x3,y3],[x4,y4]], dtype=np.float32)
    return veh_id, poly, conf, state


def read_txt_file(fp):
    dets = []
    with open(fp, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line=line.strip()
            if not line:
                continue
            parsed = parse_line_semicolon(line)
            if parsed is None:
                continue
            dets.append(parsed)
    return dets


def poly_to_cells(poly_xy, H, W, cell):
    h_grid, w_grid = int(np.ceil(H/cell)), int(np.ceil(W/cell))
    poly_grid = (poly_xy / cell).astype(np.float32)
    mask = np.zeros((h_grid, w_grid), dtype=np.uint8)
    cv2.fillPoly(mask, [poly_grid.astype(np.int32)], 1)
    return mask.astype(bool)


# -----------------------------
# Main function
# -----------------------------
def run_parking_em_gmm(txt_dir, image_path, fps=25.0, cell=4, conf_min=0.0,
                       alpha=0.6, out_prefix="parking_result"):
    """
    Exécute l'analyse EM/GMM sur les fichiers txt + image.
    Retourne (P_img, overlay_img, cmap_img)
    """
    # Lecture image
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    H, W = img.shape[:2]
    h_grid, w_grid = int(np.ceil(H/cell)), int(np.ceil(W/cell))

    G = np.zeros((h_grid, w_grid), dtype=np.float32)
    R = np.zeros((h_grid, w_grid), dtype=np.float32)
    D = np.zeros((h_grid, w_grid), dtype=np.float32)
    U_sets = dict()

    txt_files = sorted(Path(txt_dir).glob('*.txt'))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {txt_dir}")

    dt = 1.0 / float(fps)

    # Boucle principale avec barre de progression
    for k, fp in enumerate(tqdm(txt_files, desc="Processing frames", unit="file")):
        dets = read_txt_file(fp)
        if not dets:
            continue
        for veh_id, poly, conf, state in dets:
            if conf < conf_min:
                continue
            mask = poly_to_cells(poly, H, W, cell)
            where = np.where(mask)
            if state.startswith('stop'):
                G[where] += 1.0
                D[where] += dt
                for i, j in zip(*where):
                    key = (int(i), int(j))
                    s = U_sets.get(key)
                    if s is None:
                        s = set()
                        U_sets[key] = s
                    s.add(veh_id)
            else:
                R[where] += 1.0

    # affichage texte simple (optionnel)
    if (k + 1) % 100 == 0:
        print(f"Processed {k+1}/{len(txt_files)} files")


    U = np.zeros((h_grid, w_grid), dtype=np.float32)
    for (i,j), s in U_sets.items():
        U[i,j] = float(len(s))

    visited = (G+R) > 0
    idxs = np.where(visited)
    if idxs[0].size == 0:
        raise RuntimeError("No visited cells (G+R==0 everywhere).")

    X_feats = []
    coords = []
    eps = 1e-6
    for i,j in zip(*idxs):
        g = G[i,j]; r = R[i,j]; d = D[i,j]; u = U[i,j]
        stop_ratio = g / (g + r + eps)
        move_ratio = r / (g + r + eps)
        X_feats.append([g, r, d, u, stop_ratio, np.log1p(d), np.log1p(u), move_ratio])
        coords.append((i,j))
    X_feats = np.asarray(X_feats, dtype=np.float32)

    # GMM
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X_feats)
    gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=0, init_params='kmeans', n_init=3)
    gmm.fit(Xz)
    probs = gmm.predict_proba(Xz)

    # Choisir composante parking
    means_by_comp = []
    labels = np.argmax(probs, axis=1)
    for c in range(2):
        mask = labels==c
        if mask.sum()==0:
            means_by_comp.append(-1e9)
        else:
            stop_ratio = X_feats[mask, 4]
            logD = X_feats[mask, 5]
            means_by_comp.append(float((stop_ratio + logD).mean()))
    parking_comp = int(np.argmax(means_by_comp))

    # Carte probas
    P_grid = np.zeros((h_grid, w_grid), dtype=np.float32)
    for (i,j), p in zip(coords, probs[:, parking_comp]):
        P_grid[i,j] = float(p)

    # Remonter à la taille image
    P_img = cv2.resize(P_grid, (W, H), interpolation=cv2.INTER_CUBIC)
    P_img = np.clip(P_img, 0, 1).astype(np.float32)

    # Colorisation
    P_uint8 = (P_img*255).astype(np.uint8)
    cmap_img = cv2.applyColorMap(P_uint8, cv2.COLORMAP_JET)
    overlay_img = cv2.addWeighted(img, 1.0, cmap_img, alpha, 0)

    # Sauvegarde
    pred_path = f"{out_prefix}_prediction_map.png"
    overlay_path = f"{out_prefix}_overlay.png"
    cv2.imwrite(pred_path, cmap_img)
    cv2.imwrite(overlay_path, overlay_img)
    print(f"[OK] Saved prediction map: {pred_path}")
    print(f"[OK] Saved overlay:        {overlay_path}")

    return P_img, overlay_img, cmap_img


# Exemple d'utilisation directe depuis ton IDE :
if __name__ == "__main__":
    txt_folder = r"C:/Users/makss/Git/Galatsi-Semester-Project/Results/TXT12"       # <---- ton dossier avec 0001.txt … 7519.txt
    image_path = r"C:/Users/makss/Git/Galatsi-Semester-Project/first_frame_static.png"  # <---- ton image drone
    run_parking_em_gmm(
        txt_dir=txt_folder,
        image_path=image_path,
        fps=25.0,
        cell=4,
        conf_min=0.0,
        alpha=0.6,
        out_prefix="Results/parking_detection/GMM_EM/parking_result"
    )
