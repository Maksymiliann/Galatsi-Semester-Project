import re
from pathlib import Path
import numpy as np
import cv2

# (optionnel) barre de progression:
try:
    from tqdm import tqdm
except:
    def tqdm(x, **k): return x  # fallback silencieux


# -----------------------------
# Parsing utils
# -----------------------------
def parse_line_semicolon(line):
    """
    Format robuste attendu (exemples):
      vehicle_id; x1; y1; x2; y2; x3; y3; x4; y4; ...; conf; state
    - 1er champ = id
    - 8 floats = coins OBB
    - 'conf' (float) & 'state' (str) présents quelque part à la fin
    """
    parts = [p.strip() for p in line.strip().split(';') if p.strip() != ""]
    if len(parts) < 11:
        return None
    # id (tolère 0/vides initiaux)
    veh_id = None
    for k in range(min(3, len(parts)-10)):
        try:
            veh_id = int(float(parts[k]))
            parts = parts[k:]
            break
        except:
            continue
    if veh_id is None:
        return None

    # 8 floats pour OBB
    floats = []
    for p in parts[1:]:
        try:
            floats.append(float(p))
        except:
            break
    if len(floats) < 8:
        return None
    x1,y1,x2,y2,x3,y3,x4,y4 = floats[:8]

    # conf & state en fin
    state, conf = None, None
    for p in reversed(parts):
        if state is None and re.match(r'[A-Za-z]+', p):
            state = p.lower()
            continue
        if conf is None:
            try:
                conf = float(p); break
            except:
                continue
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


# -----------------------------
# Raster utils
# -----------------------------
def poly_to_cells(poly_xy, H, W, cell):
    """
    Remplit l'OBB dans une grille downsamplée (cell pixels).
    Retourne un masque bool (h_grid, w_grid).
    """
    h_grid, w_grid = int(np.ceil(H/cell)), int(np.ceil(W/cell))
    poly_grid = (poly_xy / cell).astype(np.float32)
    mask = np.zeros((h_grid, w_grid), dtype=np.uint8)
    cv2.fillPoly(mask, [poly_grid.astype(np.int32)], 1)
    return mask.astype(bool)


# -----------------------------
# Core
# -----------------------------
def run_parking_dwell_state(
    txt_dir: str,
    image_path: str,
    fps: float = 25.0,
    cell: int = 4,
    conf_min: float = 0.0,

    # Poids de la carte "popularité" (somme brute)
    w_stop: float = 1.0,        # poids du dwell (seconds in stop)
    w_move: float = 1.0,        # pénalité des passages en move (compte)

    # Mix avec une carte de ratio (invariante à la durée)
    ratio_weight: float = 0.6,  # λ : poids du ratio (0..1)
    w_move_ratio: float = 1.5,  # pondération MOVE dans le ratio

    # Cap par ID (protège les arrêts courts) — basé sur le centroïde (rapide)
    per_id_stop_cap_sec: float = 120.0,   # cap par véhicule ET par cellule (ex: 2 min)
    use_id_cap_for_move: bool = False,    # True si tu veux caper MOVE aussi (peu utile)
    per_id_move_cap_hits: float = 120.0,  # cap MOVE par ID si activé (en hits)
    mix_id_cap_weight: float = 0.5,       # 0=ignore cap-ID, 1=ne garde que cap-ID

    # Visu
    gaussian_sigma: float = 1.5, # lissage (en pixels image après upsampling)
    alpha_overlay: float = 0.6,  # opacité overlay JET
    thr: float = 0.70,           # seuil [0..1] sur la carte finale (mêmes couleurs/opacité)

    out_prefix: str = "parking_dwell_state"
):
    """
    Sorties:
      - {out_prefix}_score_map.png        (JET color, carte finale mixée)
      - {out_prefix}_overlay.png          (image + heatmap)
      - {out_prefix}_score_map_thresh.png (JET color, masquée sous le seuil)
      - {out_prefix}_overlay_thresh.png   (overlay masqué sous le seuil)
    Retourne:
      (score_img_float, overlay_bgr, heatmap_bgr, heatmap_thresh_bgr, overlay_thresh_bgr)
    """
    # 1) image / grilles
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    H, W = img.shape[:2]
    hG, wG = int(np.ceil(H/cell)), int(np.ceil(W/cell))

    # 2) cartes cumulatives "classiques"
    dwell_stop = np.zeros((hG, wG), dtype=np.float32)  # +dt sur pixels couverts par OBB stop
    move_hits  = np.zeros((hG, wG), dtype=np.float32)  # +1  sur pixels couverts par OBB move

    # 2b) accumulateurs par (veh_id, cellule) sur le CENTROÏDE (rapide)
    stop_idcell = {}  # key=(veh_id, lin_idx) -> dwell secondes (capé après)
    move_idcell = {}  # si use_id_cap_for_move=True

    txt_files = sorted(Path(txt_dir).glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files in {txt_dir}")

    dt = 1.0 / max(1e-6, float(fps))

    # 3) accumulation
    for fp in tqdm(txt_files, desc="Accumulating dwell/move (+ per-ID caps)", unit="file"):
        dets = read_txt_file(fp)
        if not dets: continue
        for veh_id, poly, conf, state in dets:
            if conf < conf_min:
                continue

            # A) accumulation "classique" sur polygone
            mask = poly_to_cells(poly, H, W, cell)
            if state.startswith("stop"):
                dwell_stop[mask] += dt
            else:
                move_hits[mask]  += 1.0  # (mettre += dt si tu préfères pénaliser en temps)

            # B) accumulation "par ID & cellule" via LE CENTROÏDE
            cx = float(poly[:,0].mean()); cy = float(poly[:,1].mean())
            i = int(np.floor(cy / cell)); j = int(np.floor(cx / cell))
            if 0 <= i < hG and 0 <= j < wG:
                idx = i * wG + j
                if state.startswith("stop"):
                    stop_idcell[(veh_id, idx)] = stop_idcell.get((veh_id, idx), 0.0) + dt
                elif use_id_cap_for_move:
                    move_idcell[(veh_id, idx)] = move_idcell.get((veh_id, idx), 0.0) + 1.0  # ou +dt

    # 4) carte STOP "capée par ID" (protège les arrêts courts)
    stop_cap_grid = np.zeros((hG, wG), dtype=np.float32)
    if per_id_stop_cap_sec is not None and per_id_stop_cap_sec > 0:
        capS = float(per_id_stop_cap_sec)
        for (_, idx), dwell in stop_idcell.items():
            stop_cap_grid.flat[idx] += min(dwell, capS)
    else:
        for (_, idx), dwell in stop_idcell.items():
            stop_cap_grid.flat[idx] += dwell

    # MOVE capé par ID (optionnel)
    if use_id_cap_for_move:
        move_cap_grid = np.zeros((hG, wG), dtype=np.float32)
        capM = float(per_id_move_cap_hits)
        for (_, idx), hits in move_idcell.items():
            move_cap_grid.flat[idx] += min(hits, capM)
    else:
        move_cap_grid = move_hits  # on garde la carte classique

    # 5) mix entre STOP classique et STOP capé-ID
    mix = float(np.clip(mix_id_cap_weight, 0.0, 1.0))
    dwell_mixed = (1.0 - mix) * dwell_stop + mix * stop_cap_grid

    # 6) score "popularité" (comme avant) puis normalisation robuste en 0..1
    popularity_grid = w_stop * dwell_mixed - w_move * move_cap_grid
    flat = popularity_grid.ravel()
    lo = np.percentile(flat, 2)
    hi = np.percentile(flat, 98)
    popularity_norm = np.clip((popularity_grid - lo) / max(1e-6, (hi - lo)), 0, 1)

    # 7) carte de RATIO (invariante à la durée)
    eps = 1e-6
    denom = dwell_mixed + (w_move_ratio * move_cap_grid) + eps
    ratio_grid = dwell_mixed / denom  # déjà dans [0,1]

    # 8) MIX final des deux cartes
    lam = float(np.clip(ratio_weight, 0.0, 1.0))
    score_grid = (1.0 - lam) * popularity_norm + lam * ratio_grid

    # 9) upsample à la taille image
    score_img = cv2.resize(score_grid.astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)

    # 10) lissage optionnel (visuel)
    if gaussian_sigma and gaussian_sigma > 0:
        ksize = int(max(3, 2*int(3*gaussian_sigma)+1))
        score_img = cv2.GaussianBlur(score_img, (ksize, ksize), gaussian_sigma)

    # 11) color map + overlay (couleurs/opacité identiques)
    score_uint8 = (np.clip(score_img, 0, 1) * 255).astype(np.uint8)
    heatmap_bgr  = cv2.applyColorMap(score_uint8, cv2.COLORMAP_JET)
    overlay_bgr  = cv2.addWeighted(img, 1.0, heatmap_bgr, alpha_overlay, 0)

    # 12) save (heatmap + overlay)
    pred_path = f"{out_prefix}_score_map.png"
    overlay_path = f"{out_prefix}_overlay.png"
    Path(Path(out_prefix).parent).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(pred_path, heatmap_bgr)
    cv2.imwrite(overlay_path, overlay_bgr)
    print(f"[OK] Saved score map: {pred_path}")
    print(f"[OK] Saved overlay:  {overlay_path}")

    # 13) Threshold avec mêmes couleurs/opacité (masque sous le seuil)
    mask = (score_img >= float(thr)).astype(np.uint8)
    mask3 = np.dstack([mask]*3)  # (H,W,3)
    heatmap_thresh = (heatmap_bgr * mask3).astype(np.uint8)
    overlay_thresh = cv2.addWeighted(img, 1.0, heatmap_thresh, alpha_overlay, 0)

    thr_map_path = f"{out_prefix}_score_map_thresh.png"
    thr_overlay_path = f"{out_prefix}_overlay_thresh.png"
    cv2.imwrite(thr_map_path, heatmap_thresh)
    cv2.imwrite(thr_overlay_path, overlay_thresh)
    print(f"[OK] Saved thresholded map: {thr_map_path}")
    print(f"[OK] Saved thresholded overlay: {thr_overlay_path}")

    return score_img.astype(np.float32), overlay_bgr, heatmap_bgr, heatmap_thresh, overlay_thresh


# -----------------------------
# Exemple d'utilisation directe depuis ton IDE
# -----------------------------
if __name__ == "__main__":
    TXT_DIR = r"C:/Users/makss/Git/Galatsi-Semester-Project/Results/TXT12"       # dossier 0001.txt ... N.txt
    IMAGE   = r"C:/Users/makss/Git/Galatsi-Semester-Project/first_frame_static.png"  # image drone correspondante

    run_parking_dwell_state(
        txt_dir=TXT_DIR,
        image_path=IMAGE,
        fps=25.0,
        cell=4,               # ↑ accélère, ↓ plus précis (4–10 selon besoin)
        conf_min=0.0,

        # Popularité (brut)
        w_stop=0.5,
        w_move=1.5,

        # Ratio
        ratio_weight=0.3,
        w_move_ratio=1.0,

        # Cap par ID
        per_id_stop_cap_sec=120.0,   # 2 min par véhicule et par cellule
        use_id_cap_for_move=False,   # garder le move "classique"
        per_id_move_cap_hits=120.0,
        mix_id_cap_weight=0.5,       # mélange 50/50 entre STOP classique et STOP capé-ID

        # Visu
        gaussian_sigma=1.5,
        alpha_overlay=0.6,
        thr=0.90,

        out_prefix="Results/parking_detection/dwell_mix/test5/parking_dwell_state"
    )
