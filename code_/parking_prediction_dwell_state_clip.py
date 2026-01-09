import re
from pathlib import Path
import numpy as np
import cv2

# (optionnel) barre de progression:
try:
    from tqdm import tqdm
except:
    def tqdm(x, **k): return x  # fallback silencieux


"""
This script builds a parking heatmap from detection TXT files by accumulating STOP dwell time and penalizing MOVE activity.

It parses semicolon-separated lines to extract (vehicle_id, 4-corner OBB polygon, confidence, state). Each polygon
is rasterized onto a coarse grid (cell pixels per grid cell) for speed. For every frame:
- STOP detections add +dt seconds to dwell_stop on the covered cells
- MOVE detections add +1 “hit” to move_hits on the covered cells

Before normalization, both accumulators can be clipped to avoid extreme values dominating the map:
- stop_cap_sec caps dwell time per cell
- move_cap_hits caps the number of move hits per cell
A signed score is then computed: score = w_stop * dwell_capped - w_move * move_capped
(optionally clipped again with score_cap_abs), and robustly normalized to [0,1] using 2–98% percentiles.

The score map is upsampled back to image resolution, optionally smoothed with a Gaussian blur, converted to a JET
colormap, and overlaid on the reference image. A thresholded version (thr) is also saved, keeping the same colors
and overlay opacity but masking out values below the threshold.

Outputs:
- *_score_map.png and *_overlay.png (continuous heatmap)
- *_score_map_thresh.png and *_overlay_thresh.png (thresholded heatmap)
"""


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
    w_stop: float = 1.0,        # poids du dwell (seconds in stop)
    w_move: float = 1.0,        # pénalité des passages en move (compte)
    gaussian_sigma: float = 1.5, # lissage (en pixels image après upsampling)
    alpha_overlay: float = 0.6,
    thr: float = 0.70,              # <- seuil [0..1]
    stop_cap_sec: float | None = 60.0,   # sature le dwell à 60 s par cellule
    move_cap_hits: float | None = 200.0, # sature les passages move (en “hits”)
    score_cap_abs: float | None = None,  # sature ensuite le score final à ±cap
    out_prefix: str = "parking_dwell_state"
):
    """
    Sorties:
      - {out_prefix}_score_map.png (JET color)
      - {out_prefix}_overlay.png  (image + heatmap)
      - Retourne (score_img_float, overlay_bgr, heatmap_bgr)
    """
    # 1) image / grilles
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    H, W = img.shape[:2]
    hG, wG = int(np.ceil(H/cell)), int(np.ceil(W/cell))

    # 2) cartes cumulatives
    # dwell stop en secondes + pénalité move (compte de recouvrement)
    dwell_stop = np.zeros((hG, wG), dtype=np.float32)  # cumule dt sur pixels couverts par OBB en état stop
    move_hits  = np.zeros((hG, wG), dtype=np.float32)  # cumule 1 par frame sur pixels couverts par OBB en move

    txt_files = sorted(Path(txt_dir).glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files in {txt_dir}")

    dt = 1.0 / max(1e-6, float(fps))

    # 3) accumulation simple dwell/penalty
    for fp in tqdm(txt_files, desc="Accumulating dwell/move", unit="file"):
        dets = read_txt_file(fp)
        if not dets: continue
        for veh_id, poly, conf, state in dets:
            if conf < conf_min: 
                continue
            mask = poly_to_cells(poly, H, W, cell)
            if state.startswith("stop"):
                dwell_stop[mask] += dt
            else:
                move_hits[mask]  += 1.0

     # 4) CLIPPING avant normalisation
    #    A) on sature chaque terme pour éviter que les très longs arrêts / gros flux dominent
    dwell_c = np.minimum(dwell_stop, stop_cap_sec) if (stop_cap_sec is not None) else dwell_stop
    move_c  = np.minimum(move_hits,  move_cap_hits) if (move_cap_hits  is not None) else move_hits

    #    B) on calcule le score signé
    score_grid = w_stop * dwell_c - w_move * move_c

    #    C) (option) sature aussi le score final de façon symétrique
    if score_cap_abs is not None:
        score_grid = np.clip(score_grid, -float(score_cap_abs), float(score_cap_abs))

    # 5) normalisation min-max robuste (clamp aux percentiles)
    flat = score_grid.flatten()
    lo = np.percentile(flat, 2)
    hi = np.percentile(flat, 98)
    score_grid = np.clip((score_grid - lo) / max(1e-6, (hi - lo)), 0, 1)

    # 6) upsample à la taille image
    score_img = cv2.resize(score_grid, (W, H), interpolation=cv2.INTER_CUBIC)

    # 7) lissage optionnel
    if gaussian_sigma and gaussian_sigma > 0:
        # sigma exprimé en pixels image
        ksize = int(max(3, 2*int(3*gaussian_sigma)+1))
        score_img = cv2.GaussianBlur(score_img, (ksize, ksize), gaussian_sigma)

    # 8) color map + overlay
    score_uint8 = (np.clip(score_img, 0, 1) * 255).astype(np.uint8)
    heatmap_bgr  = cv2.applyColorMap(score_uint8, cv2.COLORMAP_JET)
    overlay_bgr  = cv2.addWeighted(img, 1.0, heatmap_bgr, alpha_overlay, 0)

    # 9) save
    pred_path = f"{out_prefix}_score_map.png"
    overlay_path = f"{out_prefix}_overlay.png"
    cv2.imwrite(pred_path, heatmap_bgr)
    cv2.imwrite(overlay_path, overlay_bgr)
    print(f"[OK] Saved score map: {pred_path}")
    print(f"[OK] Saved overlay:  {overlay_path}")

        # --- Threshold avec mêmes couleurs/opacité ---
    # mask binaire
    mask = (score_img >= float(thr)).astype(np.uint8)
    mask3 = np.dstack([mask]*3)  # (H,W,3)

    # heatmap JET, mais coupée sous le seuil (noir ailleurs)
    heatmap_thresh = (heatmap_bgr * mask3).astype(np.uint8)

    # overlay identique (même alpha), mais uniquement là où mask==1
    overlay_thresh = cv2.addWeighted(img, 1.0, heatmap_thresh, alpha_overlay, 0)

    # save
    thr_map_path = f"{out_prefix}_score_map_thresh.png"
    thr_overlay_path = f"{out_prefix}_overlay_thresh.png"
    cv2.imwrite(thr_map_path, heatmap_thresh)
    cv2.imwrite(thr_overlay_path, overlay_thresh)
    print(f"[OK] Saved thresholded map: {thr_map_path}")
    print(f"[OK] Saved thresholded overlay: {thr_overlay_path}")

    return score_img.astype(np.float32), overlay_bgr, heatmap_bgr


# Exemple d'utilisation directe depuis ton IDE:
if __name__ == "__main__":
    TXT_DIR = r"C:/Users/makss/Git/Galatsi-Semester-Project/Results/TXT12"       # dossier 0001.txt ... N.txt
    IMAGE   = r"C:/Users/makss/Git/Galatsi-Semester-Project/first_frame_static.png"  # image drone correspondante

    run_parking_dwell_state(
        txt_dir=TXT_DIR,
        image_path=IMAGE,
        fps=25.0,
        cell=10,            # grille plus fine -> plus précis mais plus lent
        conf_min=0.0,
        w_stop=1.0,        # augmente si tu veux favoriser le dwell
        w_move=1.0,        # augmente si tu veux punir plus les zones de passage
        gaussian_sigma=1.5,# lissage visuel
        alpha_overlay=0.6,
        thr = 0.9,
        stop_cap_sec=60.0,     # cap dwell
        move_cap_hits=200.0,   # cap move
        score_cap_abs=300,    # (optionnel) cap final
        out_prefix="Results/parking_detection/dwell_clip/test1/parking_dwell_state"
    )
