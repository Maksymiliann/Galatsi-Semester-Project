import re
from pathlib import Path
import numpy as np
import cv2

try:
    from tqdm import tqdm
except:
    def tqdm(x, **k): return x  # fallback silencieux

# -----------------------------
# Parsing utils
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

    # score de base (somme pondérée)
    w_stop: float = 1.0,        # poids du dwell (seconds in stop)
    w_move: float = 1.0,        # pénalité move (hits par défaut)

    # visu
    gaussian_sigma: float = 1.5, # lissage (pixels image après upsampling)
    alpha_overlay: float = 0.6,

    # décision par SEUIL ABSOLU sur le score brut
    score_min_abs: float = 30.0,  # <- NOUVEAU: seuil absolu sur score_brut
    # (ex. si move_hits est en hits à 25 fps: 30 ~ 30 s de stop net vs move=0
    #  si tu passes move en secondes, ajuste en conséquence)

    out_prefix: str = "parking_dwell_state"
):
    """
    Sorties:
      - {out_prefix}_score_map.png          (JET color normalisée pour la VISU)
      - {out_prefix}_overlay.png            (image + heatmap)
      - {out_prefix}_score_map_thresh.png   (JET masquée par score_brut >= score_min_abs)
      - {out_prefix}_overlay_thresh.png     (overlay masqué par le même critère)
    Retourne: (score_img_float_norm, overlay_bgr, heatmap_bgr, heatmap_thresh_bgr, overlay_thresh_bgr)
    """
    # 1) image / grilles
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    H, W = img.shape[:2]
    hG, wG = int(np.ceil(H/cell)), int(np.ceil(W/cell))

    # 2) accumulateurs
    dwell_stop = np.zeros((hG, wG), dtype=np.float32)  # secondes en stop
    move_hits  = np.zeros((hG, wG), dtype=np.float32)  # hits (frames) en move

    txt_files = sorted(Path(txt_dir).glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files in {txt_dir}")

    dt = 1.0 / max(1e-6, float(fps))

    # 3) accumulation
    for fp in tqdm(txt_files, desc="Accumulating dwell/move", unit="file"):
        dets = read_txt_file(fp)
        if not dets: continue
        for _, poly, conf, state in dets:
            if conf < conf_min:
                continue
            mask = poly_to_cells(poly, H, W, cell)
            if state.startswith("stop"):
                dwell_stop[mask] += dt
            else:
                move_hits[mask]  += 1.0   # mets += dt si tu veux des secondes

    # 4) score BRUT (pas de percentile pour la décision)
    score_brut = w_stop * dwell_stop - w_move * move_hits

    # 5) VISU: on normalise juste pour produire une heatmap lisible (JET)
    flat = score_brut.ravel()
    lo = np.percentile(flat, 2)
    hi = np.percentile(flat, 98)
    score_norm = np.clip((score_brut - lo) / max(1e-6, (hi - lo)), 0, 1)

    # 6) upsample à la taille image (visu)
    score_img = cv2.resize(score_norm.astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)

    # 7) lissage optionnel (visu)
    if gaussian_sigma and gaussian_sigma > 0:
        ksize = int(max(3, 2*int(3*gaussian_sigma)+1))
        score_img = cv2.GaussianBlur(score_img, (ksize, ksize), gaussian_sigma)

    # 8) heatmap + overlay (visu)
    score_uint8 = (np.clip(score_img, 0, 1) * 255).astype(np.uint8)
    heatmap_bgr  = cv2.applyColorMap(score_uint8, cv2.COLORMAP_JET)
    overlay_bgr  = cv2.addWeighted(img, 1.0, heatmap_bgr, alpha_overlay, 0)

    Path(Path(out_prefix).parent).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(f"{out_prefix}_score_map.png",  heatmap_bgr)
    cv2.imwrite(f"{out_prefix}_overlay.png",    overlay_bgr)
    print(f"[OK] Saved score map & overlay (visual only).")

    # 9) DÉCISION: seuil ABSOLU sur le score BRUT
    decision_grid = (score_brut >= float(score_min_abs)).astype(np.uint8)

    # upsample binaire (nearest)
    decision_img = cv2.resize(decision_grid, (W, H), interpolation=cv2.INTER_NEAREST)

    # Heatmap JET masquée par la décision (mêmes couleurs/opacité)
    mask3 = np.dstack([decision_img]*3)
    heatmap_thresh = (heatmap_bgr * mask3).astype(np.uint8)
    overlay_thresh = cv2.addWeighted(img, 1.0, heatmap_thresh, alpha_overlay, 0)

    cv2.imwrite(f"{out_prefix}_score_map_thresh.png", heatmap_thresh)
    cv2.imwrite(f"{out_prefix}_overlay_thresh.png",   overlay_thresh)
    print(f"[OK] Saved absolute-thresholded map & overlay (score_brut >= {score_min_abs}).")

    return score_img.astype(np.float32), overlay_bgr, heatmap_bgr, heatmap_thresh, overlay_thresh


# Exemple d'utilisation
if __name__ == "__main__":
    TXT_DIR = r"C:/Users/makss/Git/Galatsi-Semester-Project/Results/TXT12"
    IMAGE   = r"C:/Users/makss/Git/Galatsi-Semester-Project/first_frame_static.png"

    run_parking_dwell_state(
        txt_dir=TXT_DIR,
        image_path=IMAGE,
        fps=25.0,
        cell=10,
        conf_min=0.0,
        w_stop=1.0,
        w_move=1.0,        # si tu passes move en secondes, ajuste w_move
        gaussian_sigma=1.5,
        alpha_overlay=0.6,
        score_min_abs=30.0,   # <-- règle ICI ton seuil absolu
        out_prefix="Results/parking_detection/dwell_test/test1/parking_dwell_state"
    )
