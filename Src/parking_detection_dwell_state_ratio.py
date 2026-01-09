import re
from pathlib import Path
import numpy as np
import cv2

try:
    from tqdm import tqdm
except:
    def tqdm(x, **k): return x  # fallback silencieux

"""
This script builds a “stop-dominance” heatmap from a folder of detection TXT files by comparing STOP vs MOVE
occupancy over space.

It parses each TXT line (semicolon-separated) to extract: vehicle_id, the 4-corner polygon, confidence score,
and the motion state (defaulting to "stop" if missing). Each polygon is rasterized onto a coarse grid
(cell x cell pixels) to speed up accumulation.

Two accumulators are built on that grid:
- stop_acc: how often (or how long) polygons labeled STOP cover each cell
- move_acc: how often (or how long) polygons labeled MOVE cover each cell
If use_time=True, contributions are weighted by 1/fps (seconds). Otherwise they are simple “hit” counts.
MOVE can be penalized more in the denominator using move_weight.

The final score is a per-cell ratio in [0,1]:
    score = stop_acc / (stop_acc + move_weight * move_acc + eps)
It is then upsampled to full image resolution, optionally smoothed with a Gaussian blur, converted to a JET
colormap, and overlaid on the reference image.

Outputs:
- *_score_map.png and *_overlay.png (continuous ratio heatmap)
- *_score_map_thresh.png and *_overlay_thresh.png (same heatmap thresholded at thr)
"""



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
# Core (ratio stop / (stop + move))
# -----------------------------
def run_parking_stop_move_ratio(
    txt_dir: str,
    image_path: str,
    fps: float = 25.0,
    cell: int = 4,
    conf_min: float = 0.0,
    use_time: bool = False,        # True => accumule en secondes, False => en "hits"
    move_weight: float = 1.0,      # pondère MOVE dans le dénominateur
    gaussian_sigma: float = 1.5,   # lissage visuel
    alpha_overlay: float = 0.6,    # opacité overlay JET
    thr: float = 0.70,             # seuil sur la carte ratio (0..1)
    out_prefix: str = "parking_ratio"
):
    """
    Sorties:
      - {out_prefix}_score_map.png       (JET)
      - {out_prefix}_overlay.png         (overlay JET)
      - {out_prefix}_score_map_thresh.png
      - {out_prefix}_overlay_thresh.png
    Retourne: (score_img_float, overlay_bgr, heatmap_bgr, heatmap_thresh_bgr, overlay_thresh_bgr)
    """
    # 1) image / grilles
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    H, W = img.shape[:2]
    hG, wG = int(np.ceil(H/cell)), int(np.ceil(W/cell))

    # 2) accumulateurs
    stop_acc = np.zeros((hG, wG), dtype=np.float32)
    move_acc = np.zeros((hG, wG), dtype=np.float32)

    txt_files = sorted(Path(txt_dir).glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files in {txt_dir}")

    dt = 1.0 / max(1e-6, float(fps))
    add_stop = (lambda M: (stop_acc.__iadd__(M.astype(np.float32) * (dt if use_time else 1.0))))
    add_move = (lambda M: (move_acc.__iadd__(M.astype(np.float32) * (dt if use_time else 1.0))))

    # 3) accumulation
    for fp in tqdm(txt_files, desc="Accumulating stop/move for ratio", unit="file"):
        dets = read_txt_file(fp)
        if not dets: continue
        for _, poly, conf, state in dets:
            if conf < conf_min: 
                continue
            mask = poly_to_cells(poly, H, W, cell)
            if state.startswith("stop"):
                add_stop(mask)
            else:
                add_move(mask)

    # 4) score ratio (déjà dans [0,1])
    eps = 1e-6
    denom = stop_acc + (move_weight * move_acc) + eps
    score_grid = stop_acc / denom

    # (pas de normalisation min-max ici pour garder l'invariance d'échelle)
    # 5) upsample image size
    score_img = cv2.resize(score_grid, (W, H), interpolation=cv2.INTER_CUBIC)

    # 6) lissage optionnel
    if gaussian_sigma and gaussian_sigma > 0:
        ksize = int(max(3, 2*int(3*gaussian_sigma)+1))
        score_img = cv2.GaussianBlur(score_img, (ksize, ksize), gaussian_sigma)

    # 7) colormap + overlay (identiques à avant)
    score_u8 = (np.clip(score_img, 0, 1) * 255).astype(np.uint8)
    heatmap_bgr  = cv2.applyColorMap(score_u8, cv2.COLORMAP_JET)
    overlay_bgr  = cv2.addWeighted(img, 1.0, heatmap_bgr, alpha_overlay, 0)

    cv2.imwrite(f"{out_prefix}_score_map.png", heatmap_bgr)
    cv2.imwrite(f"{out_prefix}_overlay.png",   overlay_bgr)

    # 8) threshold avec mêmes couleurs/opacité
    mask = (score_img >= float(thr)).astype(np.uint8)
    mask3 = np.dstack([mask]*3)
    heatmap_thresh = (heatmap_bgr * mask3).astype(np.uint8)
    overlay_thresh = cv2.addWeighted(img, 1.0, heatmap_thresh, alpha_overlay, 0)

    cv2.imwrite(f"{out_prefix}_score_map_thresh.png", heatmap_thresh)
    cv2.imwrite(f"{out_prefix}_overlay_thresh.png",   overlay_thresh)

    print("[OK] Saved:",
          f"{out_prefix}_score_map.png, {out_prefix}_overlay.png,",
          f"{out_prefix}_score_map_thresh.png, {out_prefix}_overlay_thresh.png")

    return score_img.astype(np.float32), overlay_bgr, heatmap_bgr, heatmap_thresh, overlay_thresh


# Exemple d'utilisation directe
if __name__ == "__main__":
    TXT_DIR = r"C:/Users/makss/Git/Galatsi-Semester-Project/Results/TXT12"
    IMAGE   = r"C:/Users/makss/Git/Galatsi-Semester-Project/first_frame_static.png"

    run_parking_stop_move_ratio(
        txt_dir=TXT_DIR,
        image_path=IMAGE,
        fps=25.0,
        cell=20,
        conf_min=0.0,
        use_time=False,       # mets True pour ratio en secondes
        move_weight=4.0,      # >1 pour pénaliser plus le MOVE
        gaussian_sigma=1.5,
        alpha_overlay=0.6,
        thr=0.90,             # garde seulement les zones très "stop-dominant"
        out_prefix="Results/parking_detection/dwell_ratio/test3/parking_ratio"
    )
