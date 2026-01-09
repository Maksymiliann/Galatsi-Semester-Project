import math
import re
from pathlib import Path
from functools import lru_cache

import numpy as np
import cv2
from tqdm import tqdm

"""
This script generates a parking “likelihood” map from detection TXT files using oriented Gaussian splatting.

It parses each detection line (semicolon-separated) to extract a vehicle id, a 4-corner oriented bounding box
polygon, a confidence score, and a motion state (defaulting to "stop" if missing). For each detection, the OBB
is converted into a rotated rectangle (center, length, width, angle) using cv2.minAreaRect.

Instead of hard rasterizing polygons, the script “splats” an anisotropic rotated Gaussian kernel onto a coarse
grid (downsampled by cell):
- STOP detections add a positive Gaussian (+w_stop)
- MOVE detections subtract a Gaussian (-w_move)
The Gaussian is elongated along the vehicle’s main axis (extend_long) and its spread is controlled by sigma_*_ratio.
Rotated kernels are cached (LRU cache) to avoid regenerating similar kernels repeatedly.

All contributions are accumulated into a signed score grid, which is then robustly normalized to [0,1] using
percentiles (2–98%) and upsampled back to full image resolution. The resulting probability-like map is saved as
a JET heatmap and an overlay on the reference image. A thresholded binary map (thr) is also produced and saved
with a colored overlay for quick visual inspection.
"""


# -----------------------------
# Parsing .txt
# -----------------------------
def parse_line_semicolon(line):
    """
    Attend un format 'id; x1; y1; x2; y2; x3; y3; x4; y4; ...; conf; state'
    Robuste aux colonnes en trop. Renvoie (veh_id, poly(4x2), conf, state)
    """
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
    if conf  is None: conf  = 1.0
    poly = np.array([[x1,y1],[x2,y2],[x3,y3],[x4,y4]], dtype=np.float32)
    return veh_id, poly, conf, state


def read_txt_file(fp):
    out = []
    with open(fp, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            p = parse_line_semicolon(line)
            if p: out.append(p)
    return out


# -----------------------------
# Geometry utils (OBB -> center/size/angle)
# -----------------------------
def obb_to_rect(poly):
    """
    poly: (4,2) float32 (x,y) corners
    returns: (cx, cy, w, h, angle_deg) from cv2.minAreaRect
    angle_deg is the rotation of the rectangle (OpenCV convention).
    """
    rect = cv2.minAreaRect(poly.astype(np.float32))  # ((cx,cy),(w,h),angle)
    (cx, cy), (w, h), angle = rect
    # On veut angle tel que l'axe "long" = longueur
    if w < h:
        w, h = h, w
        angle = angle + 90.0
    return float(cx), float(cy), float(w), float(h), float(angle)


# -----------------------------
# Gaussian kernel cache (anisotropic + rotated)
# -----------------------------
def _gaussian2d(w, h, sx, sy):
    """
    Génère une gaussienne 2D non-rotée centrée (taille w x h), écarts-types sx, sy.
    """
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    xs = np.arange(w, dtype=np.float32) - cx
    ys = np.arange(h, dtype=np.float32) - cy
    X, Y = np.meshgrid(xs, ys)
    G = np.exp(-0.5 * ((X / max(1e-6, sx))**2 + (Y / max(1e-6, sy))**2))
    G /= (2.0 * np.pi * max(1e-6, sx) * max(1e-6, sy))  # normalisation (optionnelle)
    G = G.astype(np.float32)
    return G

@lru_cache(maxsize=2048)
def get_rotated_gaussian_kernel(kernel_w, kernel_h, sx, sy, angle_deg):
    """
    Retourne un kernel gaussien anisotrope puis **tourné** de angle_deg.
    Les arguments sont arrondis pour permettre le caching (évite de régénérer).
    """
    # discretise pour la cache key
    kernel_w = int(kernel_w); kernel_h = int(kernel_h)
    sx = float(round(sx, 2)); sy = float(round(sy, 2))
    angle_deg = float(round(angle_deg, 1))

    G = _gaussian2d(kernel_w, kernel_h, sx, sy)

    # rotation dans le patch
    M = cv2.getRotationMatrix2D(((kernel_w - 1)/2.0, (kernel_h - 1)/2.0), angle_deg, 1.0)
    R = cv2.warpAffine(G, M, (kernel_w, kernel_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return R


def add_kernel_at(accum, kernel, cx, cy):
    """
    Ajoute 'kernel' (2D float32) à 'accum' (2D float32) centré sur (cx,cy) dans la grille.
    Gère les bords (crop).
    """
    H, W = accum.shape
    kh, kw = kernel.shape
    x0 = int(round(cx - (kw - 1) / 2.0))
    y0 = int(round(cy - (kh - 1) / 2.0))
    x1 = x0 + kw
    y1 = y0 + kh

    # ROI intersection
    xs0 = max(0, x0); ys0 = max(0, y0)
    xs1 = min(W, x1); ys1 = min(H, y1)
    if xs1 <= xs0 or ys1 <= ys0:
        return  # tout dehors

    # crop kernel en même temps
    kx0 = xs0 - x0; ky0 = ys0 - y0
    kx1 = kx0 + (xs1 - xs0); ky1 = ky0 + (ys1 - ys0)

    accum[ys0:ys1, xs0:xs1] += kernel[ky0:ky1, kx0:kx1]


# -----------------------------
# Core pipeline: Gaussian splatting
# -----------------------------
def run_parking_gaussian_splat(
    txt_dir: str,
    image_path: str,
    cell: int = 4,                 # downsample factor (px -> grid)
    conf_min: float = 0.0,
    w_stop: float = 1.0,           # poids positif (STOP)
    w_move: float = 1.0,           # poids négatif (MOVE)
    extend_long: float = 1.8,      # étend la gaussienne au-delà de la longueur OBB
    sigma_long_ratio: float = 0.35,# sigma_long = ratio * (longueur_effective / cell)
    sigma_cross_ratio: float = 0.60,# sigma_cross = ratio * (largeur / cell)
    kclip: float = 3.0,            # taille noyau ~ 2*kclip*sigma + 1
    alpha_overlay: float = 0.6,    # opacité overlay
    thr: float = 0.6,              # seuil (0..1) pour la map seuillée
    out_prefix: str = "parking_splat"
):
    """
    Stop => +Gaussian orientée ; Move => -Gaussian orientée.
    Sorties: 
      - {out_prefix}_prediction_map.png  (colormap JET)
      - {out_prefix}_overlay.png
      - {out_prefix}_prediction_map_thresh.png
      - {out_prefix}_overlay_thresh.png
    Retourne: (P_img_float, overlay_bgr, heatmap_bgr, P_bin_img_float, overlay_bin_bgr)
    """
    # 1) Image & grille
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    H, W = img.shape[:2]
    hG, wG = int(np.ceil(H / cell)), int(np.ceil(W / cell))

    # 2) Accumulateur (score signé)
    S = np.zeros((hG, wG), dtype=np.float32)

    txt_files = sorted(Path(txt_dir).glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files in {txt_dir}")

    for fp in tqdm(txt_files, desc="Gaussian splatting", unit="file"):
        dets = read_txt_file(fp)
        if not dets:
            continue

        for _, poly, conf, state in dets:
            if conf < conf_min: 
                continue

            # OBB -> rect (centre, taille, angle)
            cx, cy, L, Wrect, angle_deg = obb_to_rect(poly)
            # vers la grille
            cgx = cx / cell
            cgy = cy / cell
            Lg  = (L * extend_long) / cell   # **on étend la longueur**
            Wg  = (Wrect) / cell

            # sigmas (en cellules de grille)
            sx = max(1.0, sigma_long_ratio  * Lg)    # le long de la voiture
            sy = max(1.0, sigma_cross_ratio * Wg)    # en travers

            # taille du noyau en grille
            kw = int(2 * int(kclip * sx) + 1)
            kh = int(2 * int(kclip * sy) + 1)

            # kernel orienté
            K = get_rotated_gaussian_kernel(kw, kh, sx, sy, angle_deg)

            # signe / poids
            if state.startswith("stop"):
                amp = +w_stop
            else:
                amp = -w_move

            # ajout au centre de la rect en grille
            add_kernel_at(S, amp * K, cgx, cgy)

    # 3) Normalisation robuste -> [0,1]
    flat = S.flatten()
    lo = np.percentile(flat, 2)
    hi = np.percentile(flat, 98)
    P_grid = (S - lo) / max(1e-6, (hi - lo))
    P_grid = np.clip(P_grid, 0, 1).astype(np.float32)

    # 4) Upsample à la taille de l'image
    P_img = cv2.resize(P_grid, (W, H), interpolation=cv2.INTER_CUBIC)

    # 5) Colorisation + overlay
    P_u8  = (P_img * 255).astype(np.uint8)
    cmap  = cv2.applyColorMap(P_u8, cv2.COLORMAP_JET)
    over  = cv2.addWeighted(img, 1.0, cmap, alpha_overlay, 0)

    cv2.imwrite(f"{out_prefix}_prediction_map.png", cmap)
    cv2.imwrite(f"{out_prefix}_overlay.png", over)

    # 6) Thresholding + sauvegarde
    P_bin = (P_img >= float(thr)).astype(np.float32)
    # jolie binaire en niveaux de gris pour la sauvegarde
    P_bin_u8 = (P_bin * 255).astype(np.uint8)
    # overlay binaire (couleur fixe, ex. vert)
    color_mask = np.zeros_like(img)
    color_mask[:, :] = (0, 255, 0)   # BGR
    mask3 = np.dstack([P_bin_u8]*3) // 255
    over_bin = (img * (1 - mask3) + color_mask * mask3 * 0.6).astype(np.uint8)

    cv2.imwrite(f"{out_prefix}_prediction_map_thresh.png", P_bin_u8)
    cv2.imwrite(f"{out_prefix}_overlay_thresh.png", over_bin)

    print(f"[OK] Saved: {out_prefix}_prediction_map.png")
    print(f"[OK] Saved: {out_prefix}_overlay.png")
    print(f"[OK] Saved: {out_prefix}_prediction_map_thresh.png")
    print(f"[OK] Saved: {out_prefix}_overlay_thresh.png")

    return P_img, over, cmap, P_bin, over_bin


# -------- Exemple d'appel depuis ton IDE --------
if __name__ == "__main__":
    TXT_DIR = r"C:/Users/makss/Git/Galatsi-Semester-Project/Results/TXT12"
    IMAGE   = r"C:/Users/makss/Git/Galatsi-Semester-Project/first_frame_static.png"

    run_parking_gaussian_splat(
        txt_dir=TXT_DIR,
        image_path=IMAGE,
        cell=4,                 # 2-6 recommandé (plus petit = plus précis, plus lent)
        conf_min=0.0,
        w_stop=1.0,
        w_move=4.5,            # pénalise plus la circulation
        extend_long=2.8,       # gaussienne dépassant la voiture dans sa longueur
        sigma_long_ratio=0.35, # adoucir/allonger dans l'axe
        sigma_cross_ratio=0.60,# s'étaler un peu en largeur
        kclip=3.0,             # 3 sigmas ~ 99.7% d'énergie
        alpha_overlay=0.6,
        thr=0.80,              # seuil pour la map seuillée
        out_prefix="Results/parking_detection/gauss_splat/test2/parking_splat"
    )
