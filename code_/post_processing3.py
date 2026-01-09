from pathlib import Path
import numpy as np
import cv2

"""
This script post-processes a binary parking mask to fill small gaps and generate smoother “parking zones”.

It loads an input mask (seeds) and optionally defines a candidate band where gap filling is allowed (none / same
as seeds / dilated seeds). Each connected component of the seed mask is then dilated *only along its main axis*:
the component orientation is estimated with cv2.minAreaRect, optionally snapped to SNAP_DEG for stability, and a
line-shaped kernel of length (2*GAP_PX+1) is used to bridge gaps in that direction.

Pixels added by this oriented dilation (but not originally in seeds) are marked as gaps. The final zones mask is
built as (seeds OR gaps), followed by a morphological closing (CLOSE_PX) and an optional final dilation to make
zones more continuous.

Outputs saved in OUT_DIR:
- seeds.png, cand_band.png (if used)
- mask_dilated_oriented.png (oriented dilation result)
- gaps_oriented.png (newly filled pixels)
- zones_mask.png (final zones)
- debug_overlay.png (seeds=white, dilation=green, gaps=red)
"""



# ============================================================
# CONFIG (modifie ici)
# ============================================================

MASK_PATH = r"C:/Users/makss/Git/Galatsi-Semester-Project/Results/occupancy_analysis/per_zones/cleaned_bw.png"     # input mask binaire (0/255 ou non-zero)
OUT_DIR   = r"C:/Users/makss/Git/Galatsi-Semester-Project/Results/parking_detection/post_processing3"      # dossier de sortie

# Dilation orientée
GAP_PX    = 20     # longueur max à combler (pixels)
SNAP_DEG  = 15     # arrondi de l'angle (15 => stable). Mets 0 pour angle exact
THICKNESS = 1      # épaisseur du kernel ligne (1-3)

# Bande candidate (où on autorise le remplissage)
# "none" = pas de contrainte
# "same_as_seeds" = uniquement sur les seeds (souvent trop strict)
# "dilated_seeds" = autorise autour des seeds (souvent le mieux)
CAND_MODE      = "dilated_seeds"
CAND_DILATE_PX = 35   # rayon (pixels) si CAND_MODE="dilated_seeds"

# Post-traitement "zones"
CLOSE_PX       = 9    # fermeture morphologique (kernel carré)
DILATE_FINAL_ITERS = 1  # dilatation finale des zones (0 pour désactiver)

# ============================================================
# UTILS
# ============================================================

def load_binary_mask(path: str) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"Cannot read mask: {path}")
    return (m > 0).astype(np.uint8)

def line_kernel(L: int, angle_deg: float, thickness: int = 1) -> np.ndarray:
    k = np.zeros((L, L), np.uint8)
    c = L // 2
    rad = np.deg2rad(angle_deg)
    dx, dy = int(np.cos(rad) * c), int(np.sin(rad) * c)
    cv2.line(k, (c - dx, c - dy), (c + dx, c + dy), 1, thickness)
    return k

def component_oriented_dilate(
    seeds_bin: np.ndarray,
    gap_px: int,
    cand_band_bin: np.ndarray | None = None,
    snap_deg: int | None = 15,
    thickness: int = 1,
    connectivity: int = 8,
) -> np.ndarray:
    seeds_bin = (seeds_bin > 0).astype(np.uint8)
    L = 2 * gap_px + 1

    num, labels = cv2.connectedComponents(seeds_bin, connectivity=connectivity)
    out = np.zeros_like(seeds_bin)

    for lbl in range(1, num):
        comp = (labels == lbl).astype(np.uint8)

        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)

        (cx, cy), (w, h), ang = cv2.minAreaRect(cnt)  # ang in (-90, 0]
        if w < 1 or h < 1:
            continue

        # orienter selon l'axe long
        if w < h:
            ang = ang + 90.0

        # snap optionnel
        if snap_deg is not None and snap_deg > 0:
            ang = round(ang / snap_deg) * snap_deg

        k_line = line_kernel(L, ang, thickness=thickness)
        dil = cv2.dilate(comp, k_line)
        out = np.maximum(out, dil)

    out = (out > 0).astype(np.uint8)
    if cand_band_bin is not None:
        out = (out & (cand_band_bin > 0).astype(np.uint8)).astype(np.uint8)
    return out

# ============================================================
# MAIN
# ============================================================

def main():
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = load_binary_mask(MASK_PATH)

    # Candidate band
    cand_band = None
    if CAND_MODE == "none":
        cand_band = None
    elif CAND_MODE == "same_as_seeds":
        cand_band = seeds.copy()
    elif CAND_MODE == "dilated_seeds":
        r = max(1, int(CAND_DILATE_PX))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        cand_band = cv2.dilate(seeds, k)
    else:
        raise ValueError(f"Unknown CAND_MODE: {CAND_MODE}")

    snap = None if SNAP_DEG == 0 else int(SNAP_DEG)

    seeds_dil = component_oriented_dilate(
        seeds_bin=seeds,
        gap_px=int(GAP_PX),
        cand_band_bin=cand_band,
        snap_deg=snap,
        thickness=int(THICKNESS),
    )

    # gaps
    if cand_band is None:
        gaps = ((seeds_dil > 0) & (seeds == 0)).astype(np.uint8)
    else:
        gaps = ((seeds_dil > 0) & (cand_band > 0) & (seeds == 0)).astype(np.uint8)

    # zones: close + (optional) dilate
    close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (int(CLOSE_PX), int(CLOSE_PX)))
    zones = cv2.morphologyEx((seeds | gaps).astype(np.uint8), cv2.MORPH_CLOSE, close_k)

    if int(DILATE_FINAL_ITERS) > 0:
        k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        zones = cv2.dilate(zones, k5, iterations=int(DILATE_FINAL_ITERS))

    # Save outputs
    cv2.imwrite(str(out_dir / "seeds.png"), (seeds * 255).astype(np.uint8))
    if cand_band is not None:
        cv2.imwrite(str(out_dir / "cand_band.png"), (cand_band * 255).astype(np.uint8))
    cv2.imwrite(str(out_dir / "mask_dilated_oriented.png"), (seeds_dil * 255).astype(np.uint8))
    cv2.imwrite(str(out_dir / "gaps_oriented.png"), (gaps * 255).astype(np.uint8))
    cv2.imwrite(str(out_dir / "zones_mask.png"), (zones * 255).astype(np.uint8))

    # Debug overlay (visu)
    overlay = np.zeros((seeds.shape[0], seeds.shape[1], 3), dtype=np.uint8)
    overlay[seeds > 0] = (255, 255, 255)   # seeds blanc
    overlay[seeds_dil > 0] = (0, 255, 0)   # dilaté vert
    overlay[gaps > 0] = (0, 0, 255)        # gaps rouge
    cv2.imwrite(str(out_dir / "debug_overlay.png"), overlay)

    print(f"[OK] Done. Outputs in: {out_dir}")

if __name__ == "__main__":
    main()
