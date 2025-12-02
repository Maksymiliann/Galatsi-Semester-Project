import cv2
import json
import csv
import numpy as np
from pathlib import Path
import random

# ========= CONFIG =========
INPUT_MASK = r"C:/Users/makss/Git/Galatsi-Semester-Project/Results/parking_detection/zone/mask_closed_cleaned.png"   # binaire 0/255 (ton image)
OUT_PREFIX = "Results/parking_detection/zone/test2/mask"

# Nettoyage / pré-segmentation
DO_OPENING = True
OPEN_K = 3                # taille structurant pour opening (supprime petits points)
FILL_HOLES = True
REMOVE_SMALL = True
MIN_AREA_PX = 5000         # supprime les zones trop petites

# Découpage principal
METHOD = "components"     # "components" (recommandé) ou "watershed"

# Watershed (si METHOD="watershed")
WS_DIST_ERODE = 1         # érosion avant DT (stabilise)
WS_FG_RATIO = 0.45        # seuil relatif sur distance transform pour "sure FG" (0..1)
WS_BG_DILATE = 7          # dilatation pour "sure BG"
# ==========================

def read_mask(path):
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(path)
    return (m > 0).astype(np.uint8)

def bin_to_color(labels):
    """labels: int32 avec 0=bg, >0 = id. Retourne image couleur aléatoire."""
    h, w = labels.shape
    rng = random.Random(123)
    colors = [(0,0,0)]
    n = labels.max()
    for _ in range(n):
        colors.append((rng.randint(60,255), rng.randint(60,255), rng.randint(60,255)))
    out = np.zeros((h,w,3), np.uint8)
    for i in range(1, n+1):
        out[labels == i] = colors[i]
    return out

def fill_holes(mask):
    # flood-fill depuis bord
    h, w = mask.shape
    ff = np.zeros((h+2, w+2), np.uint8)
    inv = (mask == 0).astype(np.uint8)*255
    tmp = inv.copy()
    cv2.floodFill(tmp, ff, (0,0), 128)
    # trous = 0 qui ne sont pas connectés au bord
    holes = (tmp == 0).astype(np.uint8)
    out = mask.copy()
    out[holes > 0] = 1
    return out

def remove_small_components(mask, min_area):
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 1
    return out

def props_from_contour(cnt):
    area = cv2.contourArea(cnt)
    if area <= 0:
        return None
    rect = cv2.minAreaRect(cnt)  # ((cx,cy), (w,h), angle)
    (cx, cy), (w, h), ang = rect
    if w < h:
        long_side, short_side = h, w
        ang = ang + 90.0
    else:
        long_side, short_side = w, h
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area if hull_area > 0 else 1.0
    return dict(
        area=int(round(area)),
        cx=float(cx), cy=float(cy),
        angle_deg=float(ang),   # axe long
        length_px=float(long_side), width_px=float(short_side),
        solidity=float(solidity)
    )

def components_labels(mask):
    num, labels = cv2.connectedComponents(mask, connectivity=8)
    return labels  # 0..num-1

def watershed_labels(mask):
    # mask {0,1}
    m = mask.copy()
    if WS_DIST_ERODE > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*WS_DIST_ERODE+1,)*2)
        m = cv2.erode(m, k)

    dist = cv2.distanceTransform(m, cv2.DIST_L2, 3)
    if dist.max() > 0:
        sure_fg = (dist >= WS_FG_RATIO * dist.max()).astype(np.uint8)
    else:
        sure_fg = m.copy()

    k_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (WS_BG_DILATE, WS_BG_DILATE))
    sure_bg = cv2.dilate(m, k_bg)
    unknown = ((sure_bg > 0) & (sure_fg == 0)).astype(np.uint8)

    # marquage
    num, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1  # 1..num, 0=unknown
    markers[unknown == 1] = 0

    # watershed exige BGR
    bgr = (m*255).astype(np.uint8)
    bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
    cv2.watershed(bgr, markers)  # modifie in-place; -1 = frontière

    # normaliser: frontières -> 0, ids >0
    markers[markers == -1] = 0
    return markers.astype(np.int32)

def export_polys(labels, out_json):
    n = int(labels.max())
    h, w = labels.shape
    items = []
    for i in range(1, n+1):
        bin_i = (labels == i).astype(np.uint8)
        cnts, _ = cv2.findContours(bin_i, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)
        props = props_from_contour(cnt)
        if props is None:
            continue
        poly = cnt.reshape(-1,2).tolist()
        items.append(dict(id=i, polygon=poly, **props))
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(dict(zones=items, image_size=[int(w), int(h)]), f, ensure_ascii=False, indent=2)

def export_stats_csv(labels, out_csv):
    n = int(labels.max())
    rows = []
    for i in range(1, n+1):
        bin_i = (labels == i).astype(np.uint8)
        cnts, _ = cv2.findContours(bin_i, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)
        props = props_from_contour(cnt)
        if props is None:
            continue
        x,y,w,h = cv2.boundingRect(cnt)
        rows.append(dict(
            id=i, area_px=props["area"],
            cx=props["cx"], cy=props["cy"],
            angle_deg=props["angle_deg"],
            length_px=props["length_px"], width_px=props["width_px"],
            solidity=props["solidity"],
            bbox_x=x, bbox_y=y, bbox_w=w, bbox_h=h
        ))
    if rows:
        keys = list(rows[0].keys())
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=keys)
            wr.writeheader(); wr.writerows(rows)

def main():
    mask = read_mask(INPUT_MASK)

    # Nettoyage
    work = mask.copy()
    if DO_OPENING:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (OPEN_K, OPEN_K))
        work = cv2.morphologyEx(work, cv2.MORPH_OPEN, k)
    if FILL_HOLES:
        work = fill_holes(work)
    if REMOVE_SMALL and MIN_AREA_PX > 0:
        work = remove_small_components(work, MIN_AREA_PX)

    # Labels
    if METHOD == "watershed":
        labels = watershed_labels(work)
    else:
        labels = components_labels(work)

    # Sauvegardes
    # id map en png (16 bits si beaucoup de zones) -> ici 8 bits suffisent souvent
    id8 = np.clip(labels, 0, 255).astype(np.uint8)
    cv2.imwrite(f"{OUT_PREFIX}_zones_id.png", id8)

    color = bin_to_color(labels)
    cv2.imwrite(f"{OUT_PREFIX}_zones_color.png", color)

    overlay = cv2.cvtColor((mask*255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    blend = cv2.addWeighted(overlay, 0.6, color, 0.6, 0)
    cv2.imwrite(f"{OUT_PREFIX}_zones_overlay.png", blend)

    export_polys(labels, f"{OUT_PREFIX}_zones_polys.json")
    export_stats_csv(labels, f"{OUT_PREFIX}_zones_stats.csv")

    print("Done.")
    print(f"- {OUT_PREFIX}_zones_id.png")
    print(f"- {OUT_PREFIX}_zones_color.png")
    print(f"- {OUT_PREFIX}_zones_overlay.png")
    print(f"- {OUT_PREFIX}_zones_polys.json")
    print(f"- {OUT_PREFIX}_zones_stats.csv")

if __name__ == "__main__":
    main()
