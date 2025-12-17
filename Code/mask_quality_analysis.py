import cv2
import numpy as np

# -----------------
# PATHS
# -----------------
GT_PATH   = r"C:/Users/makss/Git/Galatsi-Semester-Project/Results/occupancy_analysis/per_zones/mask_ids_color.png"
MASK_PATH = r"C:/Users/makss/Git/Galatsi-Semester-Project/Results/occupancy_analysis/per_zones/cleaned_bw.png"
#MASK_PATH = r"C:/Users/makss/Git/Galatsi-Semester-Project/Results/parking_detection/dwell_mult_4/test4_thr_0.85/parking_dwell_state_MULTI_REG_zones_mask.png"

# (optionnel) image de fond pour visualiser (satellite / frame)
# Mets None si tu n’en as pas
RGB_PATH  = None  # ex: r"C:/Users/makss/.../satellite.png"

OUT_OVERLAY = r"C:/Users/makss/Git/Galatsi-Semester-Project/Results/occupancy_analysis/per_zones/overlay_fp_fn.png"

# -----------------
# OPTIONS
# -----------------
THRESH = 127
INVERT_GT = False
INVERT_MASK = False

# Si ton GT est un mask "zones ID" (valeurs 0..N), tu veux souvent :
# parking = (valeur > 0)
GT_NONZERO_IS_PARKING = True

# Affichage
ALPHA = 0.55         # intensité overlay
SHOW_TP = True       # TP en vert (sinon uniquement FP rouge / FN bleu)

# -----------------
# LOAD + BINARIZE
# -----------------
def load_to_bool(path, thresh=127, invert=False, nonzero_is_true=False):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    assert img is not None, f"Impossible de lire {path}"

    if nonzero_is_true:
        b = (img > 0).astype(np.uint8) * 255
    else:
        _, b = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)

    if invert:
        b = 255 - b

    return (b > 0)

gt   = load_to_bool(GT_PATH, THRESH, INVERT_GT, nonzero_is_true=GT_NONZERO_IS_PARKING)
mask = load_to_bool(MASK_PATH, THRESH, INVERT_MASK, nonzero_is_true=False)

# resize si besoin
if gt.shape != mask.shape:
    mask = cv2.resize(
        mask.astype(np.uint8) * 255,
        (gt.shape[1], gt.shape[0]),
        interpolation=cv2.INTER_NEAREST
    ) > 0

H, W = gt.shape

# -----------------
# METRICS
# -----------------
tp = np.logical_and(gt, mask).sum()
fp = np.logical_and(~gt, mask).sum()
fn = np.logical_and(gt, ~mask).sum()
union = np.logical_or(gt, mask).sum()

iou  = tp / union if union > 0 else 1.0
dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 1.0
prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
rec  = tp / (tp + fn) if (tp + fn) > 0 else 1.0

# -----------------
# STRUCTURED PRINT
# -----------------
result = {
    "mask": MASK_PATH.split("\\")[-1],
    "IoU": iou,
    "Dice": dice,
    "Precision": prec,
    "Recall": rec,
    "TP": int(tp),
    "FP": int(fp),
    "FN": int(fn),
    "UnionPixels": int(union)
}

print("\n===== IoU RESULTS (parking vs non-parking) =====")
for k, v in result.items():
    if isinstance(v, float):
        print(f"{k:<14}: {v:.4f}")
    else:
        print(f"{k:<14}: {v}")

# -----------------
# VISU FP / FN overlay
# -----------------
tp_map = gt & mask
fp_map = (~gt) & mask
fn_map = gt & (~mask)

# base (fond)
if RGB_PATH is not None:
    base = cv2.imread(RGB_PATH, cv2.IMREAD_COLOR)
    assert base is not None, f"Impossible de lire {RGB_PATH}"
    if base.shape[:2] != (H, W):
        base = cv2.resize(base, (W, H), interpolation=cv2.INTER_AREA)
else:
    # fond gris basé sur GT
    base = cv2.cvtColor((gt.astype(np.uint8) * 255), cv2.COLOR_GRAY2BGR)

overlay = np.zeros_like(base, dtype=np.uint8)
overlay[fp_map] = (0, 0, 255)   # ROUGE (BGR)
overlay[fn_map] = (255, 0, 0)   # BLEU  (BGR)
if SHOW_TP:
    overlay[tp_map] = (0, 255, 0)  # VERT

mask_any = fp_map | fn_map | (tp_map if SHOW_TP else False)

out = base.copy()
out[mask_any] = cv2.addWeighted(base[mask_any], 1 - ALPHA, overlay[mask_any], ALPHA, 0)

cv2.imwrite(OUT_OVERLAY, out)
print(f"\n✅ Overlay sauvegardé: {OUT_OVERLAY}")
print("Légende: FP=rouge (sur-détection), FN=bleu (raté), TP=vert (correct)")
