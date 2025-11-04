import cv2
import numpy as np
from pathlib import Path

# =========================
# Config utilisateur
# =========================
# Mets ici le chemin de ton zone_mask binaire (0/255) :
INPUT_PATH = r"C:/Users/makss/Git/Galatsi-Semester-Project/Results/parking_detection/post_processing2/parking_dwell_state_MULTI_REG_zones_mask.png"
OUT_PREFIX = "Results/parking_detection/post_processing2/test4/mask"

# --- Stratégie 1 : fermeture multi-orientations
DO_MULTI_ANGLE_CLOSING = True
ANGLES_DEG = [0, 15, 30, 45, 60, 75, 90]   # directions testées
LINE_LEN_PX = 31                            # longueur du kernel (contrôle le gap max comblé)
LINE_THICK_PX = 1                           # épaisseur du kernel
CLOSING_REPEATS = 1                         # refaire la fermeture plusieurs fois si besoin

# --- Stratégie 2 : ponts par Hough (détection + redessin)
DO_HOUGH_BRIDGING = False
CANNY1, CANNY2 = 50, 150
HOUGH_MIN_LEN = 20       # longueur min segment détecté
HOUGH_MAX_GAP = 15       # gap max interne que Hough comble
HOUGH_THETA_TOL = 10     # tolérance d’angle (degrés) pour fusionner 2 segments
MERGE_DIST_PX = 20       # distance max entre extrémités pour les relier
REDRAW_THICKNESS = 3     # épaisseur des lignes redraw

# =========================
# Utils
# =========================
def read_binary(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    # normaliser en {0,1}
    return (img > 0).astype(np.uint8)

def save_bin(prefix, suffix, bin_img):
    cv2.imwrite(f"{prefix}{suffix}.png", (bin_img * 255).astype(np.uint8))

def line_kernel(L, angle_deg, thickness=1):
    """Kernel ligne centré, de taille LxL, orienté à angle_deg (0 = horizontal vers la droite)."""
    k = np.zeros((L, L), np.uint8)
    c = L // 2
    rad = np.deg2rad(angle_deg)
    dx, dy = int(np.cos(rad) * c), int(np.sin(rad) * c)
    cv2.line(k, (c - dx, c - dy), (c + dx, c + dy), 1, thickness)
    return k

def multi_angle_closing(mask01, angles, L, thickness=1, repeats=1):
    """Fermeture morphologique avec kernels orientés; on prend l'union."""
    base = mask01.copy()
    for _ in range(repeats):
        out = np.zeros_like(base)
        for a in angles:
            k = line_kernel(L, a, thickness=thickness)
            closed = cv2.morphologyEx(base, cv2.MORPH_CLOSE, k)
            out = np.maximum(out, closed)
        base = out
    return base

def seg_angle_deg(x1,y1,x2,y2):
    ang = np.degrees(np.arctan2((y2 - y1), (x2 - x1)))
    ang = (ang + 180.0) % 180.0  # modulo 180 (colinéarité sans orientation)
    return ang

def near_and_collinear(s1, s2, theta_tol=10, dist_thresh=20):
    # s = (x1,y1,x2,y2)
    a1 = seg_angle_deg(*s1)
    a2 = seg_angle_deg(*s2)
    if min(abs(a1 - a2), 180 - abs(a1 - a2)) > theta_tol:
        return False
    # distances entre extrémités
    p = np.array([[s1[0], s1[1]], [s1[2], s1[3]]], dtype=np.float32)
    q = np.array([[s2[0], s2[1]], [s2[2], s2[3]]], dtype=np.float32)
    dists = np.linalg.norm(p[:, None, :] - q[None, :, :], axis=2)
    return dists.min() <= dist_thresh

def merge_segments(segments, theta_tol=10, dist_thresh=20, iters=2):
    """Fusionne grossièrement des segments quasi colinéaires et proches, en étendant leurs extrémités."""
    segs = [list(map(int, s)) for s in segments]
    for _ in range(iters):
        used = [False]*len(segs)
        merged = []
        for i in range(len(segs)):
            if used[i]: 
                continue
            x1,y1,x2,y2 = segs[i]
            ax = np.array([x1,y1,x2,y2], dtype=float)
            changed = True
            while changed:
                changed = False
                for j in range(i+1, len(segs)):
                    if used[j]: 
                        continue
                    if near_and_collinear(ax, segs[j], theta_tol, dist_thresh):
                        # étendre l'enveloppe projettée sur la droite commune
                        candidates = [
                            (ax[0],ax[1]), (ax[2],ax[3]),
                            (segs[j][0],segs[j][1]), (segs[j][2],segs[j][3])
                        ]
                        # garder les 2 pts les plus éloignés
                        pts = np.array(candidates, dtype=float)
                        D = np.linalg.norm(pts[:,None,:] - pts[None,:,:], axis=2)
                        i_max = np.unravel_index(np.argmax(D), D.shape)
                        p1, p2 = pts[i_max[0]], pts[i_max[1]]
                        ax = np.array([p1[0],p1[1],p2[0],p2[1]])
                        used[j] = True
                        changed = True
            used[i] = True
            merged.append(tuple(map(int, ax)))
        segs = merged
    return segs

def hough_bridge(mask01, canny1, canny2, min_len, max_gap, theta_tol, merge_dist, redraw_thick):
    # bords + Hough
    edges = cv2.Canny((mask01*255).astype(np.uint8), canny1, canny2)
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=40,
                            minLineLength=min_len, maxLineGap=max_gap)
    if lines is None:
        return np.zeros_like(mask01), []
    segs = [tuple(l[0]) for l in lines]  # (x1,y1,x2,y2)
    segs_merged = merge_segments(segs, theta_tol=theta_tol, dist_thresh=merge_dist, iters=2)

    # redessiner sur une image vide
    canv = np.zeros_like(mask01)
    for (x1,y1,x2,y2) in segs_merged:
        cv2.line(canv, (x1,y1), (x2,y2), 1, redraw_thick)
    return canv, segs_merged

def make_overlay(gray01, added01):
    h,w = gray01.shape
    bgr = cv2.cvtColor((gray01*255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    # verts = ce qui a été ajouté
    bgr[added01>0] = (0,255,0)
    return bgr

# =========================
# Main
# =========================
def main():
    mask01 = read_binary(INPUT_PATH)

    results = []
    if DO_MULTI_ANGLE_CLOSING:
        closed = multi_angle_closing(mask01, ANGLES_DEG, LINE_LEN_PX, LINE_THICK_PX, CLOSING_REPEATS)
        save_bin(OUT_PREFIX, "_closed", closed)
        results.append(("closed", closed))
    else:
        closed = np.zeros_like(mask01)

    if DO_HOUGH_BRIDGING:
        hough_img, segs = hough_bridge(mask01, CANNY1, CANNY2, HOUGH_MIN_LEN, HOUGH_MAX_GAP,
                                       HOUGH_THETA_TOL, MERGE_DIST_PX, REDRAW_THICKNESS)
        save_bin(OUT_PREFIX, "_hough_bridged", hough_img)
        results.append(("hough", hough_img))
    else:
        hough_img = np.zeros_like(mask01)

    # combinaison OR logique pour maximiser la continuité
    combined = np.maximum(np.maximum(mask01, closed), hough_img)
    save_bin(OUT_PREFIX, "_combined", combined)

    # overlay debug : en vert = ajouts par rapport à l’original
    added = (combined > 0) & (mask01 == 0)
    overlay = make_overlay(mask01, added.astype(np.uint8))
    cv2.imwrite(f"{OUT_PREFIX}_overlay_postproc.png", overlay)

    print("Done.")
    print("Outputs:")
    for name, _img in results:
        print(f" - {OUT_PREFIX}_{name}.png")
    print(f" - {OUT_PREFIX}_combined.png")
    print(f" - {OUT_PREFIX}_overlay_postproc.png")

if __name__ == "__main__":
    main()
