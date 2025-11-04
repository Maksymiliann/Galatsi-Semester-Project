import cv2
import numpy as np
from pathlib import Path

# ===============================================================
#                RANSAC LINE DETECTION FUNCTIONS
# ===============================================================

def ransac_lines_from_mask(mask, max_iter=1000, dist_thresh=10, min_inliers=300, max_lines=50, sample_stride=4):
    """
    Détection de droites robustes par RANSAC sur un mask binaire {0,255}.
    Retourne une liste de lignes sous forme de dictionnaires:
    {'a','b','c','pts','endpoints':((x1,y1),(x2,y2))} où ax+by+c=0.
    """
    ys, xs = np.where(mask > 0)
    if len(xs) < 2:
        return []
    if sample_stride > 1:
        xs, ys = xs[::sample_stride], ys[::sample_stride]
    pts = np.stack([xs, ys], axis=1).astype(np.float32)

    lines = []
    remaining = pts.copy()
    rng = np.random.default_rng(42)

    for _ in range(max_lines):
        if len(remaining) < min_inliers:
            break

        best_inliers_idx = None
        best_count = 0

        # === RANSAC ===
        for _ in range(max_iter):
            if len(remaining) < 2: break
            i1, i2 = rng.choice(len(remaining), size=2, replace=False)
            p1, p2 = remaining[i1], remaining[i2]
            if np.allclose(p1, p2): 
                continue

            dx, dy = (p2 - p1)
            a, b = dy, -dx
            norm = np.hypot(a, b)
            if norm < 1e-6:
                continue
            a, b = a / norm, b / norm
            c = -(a * p1[0] + b * p1[1])

            dists = np.abs(a * remaining[:, 0] + b * remaining[:, 1] + c)
            inliers_idx = np.where(dists < dist_thresh)[0]
            count = inliers_idx.size

            if count > best_count:
                best_count = count
                best_inliers_idx = inliers_idx

        if best_inliers_idx is None or best_count < min_inliers:
            break

        inliers = remaining[best_inliers_idx]
        mean = inliers.mean(axis=0)
        Xc = inliers - mean
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        normal = Vt[1, :]
        a, b = normal / (np.hypot(normal[0], normal[1]) + 1e-12)
        c = -(a * mean[0] + b * mean[1])

        direction = np.array([-b, a])
        t = (inliers - mean) @ direction
        tmin, tmax = t.min(), t.max()
        p1 = (mean + tmin * direction).astype(int)
        p2 = (mean + tmax * direction).astype(int)

        lines.append({
            'a': float(a), 'b': float(b), 'c': float(c),
            'pts': inliers,
            'endpoints': (tuple(p1), tuple(p2))
        })

        mask_keep = np.ones(len(remaining), dtype=bool)
        mask_keep[best_inliers_idx] = False
        remaining = remaining[mask_keep]

    return lines


def draw_lines(img_gray, lines, thickness=2, color=(0, 0, 255)):
    out = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    for L in lines:
        (x1, y1), (x2, y2) = L['endpoints']
        cv2.line(out, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
    return out

# ===============================================================
#                          MAIN
# ===============================================================

if __name__ == "__main__":
    # === Chemin de ton image ===
    img_path = r"C:/Users/makss/Git/Galatsi-Semester-Project/Results/parking_detection/post_proc/test1/parking_dwell_state_MULTI_REG_parking_location_mask.png"
    out_path = r"C:/Users/makss/Git/Galatsi-Semester-Project/Results/parking_detection/post_proc/test1/_ransac_lines4.png"

    # === Lecture de l'image ===
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Impossible de lire {img_path}")

    # === Nettoyage / pré-traitement ===
    _, mask = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # === Détection de lignes via RANSAC ===
    print("Détection des lignes RANSAC...")
    lines = ransac_lines_from_mask(mask, dist_thresh=20, min_inliers=300, max_lines=10, sample_stride=2)
    print(f"{len(lines)} lignes détectées")

    # === Affichage et sauvegarde ===
    overlay = draw_lines(img, lines, thickness=2)
    cv2.imwrite(str(out_path), overlay)
    print(f"Résultat sauvegardé : {out_path}")

