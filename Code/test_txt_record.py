import cv2
import numpy as np
from pathlib import Path

# ------------------------
# CONFIGURATION À MODIFIER
# ------------------------
TXT_PATH = Path("Results/TXT12/0001.txt")                       # ton fichier txt
IMG_PATH = Path("first_frame_static.png") # ton image de base
OUT_PATH = Path("overlay_from_txt.png")           # image de sortie
THICKNESS = 2                                     # épaisseur des contours
SHOW_IDS = True                                   # True = afficher les IDs
# ------------------------


def parse_detection_line(line: str):
    """Parse une ligne du fichier txt."""
    parts = [p.strip() for p in line.strip().split(";")]
    if len(parts) < 12:
        return None
    try:
        veh_id = int(float(parts[0]))
        xs = [float(parts[i]) for i in [1, 3, 5, 7]]
        ys = [float(parts[i]) for i in [2, 4, 6, 8]]
        det_class = int(float(parts[9]))
        conf_score = float(parts[10])
        state = parts[11]
        pts = np.array(list(zip(xs, ys)), dtype=np.float32)
        return dict(veh_id=veh_id, points=pts,
                    det_class=det_class, conf_score=conf_score, state=state)
    except Exception:
        return None


def load_detections(txt_path: Path):
    """Lit toutes les détections du fichier txt."""
    detections = []
    with txt_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            if "veh_bb_x1" in line or "vehicle_id" in line:  # ignorer header
                continue
            det = parse_detection_line(line)
            if det is not None:
                detections.append(det)
    return detections


def draw_detections(image, detections, color=(0, 255, 0), thickness=2, show_ids=True):
    """Dessine les polygones sur l'image."""
    out = image.copy()
    for det in detections:
        pts = det["points"].astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(out, [pts], isClosed=True, color=color,
                      thickness=thickness, lineType=cv2.LINE_AA)
        if show_ids:
            p0 = tuple(det["points"][0].astype(int))
            cv2.circle(out, p0, 2, color, -1, lineType=cv2.LINE_AA)
            cv2.putText(out, str(det["veh_id"]), (p0[0] + 3, p0[1] - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return out


if __name__ == "__main__":
    # Charger image
    img = cv2.imread(str(IMG_PATH))
    if img is None:
        raise FileNotFoundError(f"Image introuvable: {IMG_PATH}")

    # Charger txt
    detections = load_detections(TXT_PATH)
    print(f"{len(detections)} détection(s) trouvée(s)")

    # Dessiner
    result = draw_detections(img, detections, thickness=THICKNESS, show_ids=SHOW_IDS)

    # Sauvegarder
    cv2.imwrite(str(OUT_PATH), result)
    print(f"Image sauvegardée -> {OUT_PATH}")

    # Facultatif : afficher l’image
    cv2.imshow("Overlay", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
