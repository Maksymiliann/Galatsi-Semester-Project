from ultralytics import YOLO
import cv2, math
import numpy as np

def tile_coords(W, H, tile=1280, overlap=0.2):
    step = int(tile * (1 - overlap))
    xs = list(range(0, max(W - tile, 0) + 1, step)) or [0]
    ys = list(range(0, max(H - tile, 0) + 1, step)) or [0]
    # assure qu'on couvre le bord droit/bas
    if xs[-1] + tile < W: xs.append(W - tile)
    if ys[-1] + tile < H: ys.append(H - tile)
    for x in xs:
        for y in ys:
            yield x, y, min(tile, W - x), min(tile, H - y)

def angle_to_deg(a):
    # Ultralytics OBB renvoie souvent l’angle en radians (≈[-pi/2, pi/2]).
    # Si ça ressemble à des degrés (>~6.28), on ne touche pas.
    return a * 180 / math.pi if abs(a) <= 3.2 else a

def draw_obb(img, cx, cy, w, h, ang_deg, color=(0, 255, 0), thickness=2):
    # cv2.minAreaRect / boxPoints attend angle en degrés (sens antihoraire, convention OpenCV)
    rect = ((cx, cy), (w, h), ang_deg)
    box = cv2.boxPoints(rect).astype(int)
    cv2.polylines(img, [box], isClosed=True, color=color, thickness=thickness)

def infer_obb_tiled(
    img_path,
    model_path="yolo11n-obb.pt",
    tile=1280,
    overlap=0.2,
    conf_thres=0.25,
    device=None,
    save_path="out_obb_tiled.jpg",
    draw_labels=False
):
    model = YOLO(model_path)
    if device:
        model.to(device)

    img = cv2.imread(img_path)
    assert img is not None, f"Image not found: {img_path}"
    H, W = img.shape[:2]

    # On copie pour dessiner
    canvas = img.copy()

    all_det = []  # (cls, conf, cx, cy, w, h, angle_deg)

    for x0, y0, tw, th in tile_coords(W, H, tile=tile, overlap=overlap):
        tile_img = img[y0:y0+th, x0:x0+tw]

        # Important: désactive les overlays auto, on gère nous-mêmes
        res = model.predict(
            source=tile_img,
            imgsz=tile,       # taille d’entrée (carré) pour la tuile
            conf=conf_thres,
            show=False,
            save=False,
            show_labels=False,
            show_conf=False,
            verbose=False
        )[0]

        obb = getattr(res, "obb", None)
        if obb is None or obb.data is None or len(obb) == 0:
            continue

        # Récupération robuste des sorties OBB
        # Priorité à xywhr si dispo, sinon fallback (rare) vers xywh + angle 0
        if hasattr(obb, "xywhr"):
            xywhr = obb.xywhr.cpu().numpy()
        else:
            # Fallback rudimentaire (au cas où) : angle 0
            xywh = obb.xywh.cpu().numpy()
            ang = np.zeros((xywh.shape[0], 1), dtype=np.float32)
            xywhr = np.concatenate([xywh, ang], axis=1)

        confs = obb.conf.cpu().numpy() if hasattr(obb, "conf") else np.ones((xywhr.shape[0],), dtype=np.float32)
        clss  = obb.cls.cpu().numpy()  if hasattr(obb, "cls")  else np.zeros((xywhr.shape[0],), dtype=np.int32)

        # Décalage tuile -> coords globales
        for (cx, cy, w, h, a), c, k in zip(xywhr, confs, clss):
            if c < conf_thres: 
                continue
            gx, gy = cx + x0, cy + y0
            ang_deg = angle_to_deg(float(a))
            all_det.append((int(k), float(c), float(gx), float(gy), float(w), float(h), ang_deg))

    # Dessin (sans texte)
    for k, c, gx, gy, w, h, ang_deg in all_det:
        draw_obb(canvas, gx, gy, w, h, ang_deg, color=(0,255,0), thickness=2)
        if draw_labels:
            label = f"{int(k)}:{c:.2f}"
            cv2.putText(canvas, label, (int(gx), int(gy)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

    cv2.imwrite(save_path, canvas)
    return save_path, all_det

# Exemple d’utilisation :
# save_path, dets = infer_obb_tiled(
#     img_path="image_3840x2160.jpg",
#     model_path="yolo11l-obb.pt",
#     tile=1280,         # ou 960/1024/1536 selon ta VRAM
#     overlap=0.25,      # 25% de chevauchement
#     conf_thres=0.35,
#     device=None,       # "cuda:0" si tu veux forcer GPU
#     save_path="out_obb_tiled.jpg",
#     draw_labels=False  # True si tu veux afficher (classe:conf)
# )
# print("Saved:", save_path, "Detections:", len(dets))


save_path, dets = infer_obb_tiled(
    img_path="first_frame_static.png",
    model_path="yolo11x-obb.pt",
    tile=640,         # ou 960/1024/1536 selon ta VRAM
    overlap=0.25,      # 25% de chevauchement
    conf_thres=0.35,
    device=None,       # "cuda:0" si tu veux forcer GPU
    save_path="out_obb_tiled.jpg",
    draw_labels=False  # True si tu veux afficher (classe:conf)
)
print("Saved:", save_path, "Detections:", len(dets))
