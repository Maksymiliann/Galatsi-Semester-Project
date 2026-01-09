from ultralytics import YOLO
import cv2, math, numpy as np

"""
Runs tiled YOLO-OBB inference on a large image and merges detections across tiles.

The image is split into overlapping square tiles, YOLO OBB detections are run on each tile, and detections are
mapped back to global coordinates. To avoid double-counting objects seen in overlapping tiles, an optional
global class-wise NMS is applied using the AABB envelope of each oriented box (approx IoU for OBB).

Finally, the kept OBBs are drawn on the original image, the annotated image is saved, and the script reports the
total number of objects and counts per class (optionally mapped to class names).
"""


def tile_coords(W, H, tile=1280, overlap=0.2):
    step = int(tile * (1 - overlap))
    xs = list(range(0, max(W - tile, 0) + 1, step)) or [0]
    ys = list(range(0, max(H - tile, 0) + 1, step)) or [0]
    if xs[-1] + tile < W: xs.append(W - tile)
    if ys[-1] + tile < H: ys.append(H - tile)
    for x in xs:
        for y in ys:
            yield x, y, min(tile, W - x), min(tile, H - y)

def angle_to_deg(a):
    return a * 180 / math.pi if abs(a) <= 3.2 else a

def draw_obb(img, cx, cy, w, h, ang_deg, color=(0, 255, 0), thickness=2):
    rect = ((cx, cy), (w, h), ang_deg)
    box = cv2.boxPoints(rect).astype(int)
    cv2.polylines(img, [box], isClosed=True, color=color, thickness=thickness)

def obb_to_aabb(cx, cy, w, h, ang_deg):
    rect = ((cx, cy), (w, h), ang_deg)
    box = cv2.boxPoints(rect)  # float
    x1, y1 = box[:,0].min(), box[:,1].min()
    x2, y2 = box[:,0].max(), box[:,1].max()
    return float(x1), float(y1), float(x2), float(y2)

def iou_aabb(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter <= 0: return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / max(1e-9, area_a + area_b - inter)

def nms_global_aabb(dets, iou_thr=0.5):
    """
    dets: list of (cls, conf, cx, cy, w, h, ang_deg)
    Returns indices to keep after global AABB NMS (approx for OBB).
    """
    if not dets:
        return []
    boxes = [obb_to_aabb(d[2], d[3], d[4], d[5], d[6]) for d in dets]
    order = np.argsort([-d[1] for d in dets])  # by confidence desc
    keep = []
    for i in order:
        if all(iou_aabb(boxes[i], boxes[k]) <= iou_thr for k in keep):
            keep.append(i)
    return keep

def infer_obb_tiled(
    img_path,
    model_path="yolo11n-obb.pt",
    tile=1280,
    overlap=0.2,
    conf_thres=0.25,
    device=None,
    save_path="out_obb_tiled.jpg",
    draw_labels=False,
    dedupe=True,         # 👈 enable global NMS to avoid double-counting
    iou_thr=0.5,
    classes=None
):
    model = YOLO(model_path)
    if device:
        model.to(device)

    img = cv2.imread(img_path)
    assert img is not None, f"Image not found: {img_path}"
    H, W = img.shape[:2]
    canvas = img.copy()
    all_det = []  # (cls, conf, cx, cy, w, h, ang_deg)

    for x0, y0, tw, th in tile_coords(W, H, tile=tile, overlap=overlap):
        tile_img = img[y0:y0+th, x0:x0+tw]
        res = model.predict(
            source=tile_img,
            imgsz=tile,
            conf=conf_thres,
            classes=classes,
            show=False, save=False,
            show_labels=False, show_conf=False,
            verbose=False
        )[0]

        obb = getattr(res, "obb", None)
        if obb is None or getattr(obb, "data", None) is None or len(obb) == 0:
            continue

        # Prefer xywhr; fallback to xywh + zero angle
        if hasattr(obb, "xywhr"):
            xywhr = obb.xywhr.cpu().numpy()
        else:
            xywh = obb.xywh.cpu().numpy()
            ang = np.zeros((xywh.shape[0], 1), dtype=np.float32)
            xywhr = np.concatenate([xywh, ang], axis=1)

        confs = obb.conf.cpu().numpy() if hasattr(obb, "conf") else np.ones((xywhr.shape[0],), dtype=np.float32)
        clss  = obb.cls.cpu().numpy()  if hasattr(obb, "cls")  else np.zeros((xywhr.shape[0],), dtype=np.int32)

        for (cx, cy, w, h, a), c, k in zip(xywhr, confs, clss):
            if c < conf_thres:
                continue
            gx, gy = cx + x0, cy + y0
            ang_deg = angle_to_deg(float(a))
            all_det.append((int(k), float(c), float(gx), float(gy), float(w), float(h), ang_deg))

    # --- de-duplication across tiles (optional) ---
    if dedupe:
        keep_idx = nms_global_aabb(all_det, iou_thr=iou_thr)
        kept = [all_det[i] for i in keep_idx]
    else:
        kept = all_det

    # --- counts ---
    num_objects = len(kept)
    counts_by_class = {}
    for k, *_ in kept:
        counts_by_class[k] = counts_by_class.get(k, 0) + 1

    # --- draw (no text by default) ---
    for k, c, gx, gy, w, h, ang_deg in kept:
        draw_obb(canvas, gx, gy, w, h, ang_deg, color=(0,255,0), thickness=2)
        if draw_labels:
            cv2.putText(canvas, f"{k}:{c:.2f}", (int(gx), int(gy)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

    cv2.imwrite(save_path, canvas)

    # map class IDs to names if available
    try:
        names = getattr(model, "names", None) or getattr(model.model, "names", None) or {}
        counts_by_name = { (names.get(k, str(k))): v for k, v in counts_by_class.items() }
    except Exception:
        counts_by_name = { str(k): v for k, v in counts_by_class.items() }

    return save_path, kept, num_objects, counts_by_name


save_path, dets, num_objects, counts = infer_obb_tiled(
    img_path="first_frame_static.png",
    model_path="yolo11l-obb.pt",
    tile=1280,
    overlap=0.25,
    conf_thres=0.25,
    dedupe=True,      # turn off if you prefer raw counts (may double-count)
    iou_thr=0.5,
    classes=[9, 10],
    save_path="out_obb_tiled.png"
)

print("Total objects:", num_objects)
print("By class:", counts)
print("Saved:", save_path)
