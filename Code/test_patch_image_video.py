import math
import cv2
import numpy as np
from ultralytics import YOLO

# ----------------------------
# Tiling utilities
# ----------------------------
def choose_grid_from_num_tiles(num_tiles: int):
    """Pick a near-square (cols, rows) grid for a desired number of tiles."""
    cols = int(math.sqrt(num_tiles))
    rows = math.ceil(num_tiles / cols)
    while cols * rows < num_tiles:
        cols += 1
    return cols, rows  # tiles_x, tiles_y

def make_tiles(frame_h, frame_w, tiles_x, tiles_y, overlap_ratio=0.2):
    """
    Compute tile rectangles with overlap.
    Returns list of (x0, y0, x1, y1).
    """
    base_w = frame_w / tiles_x
    base_h = frame_h / tiles_y
    ov_w = int(base_w * overlap_ratio)
    ov_h = int(base_h * overlap_ratio)

    tiles = []
    for j in range(tiles_y):
        for i in range(tiles_x):
            x0 = int(i * base_w)
            y0 = int(j * base_h)
            x1 = int(min(frame_w, (i + 1) * base_w))
            y1 = int(min(frame_h, (j + 1) * base_h))
            # inflate with overlap (clamp to image)
            x0 = max(0, x0 - ov_w if i > 0 else x0)
            y0 = max(0, y0 - ov_h if j > 0 else y0)
            x1 = min(frame_w, x1 + (ov_w if i < tiles_x - 1 else 0))
            y1 = min(frame_h, y1 + (ov_h if j < tiles_y - 1 else 0))
            tiles.append((x0, y0, x1, y1))
    return tiles

# ----------------------------
# Detection extraction & drawing
# ----------------------------
def extract_detections(result, x_offset=0, y_offset=0, task="auto"):
    """
    Extract detections from a single Ultralytics result.
    Supports normal boxes and OBB. Returns a list of dicts:
       { 'cls': int, 'conf': float,
         'xyxy': np.array([x1,y1,x2,y2]),   # AABB for NMS/merge
         'poly': np.array((4,2)) or None }  # polygon if OBB
    """
    dets = []

    # infer task if needed
    if task == "auto":
        task = "obb" if hasattr(result, "obb") and result.obb is not None else "detect"

    if task == "obb" and hasattr(result, "obb") and result.obb is not None:
        obb = result.obb
        confs = obb.conf.cpu().numpy() if hasattr(obb, "conf") else np.array([])
        clss = obb.cls.cpu().numpy().astype(int) if hasattr(obb, "cls") else np.array([])

        poly = None
        if hasattr(obb, "xyxyxyxy"):
            poly = obb.xyxyxyxy.cpu().numpy().reshape(-1, 4, 2)  # (N,8)->(N,4,2)
        elif hasattr(obb, "xyxy"):
            aabb = obb.xyxy.cpu().numpy()
            for i in range(aabb.shape[0]):
                x1, y1, x2, y2 = aabb[i]
                dets.append({
                    "cls": int(clss[i]) if len(clss) > i else -1,
                    "conf": float(confs[i]) if len(confs) > i else 0.0,
                    "xyxy": np.array([x1 + x_offset, y1 + y_offset, x2 + x_offset, y2 + y_offset], dtype=float),
                    "poly": None
                })
            return dets

        if poly is not None:
            for i in range(poly.shape[0]):
                p = poly[i]
                p = p.copy()
                p[:, 0] += x_offset
                p[:, 1] += y_offset
                x1, y1 = p[:, 0].min(), p[:, 1].min()
                x2, y2 = p[:, 0].max(), p[:, 1].max()
                dets.append({
                    "cls": int(clss[i]) if len(clss) > i else -1,
                    "conf": float(confs[i]) if len(confs) > i else 0.0,
                    "xyxy": np.array([x1, y1, x2, y2], dtype=float),
                    "poly": p
                })
        return dets

    # fallback: normal detect (AABB)
    if hasattr(result, "boxes") and result.boxes is not None:
        boxes = result.boxes
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy() if hasattr(boxes, "conf") else np.ones(xyxy.shape[0], dtype=float)
        clss = boxes.cls.cpu().numpy().astype(int) if hasattr(boxes, "cls") else -np.ones(xyxy.shape[0], dtype=int)
        for i in range(xyxy.shape[0]):
            x1, y1, x2, y2 = xyxy[i]
            dets.append({
                "cls": int(clss[i]),
                "conf": float(confs[i]),
                "xyxy": np.array([x1 + x_offset, y1 + y_offset, x2 + x_offset, y2 + y_offset], dtype=float),
                "poly": None
            })
    return dets

def draw_detection(frame, det, class_names=None, color=(0,255,0),
                   show_text=True, show_aabb=False, show_poly=True):
    x1, y1, x2, y2 = det["xyxy"].astype(int)

    # Only draw OBB polygon if present
    if det.get("poly") is not None:
        if show_poly:
            poly = det["poly"].astype(int)
            cv2.polylines(frame, [poly], isClosed=True, color=color, thickness=2)
        if show_aabb:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)  # optional
    else:
        # No OBB → draw AABB
        if show_aabb or True:   # keep default behavior for non-OBB models
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    if show_text:
        label = f"{det['cls']}"
        if class_names and 0 <= det["cls"] < len(class_names):
            label = class_names[det["cls"]]
        label = f"{label} {det['conf']:.2f}"
        cv2.putText(frame, label, (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


# ----------------------------
# Simple class-wise NMS (AABB IoU)
# ----------------------------
def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter + 1e-9
    return inter / union

def nms_merge(dets, iou_thr=0.5):
    """
    Very simple class-wise NMS using AABB IoU.
    Keeps highest-conf box per cluster; for OBB we use AABB envelope.
    """
    out = []
    dets = sorted(dets, key=lambda d: d["conf"], reverse=True)
    used = [False] * len(dets)
    for i, di in enumerate(dets):
        if used[i]:
            continue
        out.append(di)
        for j in range(i + 1, len(dets)):
            if used[j]:
                continue
            dj = dets[j]
            if di["cls"] != dj["cls"]:
                continue
            if iou_xyxy(di["xyxy"], dj["xyxy"]) > iou_thr:
                used[j] = True
    return out

# ----------------------------
# Common tiled inference on a single frame (image array)
# ----------------------------
def process_frame_tiled(
    frame,
    model,
    imgsz=1280,
    conf=0.25,
    device=None,
    tiles=None,           # (tiles_x, tiles_y)
    num_tiles=None,       # or desired number -> near-square grid
    overlap_ratio=0.2,
    nms_iou=0.5,
    obb=False,
    classes=None         # e.g. [9, 10]
):
    H, W = frame.shape[:2]
    # decide grid
    if tiles is None:
        if num_tiles is None:
            tiles_x, tiles_y = (2, 2)
        else:
            tiles_x, tiles_y = choose_grid_from_num_tiles(num_tiles)
    else:
        tiles_x, tiles_y = tiles

    tiles_rc = make_tiles(H, W, tiles_x, tiles_y, overlap_ratio=overlap_ratio)

    all_dets = []
    for (x0, y0, x1, y1) in tiles_rc:
        tile = frame[y0:y1, x0:x1]
        results = model.predict(
            source=tile,
            imgsz=imgsz,
            conf=conf,
            device=device,
            verbose=False,
            classes=classes  # filter at inference if provided
        )
        for res in results:
            dets = extract_detections(
                res, x_offset=x0, y_offset=y0,
                task="obb" if obb else "detect"
            )
            all_dets.extend(dets)

    merged = nms_merge(all_dets, iou_thr=nms_iou)
    return merged

def count_by_class(dets):
    counts = {}
    for d in dets:
        counts[d["cls"]] = counts.get(d["cls"], 0) + 1
    return counts

# ----------------------------
# Image tiling inference
# ----------------------------
def patch_infer_image(
    image_path: str,
    model_path: str,
    output_path: str = None,
    imgsz: int = 1280,
    conf: float = 0.25,
    device: str = None,
    tiles: tuple = None,
    num_tiles: int = None,
    overlap_ratio: float = 0.2,
    nms_iou: float = 0.5,
    obb: bool = False,
    classes=None,               # e.g. [9, 10]
    show_text: bool = True
):
    """
    Runs patch inference on a single image and optionally saves an annotated image.
    Returns (merged_dets, total_count, counts_by_name).
    """
    model = YOLO(model_path)

    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    merged = process_frame_tiled(
        img, model, imgsz=imgsz, conf=conf, device=device,
        tiles=tiles, num_tiles=num_tiles, overlap_ratio=overlap_ratio,
        nms_iou=nms_iou, obb=obb, classes=classes
    )

    # draw
    out_img = img.copy()
    class_names = model.names if hasattr(model, "names") else None
    for det in merged: 
        draw_detection(out_img, det, class_names, show_text=False, show_aabb=False, show_poly=True)

    if output_path:
        cv2.imwrite(output_path, out_img)

    # counts
    counts_raw = count_by_class(merged)
    if class_names:
        counts_by_name = {
            (class_names[k] if 0 <= k < len(class_names) else str(k)): v
            for k, v in counts_raw.items()
        }
    else:
        counts_by_name = {str(k): v for k, v in counts_raw.items()}

    return merged, sum(counts_raw.values()), counts_by_name

# ----------------------------
# Main video tiling inference (unchanged API + extras)
# ----------------------------
def patch_infer_video(
    video_path: str,
    model_path: str,
    output_path: str = None,
    imgsz: int = 1280,
    conf: float = 0.25,
    device: str = None,  # e.g. "0" for first CUDA, or "cpu"
    tiles: tuple = None,  # (tiles_x, tiles_y)
    num_tiles: int = None,  # choose grid from a target number
    overlap_ratio: float = 0.2,
    nms_iou: float = 0.5,
    stride: int = 1,  # process every Nth frame
    max_frames: int = None,  # limit for quick tests
    obb: bool = False,       # OBB model (e.g., yolo11x-obb)
    classes=None,            # e.g. [9, 10]
    show_text: bool = True
):
    """
    Runs patch inference on a video and optionally saves an annotated video.
    Prints per-frame counts.
    """
    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    ret, frame0 = cap.read()
    if not ret:
        raise RuntimeError("Could not read first frame.")
    H, W = frame0.shape[:2]

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    frame_idx = 0
    processed = 0
    class_names = model.names if hasattr(model, "names") else None

    while True:
        frame = frame0 if frame_idx == 0 else cap.read()[1]
        if frame is None:
            break
        if frame_idx % stride != 0:
            frame_idx += 1
            continue

        merged = process_frame_tiled(
            frame, model, imgsz=imgsz, conf=conf, device=device,
            tiles=tiles, num_tiles=num_tiles, overlap_ratio=overlap_ratio,
            nms_iou=nms_iou, obb=obb, classes=classes
        )

        # draw + optional write
        out_frame = frame.copy()
        for det in merged: 
            draw_detection(out_frame, det, class_names, show_text=False, show_aabb=False, show_poly=True)
        if writer is not None:
            writer.write(out_frame)

        # counts per frame (print)
        counts_raw = count_by_class(merged)
        total = sum(counts_raw.values())
        if class_names:
            counts_by_name = {
                (class_names[k] if 0 <= k < len(class_names) else str(k)): v
                for k, v in counts_raw.items()
            }
        else:
            counts_by_name = {str(k): v for k, v in counts_raw.items()}
        print(f"Frame {frame_idx}: total={total} | {counts_by_name}")

        processed += 1
        frame_idx += 1
        if max_frames is not None and processed >= max_frames:
            break

    cap.release()
    if writer is not None:
        writer.release()
    print(f"Done. Processed {processed} frames{' and wrote ' + output_path if output_path else ''}.")

# ----------------------------
# Examples
# ----------------------------
if __name__ == "__main__":
    # IMAGE example (OBB, only small/large vehicles, no text)
    # merged, total, counts = patch_infer_image(
    #     image_path="first_frame_static.png",
    #     model_path="yolo11n-obb.pt",
    #     output_path="Results/Images/static/test_patch_image_video/imgsz = 2560/obbn/imgsz2560tiles3-2.png",
    #     imgsz=2560,
    #     conf=0.25,
    #     tiles=(3, 2),          # or num_tiles=6
    #     overlap_ratio=0.35,
    #     nms_iou=0.5,
    #     obb=True,
    #     classes=[9, 10],       # e.g. small & large vehicle
    #     show_text=False
    # )
    # print("Image total:", total, "| by class:", counts)

    # VIDEO example 
    patch_infer_video(
        video_path="Dataset/Galatsi_Data_Semester_Project_archive/DJI_0808.MOV",
        model_path="yolo11x-obb.pt",
        output_path="annotated.mp4",
        imgsz=1280,
        conf=0.25,
        tiles=(3, 2),
        overlap_ratio=0.2,
        nms_iou=0.5,
        stride=1,
        obb=True,
        classes=[9, 10],
        show_text=False
    )
