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
    # base tile size
    base_w = frame_w / tiles_x
    base_h = frame_h / tiles_y

    # overlap in pixels
    ov_w = int(base_w * overlap_ratio)
    ov_h = int(base_h * overlap_ratio)

    tiles = []
    for j in range(tiles_y):
        for i in range(tiles_x):
            x0 = int(i * base_w)
            y0 = int(j * base_h)
            x1 = int(min(frame_w, (i + 1) * base_w))
            y1 = int(min(frame_h, (j + 1) * base_h))

            # inflate with overlap (but clamp to image)
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
       {
         'cls': int, 'conf': float,
         'xyxy': np.array([x1,y1,x2,y2])   # axis-aligned box (used for NMS)
         'poly': np.array(shape=(4,2)) or None  # oriented polygon if OBB
       }
    Offsets are added to map tile coords back to global image coords.
    """
    dets = []

    # Try to infer task if needed
    if task == "auto":
        if hasattr(result, "obb") and result.obb is not None:
            task = "obb"
        else:
            task = "detect"

    if task == "obb" and hasattr(result, "obb") and result.obb is not None:
        obb = result.obb
        confs = obb.conf.cpu().numpy() if hasattr(obb, "conf") else np.array([])
        clss = obb.cls.cpu().numpy().astype(int) if hasattr(obb, "cls") else np.array([])
        # Try to get polygon points (xyxyxyxy) if available, else fall back to aabb
        poly = None
        if hasattr(obb, "xyxyxyxy"):
            poly = obb.xyxyxyxy.cpu().numpy()  # (N,8) -> 4 points
            poly = poly.reshape(-1, 4, 2)
        elif hasattr(obb, "xyxy"):
            # Some builds provide obb.xyxy (AABB). We'll draw aabb and poly=None.
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

        # If we have polygons, compute AABB for NMS
        if poly is not None:
            for i in range(poly.shape[0]):
                p = poly[i]
                p[:, 0] += x_offset
                p[:, 1] += y_offset
                x1, y1 = p[:, 0].min(), p[:, 1].min()
                x2, y2 = p[:, 0].max(), p[:, 1].max()
                dets.append({
                    "cls": int(clss[i]) if len(clss) > i else -1,
                    "conf": float(confs[i]) if len(confs) > i else 0.0,
                    "xyxy": np.array([x1, y1, x2, y2], dtype=float),
                    "poly": p.copy()
                })
        return dets

    # Fallback: normal detect (axis-aligned boxes)
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

def draw_detection(frame, det, class_names=None, color=(0, 255, 0)):
    """Draw AABB (and OBB polygon if present)."""
    x1, y1, x2, y2 = det["xyxy"].astype(int)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    if det.get("poly") is not None:
        poly = det["poly"].astype(int)
        cv2.polylines(frame, [poly], isClosed=True, color=color, thickness=2)

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
    Keeps the highest conf box and drops overlaps > iou_thr.
    For OBB, this uses the AABB envelope stored in det['xyxy'].
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
# Main video tiling inference
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
    obb: bool = False  # set True if using OBB model (e.g., yolo11x-obb)
):
    """
    Runs patch inference on a video and optionally saves an annotated video.
    """
    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ret, frame0 = cap.read()
    if not ret:
        raise RuntimeError("Could not read first frame.")
    H, W = frame0.shape[:2]

    # Decide grid
    if tiles is None:
        if num_tiles is None:
            tiles_x, tiles_y = (2, 2)  # default 4 tiles
        else:
            tiles_x, tiles_y = choose_grid_from_num_tiles(num_tiles)
    else:
        tiles_x, tiles_y = tiles

    # Prepare video writer if requested
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    frame_idx = 0
    processed = 0
    class_names = model.names if hasattr(model, "names") else None

    while True:
        if frame_idx == 0:
            frame = frame0
        else:
            ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % stride != 0:
            frame_idx += 1
            continue

        tiles_rc = make_tiles(H, W, tiles_x, tiles_y, overlap_ratio=overlap_ratio)

        # gather dets from all tiles
        all_dets = []
        for (x0, y0, x1, y1) in tiles_rc:
            tile = frame[y0:y1, x0:x1]

            # Run the model
            # Using .predict with numpy image; you can also use .__call__
            results = model.predict(
                source=tile,
                imgsz=imgsz,
                conf=conf,
                device=device,
                verbose=False
            )

            # Extract detections in global coords
            for res in results:
                dets = extract_detections(
                    res, x_offset=x0, y_offset=y0,
                    task="obb" if obb else "detect"
                )
                all_dets.extend(dets)

        # Merge duplicates across overlapping tiles
        merged = nms_merge(all_dets, iou_thr=nms_iou)

        # Draw
        out_frame = frame.copy()
        for det in merged:
            draw_detection(out_frame, det, class_names)

        if writer is not None:
            writer.write(out_frame)

        processed += 1
        frame_idx += 1
        if max_frames is not None and processed >= max_frames:
            break

    cap.release()
    if writer is not None:
        writer.release()

    print(f"Done. Processed {processed} frames"
          f"{' and wrote ' + output_path if output_path else ''}.")


# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    """
    Examples:
      - Choose tiles by grid (3x2 = 6 parts), 20% overlap:
          tiles=(3,2), overlap_ratio=0.2

      - Choose tiles by number (num_tiles=9) -> auto 3x3, 15% overlap:
          num_tiles=9, overlap_ratio=0.15

      - OBB model (e.g., yolo11x-obb):
          obb=True, model_path="yolo11x-obb.pt"
    """
    patch_infer_video(
        video_path="Dataset/Galatsi_Data_Semester_Project_archive/DJI_0004.MP4",
        model_path="yolo11x-obb.pt",  # or any YOLO model
        output_path="annotated.mp4",
        imgsz=1280,
        conf=0.25,
        device=None,          # e.g. "0" for CUDA:0
        tiles=(3, 2),         # or use num_tiles=6
        num_tiles=None,
        overlap_ratio=0.2,
        nms_iou=0.5,
        stride=1,
        max_frames=None,
        obb=True              # set False for normal detection models
    )
