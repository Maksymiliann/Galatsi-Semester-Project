import math
import cv2
import numpy as np
from ultralytics import YOLO
from stabilo import Stabilizer
import torch
from torchvision.ops import batched_nms
import supervision as sv   
from supervision import Detections
from supervision.tracker.byte_tracker.core import ByteTrack 
from collections import defaultdict, deque
import os


# ---------- TXT export helpers ----------
TXT_HEADER = (
    "vehicle_id; veh_bb_x1; veh_bb_y1; veh_bb_x2; veh_bb_y2; "
    "veh_bb_x3; veh_bb_y3; veh_bb_x4; veh_bb_y4; det_class; conf_score; state"
)

# Fixed widths for pretty-aligned columns (works in monospaced editors)
#        id    x1     y1     x2     y2     x3     y3     x4     y4   class   conf   state
_TXT_WIDTHS = [6,   10,    10,    10,    10,    10,    10,    10,    10,    8,     7,     6]

def _fmt_row_semicolon(fields):
    """Right-align each field to a fixed width, then join with '; '."""
    out = []
    for val, w in zip(fields, _TXT_WIDTHS):
        if isinstance(val, float):
            out.append(f"{val:>{w}.2f}")
        else:
            out.append(f"{str(val):>{w}}")
    return "; ".join(out)



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
         'xyxy': np.array([x1,y1,x2,y2]),   # AABB for NMS/merge + tracking
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
                    "xyxy": np.array([x1 + x_offset, y1 + y_offset,
                                      x2 + x_offset, y2 + y_offset], dtype=float),
                    "poly": None
                })
            return dets

        if poly is not None:
            for i in range(poly.shape[0]):
                p = poly[i].copy()
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
                "xyxy": np.array([x1 + x_offset, y1 + y_offset,
                                  x2 + x_offset, y2 + y_offset], dtype=float),
                "poly": None
            })
    return dets

def draw_detection(frame, det, class_names=None, color=(0,255,0),
                   show_text=True, show_aabb=False, show_poly=True, id_text=None):
    """
    Draw either OBB polygon or AABB + optional text and tracker ID.
    """
    x1, y1, x2, y2 = det["xyxy"].astype(int)

    # Only draw OBB polygon if present
    if det.get("poly") is not None and show_poly:
        poly = det["poly"].astype(int)
        cv2.polylines(frame, [poly], isClosed=True, color=color, thickness=2)
        if show_aabb:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)  # optional
    else:
        # No OBB → draw AABB
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    text_bits = []
    if id_text is not None:
        text_bits.append(f"ID {id_text}")
    if show_text:
        label = f"{det['cls']}"
        if class_names and 0 <= det["cls"] < len(class_names):
            label = class_names[det["cls"]]
        text_bits.append(f"{label} {det['conf']:.2f}")
    if text_bits:
        cv2.putText(frame, " | ".join(text_bits), (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

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


def nms_merge_fast(dets, iou_thr=0.5, device=None):
    """
    Class-wise NMS très rapide (PyTorch/torchvision). Utilise l'AABB même pour OBB.
    """
    if not dets:
        return []
    dev = "cuda" if (device is None or device == "0") else device
    # si CPU: dev="cpu"

    boxes = torch.tensor([d["xyxy"] for d in dets], dtype=torch.float32, device=dev)
    scores = torch.tensor([d["conf"] for d in dets], dtype=torch.float32, device=dev)
    labels = torch.tensor([d["cls"]  for d in dets], dtype=torch.int64,   device=dev)

    keep = batched_nms(boxes, scores, labels, iou_thr)
    keep_idx = keep.detach().cpu().tolist()
    return [dets[i] for i in keep_idx]


# ----------------------------
# Common tiled inference on a single frame (image array)
# ----------------------------
def process_frame_tiled(
    frame, model, imgsz=1280, conf=0.25, device=None,
    tiles=None, num_tiles=None, overlap_ratio=0.2, nms_iou=0.5,
    obb=False, classes=None, max_det=200, half=True
):
    H, W = frame.shape[:2]
    if tiles is None:
        tiles_x, tiles_y = choose_grid_from_num_tiles(num_tiles) if num_tiles else (2, 2)
    else:
        tiles_x, tiles_y = tiles

    tiles_rc = make_tiles(H, W, tiles_x, tiles_y, overlap_ratio=overlap_ratio)

    # 1) prépare la liste d’images tuiles
    imgs = [frame[y0:y1, x0:x1] for (x0, y0, x1, y1) in tiles_rc]

    # 2) une seule prédiction batch
    results = model.predict(
        source=imgs,
        imgsz=imgsz,
        conf=conf,
        device=device,
        verbose=False,
        classes=classes,
        max_det=max_det,   # limite le nb de boxes par tuile (ajuste selon besoin)
        half=half          # FP16 → plus rapide sur GPU
    )

    # 3) récupère les détections et remonte en coords full-frame
    all_dets = []
    for (x0, y0, x1, y1), res in zip(tiles_rc, results):
        dets = extract_detections(res, x_offset=x0, y_offset=y0, task="obb" if obb else "detect")
        all_dets.extend(dets)

    # NMS global (voir §2 pour version rapide)
    merged = nms_merge_fast(all_dets, iou_thr=nms_iou, device=device)
    return merged


def count_by_class(dets):
    """Count detections by class index."""
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
# Main video tiling inference + ByteTrack
# ----------------------------
import os  # <-- AJOUT

# ----------------------------
# Main video tiling inference + ByteTrack + per-frame TXT export
# ----------------------------
def patch_infer_video(
    video_path: str,
    model_path: str,
    output_path: str = None,
    imgsz: int = 1280,
    conf: float = 0.25,
    device: str = None,          # "0" | "cpu"
    tiles: tuple = None,         # (tiles_x, tiles_y)
    num_tiles: int = None,
    overlap_ratio: float = 0.2,
    nms_iou: float = 0.5,
    stride: int = 1,
    max_frames: int = None,
    obb: bool = False,
    classes=None,
    show_text: bool = True,
    use_stabilization: bool = True,
    reset_ref_every: int = 0,

    # ---- interrupteurs tracking / traces ----
    do_tracking: bool = True,    # active/désactive ByteTrack
    draw_traces: bool = False,   # dessiner les trajectoires

    # ---- ByteTrack params ----
    track_thresh: float = 0.25,
    match_thresh: float = 0.8,
    track_buffer: int = 30,

    # ---- mouvement vs arrêt (fenêtre + hystérèse) ----
    nb_frames_moving: int = 5,      # fenêtre de comparaison (frames)
    stop_speed_thresh: float = 0.8, # px/frame : <= STOP (vert)
    move_speed_thresh: float = 1.5, # px/frame : >= MOVING (rouge)
    use_hysteresis: bool = True,
    ema_alpha: float = 0.0,         # 0.0=off ; >0 pour lisser la vitesse (EMA)

    # ---- export TXT par frame ----
    export_txt_dir: str = None,      # ex: "Results/TXT" ; si None => pas d'export
    export_txt: bool = False      # <--- NEW: interrupteur global on/off
):
    """
    Run tiled YOLO inference + optional stabilization + optional ByteTrack on a video.
    Saves annotated output if output_path is given.

    Per-frame TXT export (if export_txt_dir is set):
      One file per processed frame: 0001.txt, 0002.txt, ...
      Each line:
        vehicle_id; x1; y1; x2; y2; x3; y3; x4; y4; det_class; conf_score; state
    """
    assert move_speed_thresh > stop_speed_thresh, "move_speed_thresh doit être > stop_speed_thresh pour une hystérèse utile."

    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    ret, frame0 = cap.read()
    if not ret:
        raise RuntimeError("Could not read first frame.")
    H, W = frame0.shape[:2]

    # Stabilizer
    stabilizer = Stabilizer() if use_stabilization else None
    if stabilizer is not None:
        stabilizer.set_ref_frame(frame0.copy())

    # Writer
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    # Tracker & trace annotator (optionnels)
    tracker = None
    trace_annotator = None
    if do_tracking:
        tracker = ByteTrack(
            track_activation_threshold=float(track_thresh),
            lost_track_buffer=int(track_buffer),
            minimum_matching_threshold=float(match_thresh),
            frame_rate=int(fps)
        )
        if draw_traces:
            trace_annotator = sv.TraceAnnotator(
                thickness=2, trace_length=60, position=sv.Position.CENTER
            )

    # --- états mouvement par track ---
    track_centers = {}    # tid -> list[(cx, cy)]
    track_speed_ema = {}  # tid -> float
    track_state = {}      # tid -> 'stop' | 'move'

    # --- export dir ---
    if export_txt and export_txt_dir is not None:
        os.makedirs(export_txt_dir, exist_ok=True)

    frame_idx = 0
    processed = 0
    class_names = model.names if hasattr(model, "names") else None

    def best_idx_for_track(tr_xyxy, merged_list):
        if not merged_list:
            return -1
        ious = [iou_xyxy(tr_xyxy, d["xyxy"]) for d in merged_list]
        return int(np.argmax(ious))

    def box_center_xyxy(xyxy_arr):
        x1, y1, x2, y2 = xyxy_arr
        return float((x1 + x2) * 0.5), float((y1 + y2) * 0.5)

    def aabb_to_poly_xyxy(xyxy_arr):
        x1, y1, x2, y2 = [int(v) for v in xyxy_arr]
        # ordre: (x1,y1)-(x2,y1)-(x2,y2)-(x1,y2)
        return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

    def update_speed_and_state(tid, cx, cy):
        hist = track_centers.get(tid)
        if hist is None:
            hist = []
            track_centers[tid] = hist
        hist.append((cx, cy))
        if len(hist) > (nb_frames_moving + 1):
            hist.pop(0)

        if len(hist) >= (nb_frames_moving + 1):
            (xa, ya) = hist[-(nb_frames_moving + 1)]
            (xb, yb) = hist[-1]
            dist = ((xb - xa)**2 + (yb - ya)**2) ** 0.5
            v_inst = dist / max(1, nb_frames_moving)  # px/frame

            if ema_alpha > 0.0:
                v_prev = track_speed_ema.get(tid, v_inst)
                v_ema = (1.0 - ema_alpha) * v_prev + ema_alpha * v_inst
                track_speed_ema[tid] = v_ema
                v_use = v_ema
            else:
                v_use = v_inst

            prev = track_state.get(tid, 'stop')
            if not use_hysteresis:
                state = 'move' if v_use >= move_speed_thresh else 'stop'
            else:
                if prev == 'stop':
                    state = 'move' if v_use >= move_speed_thresh else 'stop'
                else:
                    state = 'stop' if v_use <= stop_speed_thresh else 'move'
            track_state[tid] = state
            return v_use, state
        else:
            return None, track_state.get(tid, 'stop')

    while True:
        frame = frame0 if frame_idx == 0 else cap.read()[1]
        rows_for_txt = []  # <- will collect all rows for this frame
        if frame is None:
            break
        if frame_idx % stride != 0:
            frame_idx += 1
            continue

        # stabilize
        if stabilizer is not None:
            try:
                stabilizer.stabilize(frame)
                stab = stabilizer.warp_cur_frame()
                frame_proc = stab if stab is not None else frame
            except Exception:
                frame_proc = frame
        else:
            frame_proc = frame

        if stabilizer is not None and reset_ref_every > 0 and frame_idx % reset_ref_every == 0 and frame_idx != 0:
            stabilizer.set_ref_frame(frame_proc.copy())

        # detect (tiled)
        merged = process_frame_tiled(
            frame_proc, model, imgsz=imgsz, conf=conf, device=device,
            tiles=tiles, num_tiles=num_tiles, overlap_ratio=overlap_ratio,
            nms_iou=nms_iou, obb=obb, classes=classes
        )

        out_frame = frame_proc.copy()
        rows_for_txt = []  # <-- on accumule ici les lignes à écrire

        if not do_tracking:
            # pas de tracking : on dessine et on exporte sans ID, state=unknown
            for det in merged:
                draw_detection(out_frame, det, class_names,
                               show_text=show_text, show_aabb=False, show_poly=True, id_text=None)
                # export
                vid = -1
                poly = det.get("poly")
                if poly is not None:
                    pts = [(int(px), int(py)) for px, py in poly]
                else:
                    pts = aabb_to_poly_xyxy(det["xyxy"])
                # sécuriser 4 points
                if len(pts) != 4:
                    pts = pts[:4] if len(pts) > 4 else (pts + [pts[-1]] * (4 - len(pts)))
                cls_i = int(det["cls"])
                conf_i = float(det["conf"])
                state = "unknown"
                row = [vid, pts[0][0], pts[0][1], pts[1][0], pts[1][1],
                        pts[2][0], pts[2][1], pts[3][0], pts[3][1],
                        cls_i, conf_i, state]
                
                rows_for_txt.append(_fmt_row_semicolon(row))
        else:
            # merged -> sv.Detections
            if len(merged) > 0:
                xyxy = np.stack([d["xyxy"] for d in merged], axis=0).astype(np.float32)
                confs = np.array([d["conf"] for d in merged], dtype=np.float32)
                clss  = np.array([d["cls"]  for d in merged], dtype=np.int32)
            else:
                xyxy = np.empty((0, 4), dtype=np.float32)
                confs = np.empty((0,), dtype=np.float32)
                clss  = np.empty((0,), dtype=np.int32)

            dets_sv = Detections(xyxy=xyxy, confidence=confs, class_id=clss)
            tracks_det = tracker.update_with_detections(dets_sv)

            # traces (facultatif)
            if draw_traces and isinstance(tracks_det, Detections) and getattr(tracks_det, "tracker_id", None) is not None:
                out_frame = trace_annotator.annotate(scene=out_frame, detections=tracks_det)

            # boîtes + IDs + export
            if isinstance(tracks_det, Detections) and hasattr(tracks_det, "tracker_id"):
                for i in range(len(tracks_det)):
                    tid = int(tracks_det.tracker_id[i]) if tracks_det.tracker_id is not None else -1
                    tr_xyxy = tracks_det.xyxy[i].astype(float)

                    # vitesse / état
                    cx, cy = box_center_xyxy(tr_xyxy)
                    v_use, state = update_speed_and_state(tid, cx, cy)

                    # couleur pour le dessin
                    color = (0, 255, 0) if state == 'stop' else (0, 0, 255)

                    # associer à un merged pour récupérer poly/cls/conf
                    best_i = best_idx_for_track(tr_xyxy, merged)
                    if best_i >= 0:
                        cls_i = int(merged[best_i]["cls"])
                        conf_i = float(merged[best_i]["conf"])
                        if merged[best_i].get("poly") is not None:
                            pts = [(int(px), int(py)) for px, py in merged[best_i]["poly"]]
                            det_for_draw = {
                                "xyxy": tr_xyxy,
                                "poly": merged[best_i]["poly"],
                                "cls": cls_i,
                                "conf": conf_i
                            }
                            id_str = f"{tid}"
                            if show_text and v_use is not None:
                                id_str = f"{tid} | {v_use:.2f}px/f"
                            draw_detection(out_frame, det_for_draw, class_names,
                                           show_text=False, show_aabb=False, show_poly=True,
                                           id_text=id_str, color=color)
                        else:
                            pts = aabb_to_poly_xyxy(tr_xyxy)
                            x1, y1, x2, y2 = [int(v) for v in tr_xyxy]
                            cv2.rectangle(out_frame, (x1, y1), (x2, y2), color, 2)
                            label = f"ID {tid}"
                            if show_text and v_use is not None:
                                label += f" | {v_use:.2f}px/f"
                            cv2.putText(out_frame, label, (x1, max(0, y1 - 6)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
                    else:
                        # pas de merged associé (rare) → AABB + valeurs par défaut
                        cls_i = -1
                        conf_i = 0.0
                        pts = aabb_to_poly_xyxy(tr_xyxy)
                        x1, y1, x2, y2 = [int(v) for v in tr_xyxy]
                        cv2.rectangle(out_frame, (x1, y1), (x2, y2), color, 2)
                        label = f"ID {tid}"
                        if show_text and v_use is not None:
                            label += f" | {v_use:.2f}px/f"
                        cv2.putText(out_frame, label, (x1, max(0, y1 - 6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

                    # export line
                    if best_i >= 0 and merged[best_i].get("poly") is not None:
                        pts = [(float(px), float(py)) for px, py in merged[best_i]["poly"]]
                    else:
                        pts = aabb_to_poly_xyxy(tr_xyxy)

                    # ensure 4 points
                    if len(pts) != 4:
                        pts = pts[:4] if len(pts) > 4 else (pts + [pts[-1]] * (4 - len(pts)))

                    row = [
                        int(tid),
                        pts[0][0], pts[0][1], pts[1][0], pts[1][1],
                        pts[2][0], pts[2][1], pts[3][0], pts[3][1],
                        int(cls_i), float(conf_i), ("stop" if state == "stop" else "move")
                    ]
                    rows_for_txt.append(_fmt_row_semicolon(row))

        if writer is not None:
            writer.write(out_frame)

        # --- write TXT file for this frame (if enabled) ---
        if export_txt and export_txt_dir is not None:
            frame_out_idx = processed + 1  # 1-based
            txt_path = os.path.join(export_txt_dir, f"{frame_out_idx:04d}.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(TXT_HEADER + "\n")
                if rows_for_txt:
                    f.write("\n".join(rows_for_txt))

        # logs
        counts_raw = count_by_class(merged)
        total = sum(counts_raw.values())
        counts_by_name = ({(class_names[k] if 0 <= k < len(class_names) else str(k)): v
                           for k, v in counts_raw.items()} if class_names
                          else {str(k): v for k, v in counts_raw.items()})
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
# Example
# ----------------------------
if __name__ == "__main__":
    # VIDEO example
    patch_infer_video(
        video_path="Dataset/Galatsi_Data_Semester_Project_archive/DJI_0004.MP4",
        model_path="yolo11m-obb.pt",
        output_path="Results/Video/tracking_color12.mp4",
        imgsz=1280,
        conf=0.35,
        tiles=(4, 3),
        overlap_ratio=0.25,
        nms_iou=0.4,
        stride=1,
        obb=True,
        classes=[9, 10],
        show_text=False,             # True to see "ID <id> <class> <conf>"
        use_stabilization=True,
        reset_ref_every=0,
        device="0",
        do_tracking=True,
        draw_traces=False,
        # ByteTrack tuning
        track_thresh=0.25,   # start tracks from this conf
        match_thresh=0.8,    # association strictness
        track_buffer=30,      # keep IDs this many frames when briefly lost
        # ---- mouvement vs arrêt (fenêtre + hystérèse) ----
        nb_frames_moving = 5,      # fenêtre de comparaison (frames)
        stop_speed_thresh = 0.5, # px/frame : <= STOP (vert)
        move_speed_thresh = 1.0, # px/frame : >= MOVING (rouge) ; doit être > stop_speed_thresh
        use_hysteresis = True,    # activer l’hystérèse
        ema_alpha = 0.8,          # 0.0=off ; >0 pour lisser la vitesse (EMA), 1 pour pas de lissage
        # ---- export TXT par frame ----
        export_txt_dir = "Results/TXT12",   
        export_txt = True      # <--- interrupteur global on/off
    )

    # IMAGE example (unchanged)
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
