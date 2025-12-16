import json
import os
import base64

import cv2
import numpy as np


# =========================
# CONFIG
# =========================
JSON_PATH      = r"C:/Users/makss/Git/Galatsi-Semester-Project/Results/Ground_truth/groundTruth.json"
OUT_MASK_PATH  = r"C:/Users/makss/Git/Galatsi-Semester-Project/Results/Ground_truth/mask_ids.png"
OUT_VIZ_PATH   = r"C:/Users/makss/Git/Galatsi-Semester-Project/Results/Ground_truth/overlay.png"


def load_base_image(data, json_path):
    """Try to reconstruct the original image from imageData or imagePath."""
    h = data["imageHeight"]
    w = data["imageWidth"]

    img = None

    # 1) Try imageData (base64 PNG inside JSON)
    image_data = data.get("imageData", None)
    if image_data:
        try:
            img_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception as e:
            print("Could not decode imageData from JSON:", e)

    # 2) Try external imagePath (relative to JSON file)
    if img is None:
        img_rel = data.get("imagePath", None)
        if img_rel is not None:
            img_path = os.path.join(os.path.dirname(json_path), img_rel)
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
            else:
                print("Image path not found on disk:", img_path)

    # 3) Fallback: blank image
    if img is None:
        print("Using blank background image.")
        img = np.zeros((h, w, 3), dtype=np.uint8)

    # Make sure size matches expected
    img = cv2.resize(img, (w, h))
    return img


def main():
    # ---- Load JSON ----
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    h = data["imageHeight"]
    w = data["imageWidth"]

    # ---- Base image (for visualization) ----
    base_img = load_base_image(data, JSON_PATH)
    viz_img = base_img.copy()

    # ---- Mask (each label -> unique integer ID) ----
    # You can use uint8 if you know you have < 255 labels
    mask = np.zeros((h, w), dtype=np.uint16)

    label_to_id = {}
    next_id = 1

    for shape in data["shapes"]:
        label = shape.get("label", "unknown")
        points = np.array(shape["points"], dtype=np.float32)

        # Convert float coords to int pixel coords
        pts = points.astype(np.int32)

        # Assign an ID to this label if not already
        if label not in label_to_id:
            label_to_id[label] = next_id
            next_id += 1
        idx = label_to_id[label]

        # ---- Fill polygon in mask with its ID ----
        cv2.fillPoly(mask, [pts], int(idx))

        # ---- Draw polygon on visualization image ----
        # deterministic pseudo-color based on idx
        color = (
            (37 * idx) % 256,
            (17 * idx) % 256,
            (93 * idx) % 256,
        )
        cv2.polylines(viz_img, [pts], isClosed=True, color=color, thickness=2)

        # Put label text roughly at centroid
        M = cv2.moments(pts)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            # fallback: first point
            cx, cy = int(pts[0][0]), int(pts[0][1])

        cv2.putText(
            viz_img,
            label,
            (cx, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    # ---- Save outputs ----
    # Mask: save as PNG, values = polygon IDs (1,2,3,...)
    # If you prefer a simple binary mask (255 where any polygon, else 0):
    #   bin_mask = (mask > 0).astype(np.uint8) * 255
    #   cv2.imwrite(OUT_MASK_PATH, bin_mask)
    cv2.imwrite(OUT_MASK_PATH, mask.astype(np.uint16))

    # Visualization image
    cv2.imwrite(OUT_VIZ_PATH, viz_img)

    print("Done!")
    print("Saved mask to:", OUT_MASK_PATH)
    print("Saved visualization to:", OUT_VIZ_PATH)
    print("Label -> ID mapping:", label_to_id)


if __name__ == "__main__":
    main()
