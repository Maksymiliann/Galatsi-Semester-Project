import cv2
import os
import glob


def parse_txt_file(txt_path):
    """
    Read one TXT file and return a list of detections.
    Each detection contains:
      - vehicle_id
      - 4 corner points
      - det_class
      - conf_score
      - state
    """
    detections = []

    if not os.path.exists(txt_path):
        return detections

    with open(txt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        parts = line.split(";")

        # Skip malformed lines
        if len(parts) < 12:
            continue

        try:
            vehicle_id = parts[0]

            x1 = int(float(parts[1]))
            y1 = int(float(parts[2]))
            x2 = int(float(parts[3]))
            y2 = int(float(parts[4]))
            x3 = int(float(parts[5]))
            y3 = int(float(parts[6]))
            x4 = int(float(parts[7]))
            y4 = int(float(parts[8]))

            det_class = parts[9]
            conf_score = float(parts[10])
            state = parts[11].strip().lower()

            detections.append({
                "vehicle_id": vehicle_id,
                "points": [(x1, y1), (x2, y2), (x3, y3), (x4, y4)],
                "det_class": det_class,
                "conf_score": conf_score,
                "state": state
            })

        except Exception:
            # Ignore badly formatted lines
            continue

    return detections


def draw_detections(frame, detections):
    """
    Draw the bounding boxes on the frame.
    parked -> green
    moving -> red
    """
    for det in detections:
        pts = det["points"]
        vehicle_id = det["vehicle_id"]
        state = det["state"]

        if state == "parked":
            color = (0, 255, 0)   # Green in BGR
        elif state == "moving":
            color = (0, 0, 255)   # Red in BGR
        else:
            color = (0, 255, 255) # Yellow for unknown state

        # Draw the 4-sided bounding box
        for i in range(4):
            pt1 = pts[i]
            pt2 = pts[(i + 1) % 4]
            cv2.line(frame, pt1, pt2, color, 2)

        # Put label near first point
        label = f"ID:{vehicle_id} {state}"
        text_x, text_y = pts[0]

        cv2.putText(
            frame,
            label,
            (text_x, max(20, text_y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA
        )

    return frame


def process_video(video_path, txt_folder, output_path):
    """
    For each frame in the video:
      - read the corresponding TXT file
      - draw bounding boxes
      - save to output video
    """

    # Get sorted TXT files
    txt_files = sorted(glob.glob(os.path.join(txt_folder, "*.txt")))

    if len(txt_files) == 0:
        print("No TXT files found in the folder.")
        return

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video loaded: {video_path}")
    print(f"Frames in video: {frame_count}")
    print(f"TXT files found: {len(txt_files)}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Match each frame with the corresponding TXT file by index
        if frame_idx < len(txt_files):
            txt_path = txt_files[frame_idx]
            detections = parse_txt_file(txt_path)
            frame = draw_detections(frame, detections)

        out.write(frame)

        if frame_idx % 50 == 0:
            print(f"Processed frame {frame_idx}/{frame_count}")

        frame_idx += 1

    cap.release()
    out.release()

    print("Done.")
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    # ====== EDIT THESE PATHS ======
    video_path = r"C:/Users/makss/Git/Galatsi-Semester-Project/Dataset/Galatsi_Data_Semester_Project_stabilized/DJI_0314_D2_S4_S1.mp4"
    txt_folder = r"C:/Users/makss/Git/Galatsi-Semester-Project/Results/TXT_0314_filtered_zone_10"
    output_path = r"C:/Users/makss/Git/Galatsi-Semester-Project/Results/Images/test/reconstructed_video_zone_10.mp4"
    # ===============================

    process_video(video_path, txt_folder, output_path)