import os
import glob
from collections import defaultdict

import pandas as pd
from tqdm import tqdm

"""
This script “pads” missing detections in a sequence of per-frame TXT files by interpolating vehicle bounding boxes.

It loads all frame files from INPUT_DIR (CSV with ';' separator), cleans column names/types, and builds a track
for each vehicle_id by collecting its detections across frames. For each vehicle track, it looks for gaps where:
- the vehicle is missing for one or more intermediate frames, and
- the vehicle is labeled "stop" at both the start and end of the gap

For those gaps, it generates synthetic detections in the missing frames by linearly interpolating the 4-corner
bounding-box coordinates (veh_bb_x1..y4) and the confidence score between the two surrounding detections.
Interpolated rows are forced to state="stop" and keep the original det_class.

Finally, the script writes a new set of TXT files to OUTPUT_DIR, inserting the interpolated rows into the
corresponding frames, sorting by vehicle_id, and preserving the original file names. It prints how many padded
rows were added in total.
"""




# =========================
# CONFIG
# =========================
INPUT_DIR = r"C:/Users/makss/Git/Galatsi-Semester-Project/Results/TXT_0319_D2_S5_S1"
OUTPUT_DIR = r"C:/Users/makss/Git/Galatsi-Semester-Project/Results/TXT_0319_D2_S5_S1_padded"

FILE_PATTERN = "*.txt"

ID_COL = "vehicle_id"
STATE_COL = "state"
CONF_COL = "conf_score"
DET_CLASS_COL = "det_class"

BBOX_COLS = [
    "veh_bb_x1", "veh_bb_y1",
    "veh_bb_x2", "veh_bb_y2",
    "veh_bb_x3", "veh_bb_y3",
    "veh_bb_x4", "veh_bb_y4",
]


# =========================
# LECTURE DES FICHIERS
# =========================
def load_all_frames(input_dir, pattern):
    file_paths = sorted(glob.glob(os.path.join(input_dir, pattern)))
    if not file_paths:
        raise ValueError(f"Aucun fichier trouvé dans {input_dir} avec {pattern}")

    frames = []
    for path in tqdm(file_paths, desc="Loading frames"):
        df = pd.read_csv(path, sep=';', engine='python')
        df.columns = [c.strip() for c in df.columns]

        if STATE_COL in df.columns:
            df[STATE_COL] = df[STATE_COL].astype(str).str.strip()
        if ID_COL in df.columns:
            df[ID_COL] = df[ID_COL].astype(str).str.strip().astype(int)

        for col in BBOX_COLS + [CONF_COL]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        frames.append(df)

    return file_paths, frames


# =========================
# TRAJECTOIRES
# =========================
def build_vehicle_tracks(frames):
    tracks = defaultdict(list)
    for f_idx, df in tqdm(enumerate(frames), total=len(frames), desc="Building tracks"):
        for _, row in df.iterrows():
            tracks[row[ID_COL]].append((f_idx, row))

    for veh_id in tracks:
        tracks[veh_id].sort(key=lambda x: x[0])

    return tracks


# =========================
# INTERPOLATION
# =========================
def interpolate_row(row_start, row_end, alpha):
    new_row = row_start.copy()

    for col in BBOX_COLS:
        v0 = row_start[col]
        v1 = row_end[col]
        if pd.notna(v0) and pd.notna(v1):
            new_row[col] = (1 - alpha) * v0 + alpha * v1

    new_row[CONF_COL] = (1 - alpha) * row_start[CONF_COL] + alpha * row_end[CONF_COL]
    new_row[STATE_COL] = "stop"
    new_row[DET_CLASS_COL] = row_start[DET_CLASS_COL]

    return new_row


# =========================
# PADDING
# =========================
def compute_padding(frames, tracks):
    new_rows_per_frame = defaultdict(list)

    for veh_id, occurrences in tqdm(tracks.items(), desc="Padding tracks"):
        for k in range(len(occurrences) - 1):
            f_start, row_start = occurrences[k]
            f_end, row_end = occurrences[k + 1]

            if f_end <= f_start + 1:
                continue

            if row_start[STATE_COL] != "stop" or row_end[STATE_COL] != "stop":
                continue

            gap_length = f_end - f_start

            for f in range(f_start + 1, f_end):
                if (frames[f][ID_COL] == veh_id).any():
                    continue

                alpha = (f - f_start) / gap_length
                new_row = interpolate_row(row_start, row_end, alpha)
                new_row[ID_COL] = veh_id
                new_rows_per_frame[f].append(dict(new_row))

    return new_rows_per_frame


# =========================
# ÉCRITURE DES FICHIERS
# =========================
def write_frames_with_padding(file_paths, frames, new_rows_per_frame, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for f_idx, (in_path, df) in tqdm(
            list(enumerate(zip(file_paths, frames))),
            desc="Writing output"):
        
        if f_idx in new_rows_per_frame:
            df_new = pd.DataFrame(new_rows_per_frame[f_idx])
            df = pd.concat([df, df_new], ignore_index=True)

        df = df.sort_values(by=ID_COL).reset_index(drop=True)

        file_name = os.path.basename(in_path)
        df.to_csv(os.path.join(output_dir, file_name), sep=';', index=False)


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    file_paths, frames = load_all_frames(INPUT_DIR, FILE_PATTERN)
    tracks = build_vehicle_tracks(frames)
    new_rows_per_frame = compute_padding(frames, tracks)

    total_new = sum(len(v) for v in new_rows_per_frame.values())
    print(f"\nTotal padded rows added: {total_new}\n")

    write_frames_with_padding(file_paths, frames, new_rows_per_frame, OUTPUT_DIR)
    print("Finished.")
