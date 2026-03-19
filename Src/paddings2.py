import os
import re
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# =========================
# CONFIG
# =========================
INPUT_DIR = r"C:/Users/makss/Git/Galatsi-Semester-Project/Results/TXT_0314_D2_S4_S1"
OUTPUT_DIR = r"C:/Users/makss/Git/Galatsi-Semester-Project/Results/TXT_0314_D2_S4_S1_padded"

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

SEP = ";"

EXPECTED_COLUMNS = [
    ID_COL,
    "veh_bb_x1", "veh_bb_y1",
    "veh_bb_x2", "veh_bb_y2",
    "veh_bb_x3", "veh_bb_y3",
    "veh_bb_x4", "veh_bb_y4",
    DET_CLASS_COL,
    CONF_COL,
    STATE_COL,
]


# =========================
# HELPERS
# =========================
def natural_key(path):
    name = Path(path).stem
    parts = re.split(r"(\d+)", name)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def read_txt_file(path):
    if os.path.getsize(path) == 0:
        return pd.DataFrame(columns=EXPECTED_COLUMNS)

    # skipinitialspace=True helps after separators like "; "
    df = pd.read_csv(path, sep=SEP, skipinitialspace=True)

    # Remove leading/trailing spaces from column names
    df.columns = df.columns.str.strip()

    # If duplicate columns still exist after stripping, keep first
    df = df.loc[:, ~df.columns.duplicated()]

    # Make sure required columns exist
    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    # Keep only the standard columns in the final output
    return df[EXPECTED_COLUMNS].copy()


def ensure_numeric_bbox(df):
    df = df.copy()
    for col in BBOX_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[CONF_COL] = pd.to_numeric(df[CONF_COL], errors="coerce")
    return df


def interpolate_row(row_a, row_b, alpha):
    new_row = {}

    new_row[ID_COL] = row_a[ID_COL]

    for col in BBOX_COLS:
        a_val = pd.to_numeric(row_a[col], errors="coerce")
        b_val = pd.to_numeric(row_b[col], errors="coerce")
        new_row[col] = a_val + alpha * (b_val - a_val)

    new_row[DET_CLASS_COL] = row_a[DET_CLASS_COL]

    a_conf = pd.to_numeric(row_a[CONF_COL], errors="coerce")
    b_conf = pd.to_numeric(row_b[CONF_COL], errors="coerce")

    if pd.notna(a_conf) and pd.notna(b_conf):
        new_row[CONF_COL] = a_conf + alpha * (b_conf - a_conf)
    else:
        new_row[CONF_COL] = a_conf if pd.notna(a_conf) else b_conf

    state_a = str(row_a[STATE_COL]).strip().lower()
    state_b = str(row_b[STATE_COL]).strip().lower()
    new_row[STATE_COL] = state_a if state_a == state_b else "stop"

    return new_row


def format_output_df(df):
    if df.empty:
        return pd.DataFrame(columns=EXPECTED_COLUMNS)

    df = df.copy()

    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    df = df[EXPECTED_COLUMNS]

    for col in BBOX_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").round(2)

    df[CONF_COL] = pd.to_numeric(df[CONF_COL], errors="coerce").round(2)

    if ID_COL in df.columns:
        df = df.sort_values(ID_COL, kind="stable")

    return df


# =========================
# MAIN
# =========================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    file_paths = sorted(glob.glob(os.path.join(INPUT_DIR, FILE_PATTERN)), key=natural_key)
    print(f"Found {len(file_paths)} files.")

    # 1. Read all frames
    frames = []
    for path in tqdm(file_paths, desc="Reading files"):
        df = read_txt_file(path)
        df = ensure_numeric_bbox(df)
        frames.append(df)

    # 2. Build appearances
    appearances = {}
    for frame_idx, df in enumerate(tqdm(frames, desc="Building appearances")):
        if df.empty:
            continue

        df_unique = df.drop_duplicates(subset=[ID_COL], keep="first")

        for _, row in df_unique.iterrows():
            vid = row[ID_COL]
            if pd.isna(vid):
                continue
            appearances.setdefault(vid, []).append((frame_idx, row.copy()))

    # 3. Padding interpolation
    padded_rows_per_frame = {i: [] for i in range(len(frames))}
    total_inserted = 0

    for vid, obs in tqdm(appearances.items(), desc="Interpolating gaps"):
        for k in range(len(obs) - 1):
            f0, row0 = obs[k]
            f1, row1 = obs[k + 1]

            gap = f1 - f0 - 1
            if gap <= 0:
                continue

            for missing_frame in range(f0 + 1, f1):
                alpha = (missing_frame - f0) / (f1 - f0)
                new_row = interpolate_row(row0, row1, alpha)
                padded_rows_per_frame[missing_frame].append(new_row)
                total_inserted += 1

    print(f"Total padded rows: {total_inserted}")

    # 4. Write output
    for frame_idx, path in enumerate(tqdm(file_paths, desc="Writing output")):
        original_df = frames[frame_idx]

        if padded_rows_per_frame[frame_idx]:
            padded_df = pd.DataFrame(padded_rows_per_frame[frame_idx])

            if not original_df.empty:
                existing_ids = set(original_df[ID_COL].astype(str))
                padded_df = padded_df[~padded_df[ID_COL].astype(str).isin(existing_ids)]

            merged_df = pd.concat([original_df, padded_df], ignore_index=True)
        else:
            merged_df = original_df.copy()

        merged_df = format_output_df(merged_df)

        out_path = os.path.join(OUTPUT_DIR, os.path.basename(path))
        merged_df.to_csv(out_path, sep=SEP, index=False)

    print("Done!")


if __name__ == "__main__":
    main()