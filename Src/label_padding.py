import os
import glob
from pathlib import Path
import pandas as pd

# =========================
# CONFIG
# =========================
INPUT_DIR = r"C:/Users/makss/Git/Galatsi-Semester-Project/Results/TXT_0314_D2_S4_S1_padded"
OUTPUT_DIR = r"C:/Users/makss/Git/Galatsi-Semester-Project/Results/TXT_0314_D2_S4_S1_padded_state_corrected"

FILE_PATTERN = "*.txt"
SEP = ";"

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

# Maximum consecutive "stop" frames between two "move" segments
# that should be relabeled as "move"
MAX_STOP_GAP_TO_FLIP = 4000

# Whether to lowercase/strip state values for safety
NORMALIZE_STATE_TEXT = True


def natural_sort_key(path):
    """
    Sort files like:
    1.txt, 2.txt, 10.txt
    instead of:
    1.txt, 10.txt, 2.txt
    """
    stem = Path(path).stem
    try:
        return int(stem)
    except ValueError:
        return stem


def normalize_state(s):
    if pd.isna(s):
        return s
    s = str(s)
    if NORMALIZE_STATE_TEXT:
        s = s.strip().lower()
    return s


def validate_columns(df, file_path):
    missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {file_path}: {missing}")


def load_all_frames(input_dir, file_pattern):
    """
    Load all txt files and store them with frame index.
    Returns one concatenated dataframe with:
    - frame_idx
    - source_file
    - original row data
    """
    file_paths = sorted(
        glob.glob(os.path.join(input_dir, file_pattern)),
        key=natural_sort_key
    )

    if not file_paths:
        raise FileNotFoundError(f"No files found in {input_dir} with pattern {file_pattern}")

    all_dfs = []

    for frame_idx, file_path in enumerate(file_paths):
        df = pd.read_csv(file_path, sep=SEP)
        validate_columns(df, file_path)

        df = df.copy()
        df["frame_idx"] = frame_idx
        df["source_file"] = file_path
        df["row_in_file"] = range(len(df))

        df[STATE_COL] = df[STATE_COL].apply(normalize_state)

        all_dfs.append(df)

    big_df = pd.concat(all_dfs, ignore_index=True)
    return big_df, file_paths


def relabel_short_stop_between_moves(states, max_stop_gap):
    """
    Input: list of states for one vehicle over time
    Output: corrected list

    Logic:
    If we find:
        move ... stop stop stop ... move
    and the stop-run length <= max_stop_gap,
    then convert that stop-run to move.
    """
    corrected = states.copy()
    n = len(corrected)

    i = 0
    while i < n:
        if corrected[i] != "stop":
            i += 1
            continue

        # Find stop segment [start, end]
        start = i
        while i < n and corrected[i] == "stop":
            i += 1
        end = i - 1

        stop_len = end - start + 1

        prev_state = corrected[start - 1] if start - 1 >= 0 else None
        next_state = corrected[end + 1] if end + 1 < n else None

        # Flip only if this stop run is sandwiched between moves
        if prev_state == "move" and next_state == "move" and stop_len <= max_stop_gap:
            for j in range(start, end + 1):
                corrected[j] = "move"

    return corrected


def apply_vehicle_state_logic(big_df, max_stop_gap):
    """
    Apply correction vehicle by vehicle.
    """
    big_df = big_df.copy()
    big_df["corrected_state"] = big_df[STATE_COL]

    # Process each vehicle independently in time order
    for veh_id, group in big_df.groupby(ID_COL):
        group_sorted = group.sort_values("frame_idx").copy()

        states = group_sorted[STATE_COL].tolist()
        corrected_states = relabel_short_stop_between_moves(states, max_stop_gap)

        big_df.loc[group_sorted.index, "corrected_state"] = corrected_states

    return big_df


def write_corrected_files(big_df, file_paths, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for frame_idx, file_path in enumerate(file_paths):
        frame_df = big_df[big_df["frame_idx"] == frame_idx].copy()

        # Replace original state with corrected state
        frame_df[STATE_COL] = frame_df["corrected_state"]

        # Keep original row order
        frame_df = frame_df.sort_values("row_in_file")

        # Keep only expected output columns
        frame_df = frame_df[EXPECTED_COLUMNS]

        out_path = os.path.join(output_dir, os.path.basename(file_path))
        frame_df.to_csv(out_path, sep=SEP, index=False)

    print(f"Done. Corrected files written to:\n{output_dir}")


def main():
    print("Loading files...")
    big_df, file_paths = load_all_frames(INPUT_DIR, FILE_PATTERN)

    print("Applying state correction logic...")
    big_df_corrected = apply_vehicle_state_logic(
        big_df,
        max_stop_gap=MAX_STOP_GAP_TO_FLIP
    )

    print("Writing corrected files...")
    write_corrected_files(big_df_corrected, file_paths, OUTPUT_DIR)

    # Optional summary
    changed = (big_df_corrected[STATE_COL] != big_df_corrected["corrected_state"]).sum()
    print(f"Number of state changes: {changed}")


if __name__ == "__main__":
    main()