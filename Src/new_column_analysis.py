import os
import glob
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

"""
This script reads a folder of TXT files containing:
    vehicle_id
    parking_zone
    timestamp_in
    timestamp_out

It generates:
1. One occupancy plot per zone:
      timestamp vs number of parked vehicles in that zone
2. Global plots:
      - number of parking events per zone
      - total parked duration per zone
      - histogram of parking durations
3. A TXT event log in chronological order:
      Timestamp 0127, vehicle ID 12 parked in zone 13
      Timestamp 0355, vehicle ID 12 left zone 13
4. A final summary section:
      - parked and left
      - parked but did not leave
      - already parked since beginning but left later
      - already parked since beginning and still parked at the end

Important:
- Each parked episode is identified using unique combinations of:
      vehicle_id, parking_zone, timestamp_in, timestamp_out
- A row is considered parked when parking_zone != -1
- If timestamp_in == first global timestamp, the vehicle is considered
  "already parked since the beginning"
- If timestamp_out == last global timestamp, the vehicle is considered
  "did not leave" within the observed time window
"""


###########################################################
# CONFIG
###########################################################
INPUT_FOLDER = r"C:/Users/makss/Git/Galatsi-Semester-Project/Results/TXT_0314_with_parking_zone_timestamps_gap_1_2"
OUTPUT_FOLDER = r"C:/Users/makss/Git/Galatsi-Semester-Project/Results/TXT_0314_parking_analysis_outputs_1_2"

EVENTS_TXT_PATH = os.path.join(OUTPUT_FOLDER, "parking_events_summary.txt")
PLOTS_FOLDER = os.path.join(OUTPUT_FOLDER, "plots")
ZONE_PLOTS_FOLDER = os.path.join(PLOTS_FOLDER, "per_zone_occupancy")

FILE_EXTENSION = "*.txt"

# If True, occupancy is reconstructed frame-by-frame from parked rows in each file
# This is usually the safest option.
USE_ROWS_DIRECTLY_FOR_OCCUPANCY = True


###########################################################
# HELPERS
###########################################################
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def extract_timestamp_from_filename(path):
    """
    Example:
        0001.txt -> 1
        0234.txt -> 234
        2298.txt -> 2298
    """
    stem = Path(path).stem
    return int(stem)


def format_ts(ts):
    """
    Pretty formatting like 0001, 0127, 2298.
    """
    try:
        return f"{int(ts):04d}"
    except Exception:
        return str(ts)


###########################################################
# LOAD ALL FILES
###########################################################
def load_all_txts(input_folder):
    txt_files = sorted(glob.glob(os.path.join(input_folder, FILE_EXTENSION)))
    if len(txt_files) == 0:
        raise FileNotFoundError(f"No TXT files found in {input_folder}")

    all_rows = []

    for txt_path in tqdm(txt_files, desc="Loading TXT files"):
        timestamp = extract_timestamp_from_filename(txt_path)

        df = pd.read_csv(txt_path, sep=";", engine="python")
        df.columns = [c.strip() for c in df.columns]

        df["timestamp"] = timestamp
        df["_src_file"] = os.path.basename(txt_path)

        all_rows.append(df)

    df_all = pd.concat(all_rows, ignore_index=True)

    # Clean / standardize some columns if present
    if "vehicle_id" in df_all.columns:
        df_all["vehicle_id"] = pd.to_numeric(df_all["vehicle_id"], errors="coerce")

    if "parking_zone" in df_all.columns:
        df_all["parking_zone"] = pd.to_numeric(df_all["parking_zone"], errors="coerce").fillna(-1).astype(int)

    for col in ["timestamp_in", "timestamp_out"]:
        if col in df_all.columns:
            df_all[col] = pd.to_numeric(df_all[col], errors="coerce")

    return df_all, txt_files


###########################################################
# BUILD PARKING EPISODES
###########################################################
def build_unique_episodes(df_all):
    """
    Each parked episode is uniquely identified by:
        vehicle_id, parking_zone, timestamp_in, timestamp_out

    Returns a dataframe with one row per episode.
    """
    required = ["vehicle_id", "parking_zone", "timestamp_in", "timestamp_out"]
    for c in required:
        if c not in df_all.columns:
            raise ValueError(f"Missing required column: {c}")

    parked = df_all[df_all["parking_zone"] != -1].copy()

    # Keep only rows where timestamps are defined
    parked = parked[
        parked["timestamp_in"].notna() &
        parked["timestamp_out"].notna() &
        parked["vehicle_id"].notna()
    ].copy()

    episodes = parked[["vehicle_id", "parking_zone", "timestamp_in", "timestamp_out"]].drop_duplicates().copy()

    episodes["vehicle_id"] = episodes["vehicle_id"].astype(int)
    episodes["parking_zone"] = episodes["parking_zone"].astype(int)
    episodes["timestamp_in"] = episodes["timestamp_in"].astype(int)
    episodes["timestamp_out"] = episodes["timestamp_out"].astype(int)

    episodes["duration"] = episodes["timestamp_out"] - episodes["timestamp_in"] + 1

    episodes = episodes.sort_values(["timestamp_in", "timestamp_out", "vehicle_id", "parking_zone"]).reset_index(drop=True)
    return episodes


###########################################################
# OCCUPANCY PER ZONE OVER TIME
###########################################################
def build_zone_occupancy(df_all):
    """
    For each timestamp and each zone:
        count number of unique parked vehicles
    """
    parked_rows = df_all[df_all["parking_zone"] != -1].copy()

    occ = (
        parked_rows.groupby(["timestamp", "parking_zone"])["vehicle_id"]
        .nunique()
        .reset_index(name="n_parked")
    )

    return occ


###########################################################
# PLOTS
###########################################################
def plot_zone_occupancy(occ, out_folder):
    ensure_dir(out_folder)

    zones = sorted([z for z in occ["parking_zone"].dropna().unique() if z != -1])

    for zone in zones:
        sub = occ[occ["parking_zone"] == zone].sort_values("timestamp")

        plt.figure(figsize=(10, 5))
        plt.plot(sub["timestamp"], sub["n_parked"])
        plt.xlabel("Timestamp")
        plt.ylabel("Number of parked vehicles")
        plt.title(f"Zone {zone} occupancy over time")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_folder, f"zone_{zone:02d}_occupancy.png"), dpi=150)
        plt.close()


def plot_events_per_zone(episodes, out_path):
    counts = (
        episodes.groupby("parking_zone")
        .size()
        .reset_index(name="n_events")
        .sort_values("parking_zone")
    )

    plt.figure(figsize=(10, 5))
    plt.bar(counts["parking_zone"].astype(str), counts["n_events"])
    plt.xlabel("Zone")
    plt.ylabel("Number of parking events")
    plt.title("Parking events per zone")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_total_duration_per_zone(episodes, out_path):
    durations = (
        episodes.groupby("parking_zone")["duration"]
        .sum()
        .reset_index()
        .sort_values("parking_zone")
    )

    plt.figure(figsize=(10, 5))
    plt.bar(durations["parking_zone"].astype(str), durations["duration"])
    plt.xlabel("Zone")
    plt.ylabel("Total parked duration")
    plt.title("Total parked duration per zone")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_duration_histogram(episodes, out_path):
    plt.figure(figsize=(10, 5))
    plt.hist(episodes["duration"], bins=30)
    plt.xlabel("Parking duration")
    plt.ylabel("Count")
    plt.title("Histogram of parking durations")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


###########################################################
# EVENT LOG
###########################################################
def build_chronological_events(episodes):
    """
    Build ordered events like:
        Timestamp 0127, vehicle ID 12 parked in zone 13
        Timestamp 0355, vehicle ID 12 left zone 13
    """
    event_rows = []

    for _, row in episodes.iterrows():
        veh = int(row["vehicle_id"])
        zone = int(row["parking_zone"])
        t_in = int(row["timestamp_in"])
        t_out = int(row["timestamp_out"])

        event_rows.append({
            "timestamp": t_in,
            "kind": 0,  # park before leave if same timestamp
            "text": f"Timestamp {format_ts(t_in)}, vehicle ID {veh} parked in zone {zone}"
        })

        event_rows.append({
            "timestamp": t_out,
            "kind": 1,
            "text": f"Timestamp {format_ts(t_out)}, vehicle ID {veh} left zone {zone}"
        })

    events_df = pd.DataFrame(event_rows).sort_values(["timestamp", "kind", "text"]).reset_index(drop=True)
    return events_df


###########################################################
# SUMMARY TEXT
###########################################################
def build_summary_lines(episodes, first_ts, last_ts):
    """
    Examples:
    - vehicle ID 12 parked in zone 13 at timestamp 0127 and left at 0355
    - vehicle ID 52 parked in zone 7 at timestamp 2934 but did not leave
    - vehicle ID 2 was already parked since the beginning in zone 5 but left at timestamp 5434
    """
    lines = []

    for _, row in episodes.iterrows():
        veh = int(row["vehicle_id"])
        zone = int(row["parking_zone"])
        t_in = int(row["timestamp_in"])
        t_out = int(row["timestamp_out"])

        started_at_beginning = (t_in == first_ts)
        ended_at_last = (t_out == last_ts)

        if started_at_beginning and ended_at_last:
            lines.append(
                f"vehicle ID {veh} was already parked since the beginning in zone {zone} "
                f"and was still there at the end"
            )
        elif started_at_beginning and not ended_at_last:
            lines.append(
                f"vehicle ID {veh} was already parked since the beginning in zone {zone} "
                f"but left at timestamp {format_ts(t_out)}"
            )
        elif not started_at_beginning and ended_at_last:
            lines.append(
                f"vehicle ID {veh} parked in zone {zone} at timestamp {format_ts(t_in)} "
                f"but did not leave"
            )
        else:
            lines.append(
                f"vehicle ID {veh} parked in zone {zone} at timestamp {format_ts(t_in)} "
                f"and left at {format_ts(t_out)}"
            )

    return lines


###########################################################
# EXTRA SUMMARY STATS
###########################################################
def build_global_stats(episodes, occ, first_ts, last_ts):
    lines = []

    lines.append("Global statistics:")
    lines.append(f"- First timestamp: {format_ts(first_ts)}")
    lines.append(f"- Last timestamp: {format_ts(last_ts)}")
    lines.append(f"- Number of parking episodes: {len(episodes)}")
    lines.append(f"- Number of unique vehicles that parked: {episodes['vehicle_id'].nunique()}")

    if len(episodes) > 0:
        lines.append(f"- Mean parking duration: {episodes['duration'].mean():.2f}")
        lines.append(f"- Median parking duration: {episodes['duration'].median():.2f}")
        lines.append(f"- Max parking duration: {episodes['duration'].max()}")

    if len(occ) > 0:
        peak_row = occ.loc[occ["n_parked"].idxmax()]
        lines.append(
            f"- Peak observed zone occupancy: zone {int(peak_row['parking_zone'])} "
            f"had {int(peak_row['n_parked'])} parked vehicles at timestamp {format_ts(int(peak_row['timestamp']))}"
        )

    return lines


###########################################################
# SAVE EVENT TXT
###########################################################
def save_event_txt(events_df, summary_lines, stats_lines, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Chronological parking events\n")
        f.write("=" * 40 + "\n\n")

        for _, row in events_df.iterrows():
            f.write(row["text"] + "\n")

        f.write("\n")
        f.write("=" * 40 + "\n")
        f.write("Summary:\n")
        f.write("=" * 40 + "\n\n")

        for line in summary_lines:
            f.write(line + "\n")

        f.write("\n")
        f.write("=" * 40 + "\n")
        f.write("Additional statistics:\n")
        f.write("=" * 40 + "\n\n")

        for line in stats_lines:
            f.write(line + "\n")


###########################################################
# OPTIONAL CSV EXPORTS
###########################################################
def save_csv_outputs(episodes, occ, output_folder):
    episodes.to_csv(os.path.join(output_folder, "parking_episodes.csv"), index=False)
    occ.to_csv(os.path.join(output_folder, "zone_occupancy_over_time.csv"), index=False)


###########################################################
# MAIN
###########################################################
if __name__ == "__main__":
    ensure_dir(OUTPUT_FOLDER)
    ensure_dir(PLOTS_FOLDER)
    ensure_dir(ZONE_PLOTS_FOLDER)

    print("[1/6] Loading all TXT files...")
    df_all, txt_files = load_all_txts(INPUT_FOLDER)

    if "parking_zone" not in df_all.columns:
        raise ValueError("The input TXT files do not contain the column 'parking_zone'.")

    if "timestamp_in" not in df_all.columns or "timestamp_out" not in df_all.columns:
        raise ValueError("The input TXT files must contain 'timestamp_in' and 'timestamp_out'.")

    first_ts = int(df_all["timestamp"].min())
    last_ts = int(df_all["timestamp"].max())

    print("[2/6] Building unique parking episodes...")
    episodes = build_unique_episodes(df_all)

    print("[3/6] Building zone occupancy...")
    occ = build_zone_occupancy(df_all)

    print("[4/6] Creating plots...")
    if len(occ) > 0:
        plot_zone_occupancy(occ, ZONE_PLOTS_FOLDER)

    if len(episodes) > 0:
        plot_events_per_zone(episodes, os.path.join(PLOTS_FOLDER, "parking_events_per_zone.png"))
        plot_total_duration_per_zone(episodes, os.path.join(PLOTS_FOLDER, "total_parked_duration_per_zone.png"))
        plot_duration_histogram(episodes, os.path.join(PLOTS_FOLDER, "parking_duration_histogram.png"))

    print("[5/6] Building event log and summary...")
    events_df = build_chronological_events(episodes)
    summary_lines = build_summary_lines(episodes, first_ts, last_ts)
    stats_lines = build_global_stats(episodes, occ, first_ts, last_ts)

    save_event_txt(events_df, summary_lines, stats_lines, EVENTS_TXT_PATH)

    print("[6/6] Saving CSV outputs...")
    save_csv_outputs(episodes, occ, OUTPUT_FOLDER)

    print("\n================ DONE ================\n")
    print(f"Input folder         : {INPUT_FOLDER}")
    print(f"Output folder        : {OUTPUT_FOLDER}")
    print(f"Event TXT            : {EVENTS_TXT_PATH}")
    print(f"Plots folder         : {PLOTS_FOLDER}")
    print(f"Zone plots folder    : {ZONE_PLOTS_FOLDER}")
    print(f"TXT files loaded     : {len(txt_files)}")
    print(f"Parking episodes     : {len(episodes)}")
    print(f"Unique parked vehs   : {episodes['vehicle_id'].nunique() if len(episodes) > 0 else 0}")