import pandas as pd
import matplotlib.pyplot as plt


# =========================
# CONFIG – CHANGE PATHS HERE
# =========================
PER_ZONE_PER_FRAME_CSV = r"C:/Users/makss/Git/Galatsi-Semester-Project/Results/occupancy_analysis/per_zones/test0314/per_zone_per_frame.csv"
PER_ZONE_SUMMARY_CSV   = r"C:/Users/makss/Git/Galatsi-Semester-Project/Results/occupancy_analysis/per_zones/test0314/per_zone_summary.csv"
GLOBAL_RESULTS_CSV     = r"C:/Users/makss/Git/Galatsi-Semester-Project/Results/occupancy_analysis/per_zones/test0314/global_results.csv"


# =========================
# SAFE CSV READER
# =========================
def read_csv_safely(path, expected_cols):
    df = pd.read_csv(path)

    if set(expected_cols).issubset(df.columns):
        return df

    if df.shape[1] == 1:
        col0 = df.columns[0]
        split = df[col0].astype(str).str.split(",", expand=True)
        split.columns = split.iloc[0]
        split = split[1:].reset_index(drop=True)
        return split

    raise ValueError(
        f"Impossible de trouver les colonnes {expected_cols} dans {path}. "
        f"Colonnes trouvées: {df.columns}"
    )


# =========================
# LOADING + PREPROCESSING
# =========================
def print_estimated_vehicles(per_zone_summary):
    print("\n=== Estimated number of parking places per zone ===")
    df = per_zone_summary.copy()
    df["zone_id"] = df["zone_id"].astype(int)
    df = df.sort_values("zone_id")

    for _, row in df.iterrows():
        # only print if column exists
        if "est_capacity_linear_rounded" in df.columns:
            print(f"Zone {int(row['zone_id'])}: ~{row['est_capacity_linear_rounded']} places")
        else:
            print(f"Zone {int(row['zone_id'])}: (no est_capacity_linear_rounded column found)")

def load_data():
    per_zone_per_frame = read_csv_safely(
        PER_ZONE_PER_FRAME_CSV,
        expected_cols=["frame", "zone_id", "vehicles_in_zone"]
    )
    per_zone_summary = read_csv_safely(
        PER_ZONE_SUMMARY_CSV,
        expected_cols=["zone_id", "avg_veh"]
    )
    global_results = read_csv_safely(
        GLOBAL_RESULTS_CSV,
        expected_cols=["frame", "perc_in_parking"]
    )

    # Ensure numeric types (important if safe reader returned strings)
    per_zone_per_frame["zone_id"] = per_zone_per_frame["zone_id"].astype(int)
    per_zone_per_frame["vehicles_in_zone"] = pd.to_numeric(per_zone_per_frame["vehicles_in_zone"], errors="coerce")

    global_results["perc_in_parking"] = pd.to_numeric(global_results["perc_in_parking"], errors="coerce")

    # Turn "0001.txt" into integer frame index
    for name, df in [("per_zone_per_frame", per_zone_per_frame), ("global_results", global_results)]:
        df["frame_idx"] = (
            df["frame"]
            .astype(str)
            .str.replace(".txt", "", regex=False)
            .astype(int)
        )
        if name == "per_zone_per_frame":
            per_zone_per_frame = df
        else:
            global_results = df

    return per_zone_per_frame, per_zone_summary, global_results


# =========================
# SMOOTHING HELPERS
# =========================
def add_rolling_mean(df, value_col, group_cols, order_col, windows):
    """
    Adds rolling-mean columns to df:
      value_col_rm{w} for each window w
    Rolling mean is computed within each group, ordered by order_col.
    """
    df = df.sort_values(group_cols + [order_col]).copy()

    for w in windows:
        rm_col = f"{value_col}_rm{w}"
        df[rm_col] = (
            df.groupby(group_cols, sort=False)[value_col]
              .transform(lambda s: s.rolling(window=w, min_periods=1).mean())
        )
    return df


# =========================
# PLOTS
# =========================
def plot_zone_occupancy_over_time(per_zone_per_frame, zone_id, windows=(5, 10, 20), overlay=False):
    """
    Plot vehicles_in_zone over time for a single zone, with rolling-mean smoothing.
    If overlay=True, plots all smoothing windows on the same figure.
    Otherwise, one figure per window.
    """
    df = per_zone_per_frame.copy()
    df["zone_id"] = df["zone_id"].astype(int)

    zone_df = df[df["zone_id"] == zone_id].sort_values("frame_idx")
    if zone_df.empty:
        print(f"No data found for zone_id = {zone_id}")
        return

    zone_df = add_rolling_mean(
        zone_df,
        value_col="vehicles_in_zone",
        group_cols=["zone_id"],
        order_col="frame_idx",
        windows=windows
    )

    if overlay:
        plt.figure()
        # optional: include raw series
        plt.plot(zone_df["frame_idx"], zone_df["vehicles_in_zone"], linewidth=1, alpha=0.4, label="raw")
        for w in windows:
            plt.plot(zone_df["frame_idx"], zone_df[f"vehicles_in_zone_rm{w}"], linewidth=2, label=f"RM{w}")
        plt.xlabel("Frame index")
        plt.ylabel("Vehicles in zone")
        plt.title(f"Zone {zone_id} occupancy over time (smoothed)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
    else:
        for w in windows:
            plt.figure()
            plt.plot(zone_df["frame_idx"], zone_df[f"vehicles_in_zone_rm{w}"], linewidth=2)
            plt.xlabel("Frame index")
            plt.ylabel("Vehicles in zone (rolling mean)")
            plt.title(f"Zone {zone_id} occupancy over time (RM{w})")
            plt.grid(True)
            plt.tight_layout()
            plt.show()


def plot_all_zones_occupancy(per_zone_per_frame, window=10):
    """
    Plot vehicles_in_zone over time for all zones on the same plot,
    using a rolling mean of length `window`.
    """
    df = per_zone_per_frame.copy()
    df["zone_id"] = df["zone_id"].astype(int)
    df = add_rolling_mean(
        df,
        value_col="vehicles_in_zone",
        group_cols=["zone_id"],
        order_col="frame_idx",
        windows=[window]
    )

    pivot = df.pivot(index="frame_idx", columns="zone_id", values=f"vehicles_in_zone_rm{window}")

    plt.figure()
    for zone_id in pivot.columns:
        plt.plot(pivot.index, pivot[zone_id], linewidth=1, label=f"Zone {zone_id}")

    plt.xlabel("Frame index")
    plt.ylabel(f"Vehicles in zone (RM{window})")
    plt.title(f"Zone occupancy over time (all zones, RM{window})")
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()


def plot_global_parking_percentage(global_results, windows=(5, 10, 20), overlay=False):
    df = global_results.sort_values("frame_idx").copy()

    # make sure numeric
    df["perc_in_parking"] = pd.to_numeric(df["perc_in_parking"], errors="coerce")

    # rolling means (no group needed)
    for w in windows:
        df[f"perc_in_parking_rm{w}"] = df["perc_in_parking"].rolling(window=w, min_periods=1).mean()

    if overlay:
        plt.figure()
        plt.plot(df["frame_idx"], df["perc_in_parking"], linewidth=1, alpha=0.4, label="raw")
        for w in windows:
            plt.plot(df["frame_idx"], df[f"perc_in_parking_rm{w}"], linewidth=2, label=f"RM{w}")
        plt.xlabel("Frame index")
        plt.ylabel("Vehicles in parking (%)")
        plt.title("Global percentage of vehicles in parking over time (smoothed)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
    else:
        for w in windows:
            plt.figure()
            plt.plot(df["frame_idx"], df[f"perc_in_parking_rm{w}"], linewidth=2)
            plt.xlabel("Frame index")
            plt.ylabel("Vehicles in parking (%) (rolling mean)")
            plt.title(f"Global percentage of vehicles in parking over time (RM{w})")
            plt.grid(True)
            plt.tight_layout()
            plt.show()


def plot_zone_summary_bar(per_zone_summary):
    df = per_zone_summary.copy()
    df["zone_id"] = df["zone_id"].astype(int)
    df["avg_veh"] = pd.to_numeric(df["avg_veh"], errors="coerce")
    df = df.sort_values("zone_id")

    plt.figure()
    plt.bar(df["zone_id"], df["avg_veh"])
    plt.xlabel("Zone ID")
    plt.ylabel("Average vehicles")
    plt.title("Average number of vehicles per zone")
    plt.tight_layout()
    plt.show()


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    per_zone_per_frame, per_zone_summary, global_results = load_data()

    print_estimated_vehicles(per_zone_summary)

    # 1) Zone plot: compare multiple smoothing windows on the same figure
    plot_zone_occupancy_over_time(per_zone_per_frame, zone_id=1, windows=(50, 150), overlay=True)

    # 2) All zones plot: pick ONE smoothing window (otherwise it gets unreadable)
    plot_all_zones_occupancy(per_zone_per_frame, window=10)

    # 3) Global plot: compare smoothing windows
    plot_global_parking_percentage(global_results, windows=(50, 150), overlay=True)

    # 4) Summary bar
    plot_zone_summary_bar(per_zone_summary)