import pandas as pd
import matplotlib.pyplot as plt

# =========================
# CONFIG – CHANGE PATHS HERE
# =========================
PER_ZONE_PER_FRAME_CSV = r"C:/Users/makss/Git/Galatsi-Semester-Project/Results/occupancy_analysis/per_zones/test1/per_zone_per_frame.csv"
PER_ZONE_SUMMARY_CSV   = r"C:/Users/makss/Git/Galatsi-Semester-Project/Results/occupancy_analysis/per_zones/test1/per_zone_summary.csv"
GLOBAL_RESULTS_CSV     = r"C:/Users/makss/Git/Galatsi-Semester-Project/Results/occupancy_analysis/per_zones/test1/global_results.csv"


# =========================
# SAFE CSV READER
# =========================
def read_csv_safely(path, expected_cols):
    """
    Lit un CSV et, si toutes les données sont dans une seule colonne,
    les re-sépare sur la virgule.
    """
    df = pd.read_csv(path)

    # Cas normal : on a déjà les bonnes colonnes
    if set(expected_cols).issubset(df.columns):
        return df

    # Cas où tout est dans une seule colonne (souvent à cause d'Excel)
    if df.shape[1] == 1:
        col0 = df.columns[0]
        # On split le contenu de cette colonne
        split = df[col0].astype(str).str.split(",", expand=True)

        # Première ligne = header
        split.columns = split.iloc[0]
        split = split[1:].reset_index(drop=True)

        return split

    raise ValueError(f"Impossible de trouver les colonnes {expected_cols} dans {path}. Colonnes trouvées: {df.columns}")


# =========================
# LOADING + PREPROCESSING
# =========================
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

    # Turn "0001.txt" into an integer frame index: 1, 2, 3, ...
    for name, df in [("per_zone_per_frame", per_zone_per_frame), ("global_results", global_results)]:
        df["frame_idx"] = (
            df["frame"]
            .str.replace(".txt", "", regex=False)
            .astype(int)
        )
        if name == "per_zone_per_frame":
            per_zone_per_frame = df
        else:
            global_results = df

    return per_zone_per_frame, per_zone_summary, global_results


# =========================
# PLOTS
# =========================

def plot_zone_occupancy_over_time(per_zone_per_frame, zone_id):
    """
    Plot vehicles_in_zone over time for a single zone.
    """
    per_zone_per_frame["zone_id"] = per_zone_per_frame["zone_id"].astype(int)
    zone_df = (
        per_zone_per_frame[per_zone_per_frame["zone_id"] == zone_id]
        .sort_values("frame_idx")
    )

    if zone_df.empty:
        print(f"No data found for zone_id = {zone_id}")
        return

    plt.figure()
    plt.plot(zone_df["frame_idx"], zone_df["vehicles_in_zone"], marker="o", linewidth=1)
    plt.xlabel("Frame index")
    plt.ylabel("Vehicles in zone")
    plt.title(f"Zone {zone_id} occupancy over time")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_all_zones_occupancy(per_zone_per_frame):
    """
    Plot vehicles_in_zone over time for all zones on the same plot.
    """
    df = per_zone_per_frame.copy()
    df["zone_id"] = df["zone_id"].astype(int)
    df = df.sort_values(["frame_idx", "zone_id"])

    # Pivot to have one column per zone
    pivot = df.pivot(index="frame_idx", columns="zone_id", values="vehicles_in_zone")

    plt.figure()
    for zone_id in pivot.columns:
        plt.plot(pivot.index, pivot[zone_id], linewidth=1, label=f"Zone {zone_id}")

    plt.xlabel("Frame index")
    plt.ylabel("Vehicles in zone")
    plt.title("Zone occupancy over time (all zones)")
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()


def plot_global_parking_percentage(global_results):
    """
    Plot global percentage of vehicles in parking over time.
    """
    df = global_results.sort_values("frame_idx")

    plt.figure()
    plt.plot(df["frame_idx"], df["perc_in_parking"], linewidth=1)
    plt.xlabel("Frame index")
    plt.ylabel("Vehicles in parking (%)")
    plt.title("Global percentage of vehicles in parking over time")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_zone_summary_bar(per_zone_summary):
    """
    Bar plot of average vehicles per zone.
    """
    df = per_zone_summary.copy()
    df["zone_id"] = df["zone_id"].astype(int)
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

    # 1) Plot occupancy over time for a specific zone
    plot_zone_occupancy_over_time(per_zone_per_frame, zone_id=1)

    # 2) Plot occupancy over time for all zones
    plot_all_zones_occupancy(per_zone_per_frame)

    # 3) Plot global percentage of vehicles in parking
    plot_global_parking_percentage(global_results)

    # 4) Plot average vehicles per zone (from summary)
    plot_zone_summary_bar(per_zone_summary)
