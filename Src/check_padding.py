import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

# -------------------------
# PATH TO YOUR TXT FOLDER
# -------------------------
folder_path = r"C:/Users/makss/Git/Galatsi-Semester-Project/Results/TXT_0314_D2_S4_S1_padded_state_corrected"

# -------------------------
# CHOOSE VEHICLE IDS HERE
# -------------------------
selected_ids = [57, 142, 212, 220, 222, 233, 238, 293, 301, 338, 350, 438, 442, 592, 5451, 10722, 11930]   #

# -------------------------
# GET TXT FILES
# -------------------------
files = sorted([f for f in os.listdir(folder_path) if f.endswith(".txt")])

# -------------------------
# BUILD PRESENCE MATRIX
# -------------------------
presence = {vid: [] for vid in selected_ids}

for file in files:

    path = os.path.join(folder_path, file)

    try:
        df = pd.read_csv(path, sep=";")
        ids_in_frame = set(df["vehicle_id"].values)
    except:
        ids_in_frame = set()

    for vid in selected_ids:
        presence[vid].append(1 if vid in ids_in_frame else 0)

# -------------------------
# PLOT
# -------------------------
plt.figure(figsize=(14,6))

frames = np.arange(len(files))

for vid in selected_ids:
    plt.plot(frames, presence[vid], marker='o', label=f"ID {vid}")

plt.xlabel("Frame")
plt.ylabel("Presence (1 = present, 0 = not present)")
plt.title("Vehicle Presence Over Frames")

plt.yticks([0,1])
plt.legend()
plt.grid(True)

plt.show()