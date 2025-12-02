import cv2
import numpy as np

# ---------- PARAMETRES ----------
MIN_AREA = 10000   # <-- aire minimum en pixels (à ajuster)
INPUT_PATH  = r"C:/Users/makss/Git/Galatsi-Semester-Project/Results/occupancy_analysis/per_zones/mask_closed.png"
OUTPUT_PATH = r"C:/Users/makss/Git/Galatsi-Semester-Project/Results/occupancy_analysis/per_zones/mask_closed_cleaned.png"

# ---------- LECTURE + BINARISATION ----------
# Lis en niveau de gris
img = cv2.imread(INPUT_PATH, cv2.IMREAD_GRAYSCALE)

# Si ton image est déjà bien binaire 0/255, cette étape ne change rien
_, bw = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# ---------- COMPOSANTES CONNEXES ----------
# labels : pour chaque pixel, l'indice de la composante
# stats  : contient entre autres l'aire de chaque composante
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
    bw, connectivity=8
)

# ---------- FILTRAGE PAR AIRE ----------
# On part d'une image noire
filtered = np.zeros_like(bw)

for i in range(1, num_labels):  # 0 = fond, on commence à 1
    area = stats[i, cv2.CC_STAT_AREA]
    if area >= MIN_AREA:
        filtered[labels == i] = 255  # on garde cette tache

# ---------- SAUVEGARDE ----------
cv2.imwrite(OUTPUT_PATH, filtered)
print("Image filtrée sauvegardée dans", OUTPUT_PATH)
