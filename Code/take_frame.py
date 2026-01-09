import cv2

"""
Extracts a specific frame from a video file and saves it as an image.
Here, it grabs the first frame (index 0) and writes it to disk for later use
(e.g., reference image, annotation, or alignment).
"""


# chemin de la vidéo
video_path = "Dataset/Galatsi_Data_Semester_Project_archive/DJI_0319_D2_S5_S1.MP4"
# chemin de sortie pour l'image
output_path = "frame_1_0319.png"

# ouvrir la vidéo
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Could not read frame")
    exit()

# aller directement à la 281ème frame (attention : index = 280 car ça commence à 0)
frame_index = 0
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

# lire la frame
ret, frame = cap.read()

if ret:
    cv2.imwrite(output_path, frame)
    print(f"Frame {frame_index+1} save as {output_path}")
else:
    print("Could not read frame")

cap.release()
