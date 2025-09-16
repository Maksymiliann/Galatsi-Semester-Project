import cv2

# Path to your video
video_path = "input.mp4"

# Open the video
cap = cv2.VideoCapture("Dataset/Galatsi_Data_Semester_Project_archive/DJI_0808.MOV")

# Read the first frame
ret, frame = cap.read()

if ret:
    # Save the first frame as an image
    cv2.imwrite("first_frame_moving.png", frame)
    print("First frame saved as first_frame_moving.png")
else:
    print("Could not read frame")

cap.release()