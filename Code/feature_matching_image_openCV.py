import cv2
import numpy as np

# --- 1. Charger les images ---
img1 = cv2.imread("first_frame_static_aligned.png")   # référence (vue plus haute)
img2 = cv2.imread("first_frame_moving_aligned.png")    # à reprojeter sur img1

# --- 2. Détection des points et descripteurs (SIFT) ---
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# --- 3. Matching (BFMatcher + ratio test) ---
bf = cv2.BFMatcher(cv2.NORM_L2)
matches = bf.knnMatch(des2, des1, k=2)   # ordre = img2 -> img1

good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append(m)

# --- 4. Calcul de l’homographie (img2 -> img1) ---
pts2 = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
pts1 = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)

# --- 5. Reprojection (warp img2 sur img1) ---
h1, w1 = img1.shape[:2]
img2_warped = cv2.warpPerspective(img2, H, (w1, h1))

# --- 6. Blend simple ---
overlay = cv2.addWeighted(img1, 0.5, img2_warped, 0.5, 0)

# --- 7. Affichage / sauvegarde ---
cv2.imwrite("Warped_img2_on_img1.png", img2_warped)
cv2.imshow("Overlay", overlay)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("matches_result.png", overlay)
