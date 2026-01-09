import cv2, numpy as np

"""
This script aligns img2 to img1 by estimating a SIFT-based homography (img2 → img1) and reporting alignment quality.

It detects SIFT keypoints/descriptors in both images, matches descriptors from img2 to img1 using k-NN matching
with Lowe’s ratio test, and estimates a robust homography with RANSAC. Inlier matches are used to compute the
mean reprojection error (in pixels) as a quantitative alignment metric. Finally, img2 is warped into img1’s
coordinate frame and the warped image + a blended overlay are saved to disk.
"""


img1 = cv2.imread("first_frame_static_aligned.png")
img2 = cv2.imread("first_frame_moving_aligned.png")

sift = cv2.SIFT_create()
k1, d1 = sift.detectAndCompute(img1, None)
k2, d2 = sift.detectAndCompute(img2, None)

bf = cv2.BFMatcher(cv2.NORM_L2)
knn = bf.knnMatch(d2, d1, k=2)                 # img2 -> img1
good = [m for m,n in knn if m.distance < 0.75*n.distance]
if len(good) < 4: raise SystemExit("Trop peu de matches.")

pts2 = np.float32([k2[m.queryIdx].pt for m in good]).reshape(-1,1,2)
pts1 = np.float32([k1[m.trainIdx].pt for m in good]).reshape(-1,1,2)

H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 3.0)
if H is None: raise SystemExit("Homography failed.")
inl = mask.ravel().astype(bool)
err = np.inf
if inl.any():
    p2 = pts2[inl].reshape(-1,2); p1 = pts1[inl].reshape(-1,2)
    p2h = np.hstack([p2, np.ones((len(p2),1), np.float32)])
    proj = (H @ p2h.T).T; proj = proj[:,:2]/proj[:,2:3]
    err = np.mean(np.linalg.norm(proj - p1, axis=1))

h1, w1 = img1.shape[:2]
warped = cv2.warpPerspective(img2, H, (w1, h1))
overlay = cv2.addWeighted(img1, 0.5, warped, 0.5, 0)

print(f"Inliers: {inl.sum()}/{len(good)}  |  Reproj err: {err:.3f}px")
cv2.imwrite("warped_img2_on_img1.png", warped)
cv2.imwrite("overlay.png", overlay)
