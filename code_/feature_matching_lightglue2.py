# lg_minimal_img2_to_img1.py
import cv2, numpy as np, torch
from lightglue import LightGlue, SuperPoint, match_pair
from lightglue.utils import load_image

"""
Minimal LightGlue-based pipeline to estimate a homography (img2 → img1) and warp img2 onto img1.

The script extracts SuperPoint keypoints, matches them with LightGlue, and estimates a RANSAC
homography from img2 to img1. The warped result is produced at the original image resolution.

Outputs:
- *_warped.png: img2 warped into img1’s coordinate frame
- *_overlay.png: blended overlay for quick alignment check
- *_matches.png: visualization of inlier matches (LightGlue resolution)
"""


IMG1 = r"C:/Users/makss/Git/Galatsi-Semester-Project/first_frame_static_aligned.png"
IMG2 = r"C:/Users/makss/Git/Galatsi-Semester-Project/another_video.png"
OUT  = "lg_out"

device   = "cuda" if torch.cuda.is_available() else "cpu"
extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
matcher   = LightGlue(features='superpoint').eval().to(device)

# charge images pour LG (tensor [3,H,W] float32 0..1)
im0 = load_image(IMG2).to(device)   # img2
im1 = load_image(IMG1).to(device)   # img1

# extraction + matching (raccourci)
feats0, feats1, m01 = match_pair(extractor, matcher, im0, im1)
m = m01['matches'].cpu().numpy().astype(np.int64)   # (K,2) -> numpy pour l'indexation

kpts0 = feats0['keypoints'].cpu().numpy().astype(np.float32)
kpts1 = feats1['keypoints'].cpu().numpy().astype(np.float32)
pts2  = kpts0[m[:,0]]   # img2
pts1  = kpts1[m[:,1]]   # img1

# homographie (img2 -> img1)
H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 3.0)
assert H is not None, "Homography failed"
inl = mask.ravel().astype(bool)

# warp sur tailles originales
img1 = cv2.imread(IMG1); img2 = cv2.imread(IMG2)
h1, w1 = img1.shape[:2]
warped  = cv2.warpPerspective(img2, H, (w1, h1))
overlay = cv2.addWeighted(img1, 0.5, warped, 0.5, 0)

# === Sauvegarde des matches (inliers) ===
def to_bgr(t):
    t = t[0] if t.ndim == 4 else t      # (3,H,W)
    a = (t.permute(1,2,0).cpu().numpy()*255).astype(np.uint8)  # RGB
    return a[:, :, ::-1]                # -> BGR

v2, v1 = to_bgr(im0), to_bgr(im1)       # visus aux résolutions LightGlue
Hc, Wc = max(v1.shape[0], v2.shape[0]), v2.shape[1] + v1.shape[1]
canvas = np.zeros((Hc, Wc, 3), np.uint8)
canvas[:v2.shape[0], :v2.shape[1]] = v2
canvas[:v1.shape[0], v2.shape[1]:] = v1
offx = v2.shape[1]
for (x2,y2),(x1,y1),ok in zip(pts2, pts1, inl):
    if not ok: continue
    p2 = (int(x2), int(y2)); p1 = (int(x1)+offx, int(y1))
    cv2.circle(canvas, p2, 2, (0,255,0), -1)
    cv2.circle(canvas, p1, 2, (0,255,0), -1)
    cv2.line(canvas, p2, p1, (0,255,0), 1, cv2.LINE_AA)

# sorties identiques + le nouveau fichier
cv2.imwrite(f"{OUT}_warped.png",  warped)
cv2.imwrite(f"{OUT}_overlay.png", overlay)
cv2.imwrite(f"{OUT}_matches.png", canvas)
print("OK ->", f"{OUT}_warped.png", f"{OUT}_overlay.png", f"{OUT}_matches.png")
