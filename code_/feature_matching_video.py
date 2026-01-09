# video_homography_img2_to_img1.py
# -*- coding: utf-8 -*-

import cv2, numpy as np, os

"""
This script aligns two time-synchronized videos by estimating a frame-by-frame homography (VID2 → VID1).

For each processed frame (optionally skipping frames with PROCESS_EVERY), it detects features in both frames
(SIFT if available, otherwise ORB), matches them with a k-NN matcher + Lowe ratio test, and estimates a robust
homography using RANSAC (or MAGSAC if available). If RESIZE_WIDTH is set, homography is estimated on resized
frames for speed and then rescaled back to the original resolution.

The estimated homography is used to warp each frame of VID2 into VID1’s coordinate system, and an overlay video
is also produced for quick visual verification. If homography estimation fails for a frame, the script can reuse
the previous frame’s homography (FALLBACK_PREV_H) as a fallback.

Outputs:
- OUT_WARP: warped version of VID2 in VID1’s view
- OUT_OVER: blended overlay of VID1 and warped VID2
- OUT_MATCH (optional): video showing inlier matches used for each homography estimate
"""



# ========= Chemins =========
VID1 = r"C:/Users/makss/Git/Galatsi-Semester-Project/Dataset/Galatsi_Data_Semester_Project_processing/aligned_outputs\DJI_0004_aligned.mp4"   # vidéo RÉFÉRENCE (vue plus haute)
VID2 = r"C:/Users/makss/Git/Galatsi-Semester-Project/Dataset/Galatsi_Data_Semester_Project_processing/aligned_outputs\DJI_0808_aligned.mp4"   # vidéo à REPROJETER sur VID1

OUT_WARP   = "out_warped_img2_on_img1.mp4"
OUT_OVER   = "out_overlay.mp4"
OUT_MATCH  = "out_matches.mp4"  # ex: "out_matches.mp4" pour une vidéo des matches (sinon None)

# ========= Réglages =========
USE_SIFT_IF_AVAILABLE = True
LOWE_RATIO = 0.75
RANSAC_THRESH_PX = 3.0
USE_MAGSAC_IF_AVAILABLE = True
PROCESS_EVERY = 1          # traite 1 frame sur N (ex: 2 pour sauter 1/2)
RESIZE_WIDTH = None        # ex: 1280 pour accélérer; None = taille originale
FALLBACK_PREV_H = True     # si estimation H échoue, on réutilise H_{t-1}
DRAW_MATCHES_EVERY = 1     # écrire les matches à chaque frame si OUT_MATCH; sinon ignorer

# ========= Utils =========
def try_create_sift():
    try: return cv2.SIFT_create()
    except Exception: return None

def make_detector():
    sift = try_create_sift() if USE_SIFT_IF_AVAILABLE else None
    if sift is not None:
        return sift, True
    return cv2.ORB_create(nfeatures=4000), False

def match_knn(des_src, des_dst, use_sift):
    if use_sift:
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=100)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
    else:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = matcher.knnMatch(des_src, des_dst, k=2)
    good = []
    for pair in knn:
        if len(pair) < 2: 
            continue
        m, n = pair
        if m.distance < LOWE_RATIO * n.distance:
            good.append(m)
    return good

def compute_homography(kp_src, kp_dst, matches):
    if len(matches) < 4:
        return None, None, 0.0
    pts_src = np.float32([kp_src[m.queryIdx].pt for m in matches])
    pts_dst = np.float32([kp_dst[m.trainIdx].pt for m in matches])
    method = cv2.RANSAC
    if USE_MAGSAC_IF_AVAILABLE and hasattr(cv2, "USAC_MAGSAC"):
        method = cv2.USAC_MAGSAC
    H, mask = cv2.findHomography(pts_src, pts_dst, method,
                                 ransacReprojThreshold=RANSAC_THRESH_PX,
                                 maxIters=10000, confidence=0.999)
    inliers = mask.ravel().astype(bool) if mask is not None else np.zeros(len(matches), bool)
    # reprojection error (inliers)
    if H is None or not inliers.any():
        return H, inliers, np.inf
    src_inl = pts_src[inliers]
    dst_inl = pts_dst[inliers]
    src_h = np.hstack([src_inl, np.ones((len(src_inl),1), np.float32)])
    proj = (H @ src_h.T).T
    proj = proj[:, :2] / proj[:, 2:3]
    err = float(np.mean(np.linalg.norm(proj - dst_inl, axis=1)))
    return H, inliers, err

def maybe_resize(img, target_w):
    if target_w is None: 
        return img, 1.0
    h, w = img.shape[:2]
    if w == target_w:
        return img, 1.0
    scale = target_w / float(w)
    new_size = (int(w*scale), int(h*scale))
    return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA), scale

def draw_match_frame(img2, kp2, img1, kp1, matches, inliers):
    h = max(img1.shape[0], img2.shape[0])
    w = img1.shape[1] + img2.shape[1]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:img2.shape[0], :img2.shape[1]] = img2
    canvas[:img1.shape[0], img2.shape[1]:] = img1
    offx = img2.shape[1]
    for m, ok in zip(matches, inliers.tolist()):
        if not ok: 
            continue
        x2, y2 = kp2[m.queryIdx].pt
        x1, y1 = kp1[m.trainIdx].pt
        p2 = (int(x2), int(y2))
        p1 = (int(x1)+offx, int(y1))
        cv2.circle(canvas, p2, 2, (0,255,0), -1)
        cv2.circle(canvas, p1, 2, (0,255,0), -1)
        cv2.line(canvas, p2, p1, (0,255,0), 1, cv2.LINE_AA)
    return canvas

# ========= Main =========
cap1, cap2 = cv2.VideoCapture(VID1), cv2.VideoCapture(VID2)
if not cap1.isOpened() or not cap2.isOpened():
    raise SystemExit("Impossible d’ouvrir l’une des vidéos.")

fps = cap1.get(cv2.CAP_PROP_FPS) or 30.0
ok1, frame1 = cap1.read()
ok2, frame2 = cap2.read()
if not ok1 or not ok2:
    raise SystemExit("Impossible de lire la première frame des vidéos.")

# option: uniformiser la taille de travail
f1_small, s1 = maybe_resize(frame1, RESIZE_WIDTH)
f2_small, s2 = maybe_resize(frame2, RESIZE_WIDTH)

H_prev = None
detector, use_sift = make_detector()

# Writers (sorties à la taille de frame1 originale)
h1, w1 = frame1.shape[:2]
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
vw_warp = cv2.VideoWriter(OUT_WARP, fourcc, fps, (w1, h1))
vw_over = cv2.VideoWriter(OUT_OVER, fourcc, fps, (w1, h1))
vw_match = None

t = 0
while ok1 and ok2:
    if t % PROCESS_EVERY != 0:
        # écrire “copie” des frames si on saute (facultatif: ici on réécrit overlay simple)
        warped = frame2
        overlay = cv2.addWeighted(frame1, 0.5, warped, 0.5, 0)
        vw_warp.write(warped)
        vw_over.write(overlay)
        ok1, frame1 = cap1.read()
        ok2, frame2 = cap2.read()
        t += 1
        continue

    f1_small, s1 = maybe_resize(frame1, RESIZE_WIDTH)
    f2_small, s2 = maybe_resize(frame2, RESIZE_WIDTH)

    # 1) features
    kp1, des1 = detector.detectAndCompute(cv2.cvtColor(f1_small, cv2.COLOR_BGR2GRAY), None)
    kp2, des2 = detector.detectAndCompute(cv2.cvtColor(f2_small, cv2.COLOR_BGR2GRAY), None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        H_est = H_prev if (FALLBACK_PREV_H and H_prev is not None) else None
        if H_est is None:
            warped = frame2
        else:
            warped = cv2.warpPerspective(frame2, H_est, (w1, h1))
        overlay = cv2.addWeighted(frame1, 0.5, warped, 0.5, 0)
        vw_warp.write(warped)
        vw_over.write(overlay)
        ok1, frame1 = cap1.read(); ok2, frame2 = cap2.read(); t += 1
        continue

    # 2) matching (ATTENTION: img2 -> img1)
    good = match_knn(des2, des1, use_sift)

    # 3) Homographie sur les images redimensionnées
    H_small, inliers, err = compute_homography(kp2, kp1, good)

    # 4) Re-scale H_small → H sur la taille ORIGINALE
    H = None
    if H_small is not None:
        # H_small mappe f2_small -> f1_small ; remet à l’échelle vers frame2 -> frame1
        S2 = np.array([[1/s2, 0, 0], [0, 1/s2, 0], [0, 0, 1]], dtype=np.float32)   # small->orig (pré)
        S1 = np.array([[s1, 0, 0], [0, s1, 0], [0, 0, 1]], dtype=np.float32)      # orig->small (post inverse)
        # H_orig = (S1^-1) * H_small * S2
        S1_inv = np.array([[1/s1, 0, 0], [0, 1/s1, 0], [0, 0, 1]], dtype=np.float32)
        H = S1_inv @ H_small @ S2

    if H is None and FALLBACK_PREV_H and H_prev is not None:
        H = H_prev

    # 5) Warp + overlay (écriture vidéo)
    if H is None:
        warped = frame2
    else:
        warped = cv2.warpPerspective(frame2, H, (w1, h1))
        H_prev = H.copy()
    overlay = cv2.addWeighted(frame1, 0.5, warped, 0.5, 0)

    vw_warp.write(warped)
    vw_over.write(overlay)

    # 6) (optionnel) vidéo de matches en résolution de travail
    if OUT_MATCH and (t % DRAW_MATCHES_EVERY == 0):
        vis = draw_match_frame(f2_small, kp2, f1_small, kp1, good, inliers if inliers is not None else [])
        if vw_match is None:
            hh, ww = vis.shape[:2]
            vw_match = cv2.VideoWriter(OUT_MATCH, fourcc, fps, (ww, hh))
        vw_match.write(vis)

    # next
    ok1, frame1 = cap1.read()
    ok2, frame2 = cap2.read()
    t += 1

    print(f"Frames traitées: {t}")

# cleanup
vw_warp.release()
vw_over.release()
if vw_match is not None: vw_match.release()
cap1.release(); cap2.release()
print("Terminé →", OUT_WARP, "|", OUT_OVER, "|" , (OUT_MATCH or "pas de matches"))
