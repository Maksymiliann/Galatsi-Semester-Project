
import cv2
import numpy as np
import os

# ========= Chemins (à adapter) =========
IMG1_PATH = r"C:/Users/makss/Git/Galatsi-Semester-Project/first_frame_static_aligned.png"   # img1 = RÉFÉRENCE (prise de plus haut)
IMG2_PATH = r"C:/Users/makss/Git/Galatsi-Semester-Project/first_frame_moving_aligned.png"   # img2 = À REPROJETER sur img1
OUT_PREFIX = "out_img2_to_img1"

# ========= Réglages =========
USE_SIFT_IF_AVAILABLE = True
LOWE_RATIO = 0.75          # 0.70–0.80 typique
RANSAC_THRESH_PX = 3.0     # seuil reprojection (px)
USE_MAGSAC_IF_AVAILABLE = True
APPLY_CLAHE = False        # petit boost de contraste si True

def try_create_sift():
    try: return cv2.SIFT_create()
    except Exception: return None

def boost_contrast_clahe(bgr):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB); l,a,b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l2 = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l2,a,b]), cv2.COLOR_LAB2BGR)

def reproj_error(H, src_inl, dst_inl):
    if H is None or len(src_inl)==0: return float("inf")
    src_h = np.hstack([src_inl, np.ones((len(src_inl),1), np.float32)])
    proj = (H @ src_h.T).T; proj = proj[:, :2] / proj[:, 2:3]
    return float(np.mean(np.linalg.norm(proj - dst_inl, axis=1)))

def main():
    img1 = cv2.imread(IMG1_PATH, cv2.IMREAD_COLOR)
    img2 = cv2.imread(IMG2_PATH, cv2.IMREAD_COLOR)
    if img1 is None or img2 is None:
        raise SystemExit("Chemin invalide. Vérifie IMG1_PATH/IMG2_PATH.")
    if APPLY_CLAHE:
        img1, img2 = boost_contrast_clahe(img1), boost_contrast_clahe(img2)

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 1) Détection/description
    detector, use_sift = None, False
    if USE_SIFT_IF_AVAILABLE:
        sift = try_create_sift()
        if sift is not None: detector, use_sift = sift, True
    if detector is None:
        detector = cv2.ORB_create(nfeatures=5000)

    k1, d1 = detector.detectAndCompute(gray1, None)
    k2, d2 = detector.detectAndCompute(gray2, None)
    if d1 is None or d2 is None or len(k1)<4 or len(k2)<4:
        raise SystemExit("Pas assez de points clés.")

    # 2) Matching (ATTENTION à l'ordre: img2 -> img1)
    if use_sift:
        FLANN_INDEX_KDTREE = 1
        matcher = cv2.FlannBasedMatcher(dict(algorithm=FLANN_INDEX_KDTREE, trees=5),
                                        dict(checks=200))
    else:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    knn = matcher.knnMatch(d2, d1, k=2)  # on compare des2 (img2) avec des1 (img1)
    good = []
    for p in knn:
        if len(p)<2: continue
        m,n = p
        if m.distance < LOWE_RATIO * n.distance:
            good.append(m)
    if len(good) < 4:
        raise SystemExit(f"Pas assez de matches après ratio test ({len(good)}).")

    pts2 = np.float32([k2[m.queryIdx].pt for m in good])  # src (img2)
    pts1 = np.float32([k1[m.trainIdx].pt for m in good])  # dst (img1)

    # 3) Homographie (img2 -> img1)
    method = cv2.RANSAC
    if USE_MAGSAC_IF_AVAILABLE and hasattr(cv2, 'USAC_MAGSAC'):
        method = cv2.USAC_MAGSAC
    H, mask = cv2.findHomography(pts2, pts1, method,
                                 ransacReprojThreshold=RANSAC_THRESH_PX,
                                 maxIters=10000, confidence=0.999)
    if H is None:
        raise SystemExit("Échec de l’estimation d’homographie.")
    inl = mask.ravel().astype(bool)
    err_px = reproj_error(H, pts2[inl], pts1[inl])

    # 4) Warp img2 -> plan img1
    h1, w1 = img1.shape[:2]
    warped = cv2.warpPerspective(img2, H, (w1, h1))
    overlay = cv2.addWeighted(img1, 0.5, warped, 0.5, 0)

    # 5) Visu + sauvegarde
    matchesMask = [int(x) for x in inl.tolist()]
    vis = cv2.drawMatches(img2, k2, img1, k1, good, None,
                          matchColor=(0,255,0), singlePointColor=(255,0,0),
                          matchesMask=matchesMask,
                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    base = os.path.splitext(os.path.basename(OUT_PREFIX))[0]
    cv2.imwrite(f"{base}_matches.png", vis)
    cv2.imwrite(f"{base}_warped_img2_on_img1.png", warped)
    cv2.imwrite(f"{base}_overlay.png", overlay)

    np.set_printoptions(precision=6, suppress=True)
    print("\n=== Homography (img2 -> img1) ===")
    print(H)
    print(f"Matches: {len(good)} | Inliers: {int(inl.sum())} ({inl.mean():.2%}) | Reproj err: {err_px:.3f}px")
    print("Écrits: *_matches.png, *_warped_img2_on_img1.png, *_overlay.png")

if __name__ == "__main__":
    main()
