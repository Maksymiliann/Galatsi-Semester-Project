# lg_homog_img2_to_img1.py
# -*- coding: utf-8 -*-
# LightGlue + SuperPoint => Homography (img2 -> img1) + warp + PNG outputs

import cv2, numpy as np, torch
from pathlib import Path

"""
This script uses SuperPoint + LightGlue to match features between two images, estimate a homography
(img2 -> img1), and warp img2 into img1’s coordinate system.

It loads img2 as image0 and img1 as image1 (so matches are img2→img1), extracts keypoints with SuperPoint,
matches them with LightGlue, and fits a robust homography with RANSAC. If feature extraction was done on
resized images, the homography is rescaled back to the original image resolution.

Outputs:
- *_matches.png: visualization of inlier matches (on the resized images used by LightGlue)
- *_warped_img2_on_img1.png: img2 warped onto img1 (original resolution)
- *_overlay.png: blended overlay for quick visual inspection
It also prints the homography matrix, inlier ratio, and mean reprojection error (px).
"""


from lightglue import LightGlue, SuperPoint  # (ou DISK/ALIKED/SIFT)
from lightglue.utils import load_image, rbd  # rbd = remove batch dimension

# ========= Chemins (à adapter) =========
IMG1_PATH = "first_frame_static_aligned.png"   # img1 = RÉFÉRENCE (prise de plus haut)
IMG2_PATH = "first_frame_moving_aligned.png"    # img2 = À REPROJETER sur img1
OUT_PREFIX = "lg_img2_to_img1" # sorties PNG: matches/warped/overlay

# ========= Réglages =========
MAX_KPTS = 2048          # points max pour SuperPoint (perf/vitesse)
RESIZE_LONG_EDGE = None  # ex: 1600 pour auto-resize; sinon None (taille d’origine)
RANSAC_THRESH_PX = 3.0   # seuil RANSAC
CONFIDENCE = 0.999
MAX_ITERS = 10000

def draw_matches_concat(img2_bgr, img1_bgr, p2, p1, inlier_mask):
    """Petit visuel concaténé (img2 | img1) + segments entre points inliers."""
    h = max(img2_bgr.shape[0], img1_bgr.shape[0])
    w = img2_bgr.shape[1] + img1_bgr.shape[1]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:img2_bgr.shape[0], :img2_bgr.shape[1]] = img2_bgr
    canvas[:img1_bgr.shape[0], img2_bgr.shape[1]:] = img1_bgr
    offx = img2_bgr.shape[1]

    for (x2,y2),(x1,y1),ok in zip(p2, p1, inlier_mask):
        if not ok: 
            continue
        pt2 = (int(x2), int(y2))
        pt1 = (int(x1)+offx, int(y1))
        cv2.circle(canvas, pt2, 2, (0,255,0), -1)
        cv2.circle(canvas, pt1, 2, (0,255,0), -1)
        cv2.line(canvas, pt2, pt1, (0,255,0), 1, cv2.LINE_AA)
    return canvas

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- 1) Charger images pour LightGlue (tensor [3,H,W], float [0,1]) ---
    # IMPORTANT: on charge en ordre (image0 = img2, image1 = img1) pour avoir matches img2->img1
    image0 = load_image(IMG2_PATH, resize=RESIZE_LONG_EDGE).to(device)  # img2
    image1 = load_image(IMG1_PATH, resize=RESIZE_LONG_EDGE).to(device)  # img1

    # Garde aussi les versions BGR pour l’affichage/warp final (taille originale)
    img1_bgr = cv2.imread(IMG1_PATH, cv2.IMREAD_COLOR)
    img2_bgr = cv2.imread(IMG2_PATH, cv2.IMREAD_COLOR)
    if img1_bgr is None or img2_bgr is None:
        raise SystemExit("Impossible de lire IMG1_PATH/IMG2_PATH en BGR.")

    # --- 2) Extracteur local + LightGlue ---
    extractor = SuperPoint(max_num_keypoints=MAX_KPTS).eval().to(device)
    matcher   = LightGlue(features='superpoint').eval().to(device)

    with torch.inference_mode():
        feats0 = extractor.extract(image0)  # <-- garder la dimension batch !
        feats1 = extractor.extract(image1)
        matches01 = matcher({'image0': feats0, 'image1': feats1})

    # Maintenant seulement on enlève la dimension batch
    feats0, feats1, matches01 = [rbd(x) for x in (feats0, feats1, matches01)]

    # 3) Récupération des points (img2 -> img1)
    m = matches01['matches'].cpu().numpy().astype(np.int64)   # <-- CPU + NumPy
    kpts0 = feats0['keypoints'].cpu().numpy()     # (N0,2)
    kpts1 = feats1['keypoints'].cpu().numpy()     # (N1,2)
    pts2  = kpts0[m[:, 0]].astype(np.float32)     # img2
    pts1  = kpts1[m[:, 1]].astype(np.float32)     # img1

    # --- 4) Estimer l’homographie sur les points (déjà dans le repère éventuellement redimensionné) ---
    H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, RANSAC_THRESH_PX,
                                 maxIters=MAX_ITERS, confidence=CONFIDENCE)
    if H is None:
        raise SystemExit("Échec de l’estimation d’homographie.")

    inliers = mask.ravel().astype(bool)
    inlier_ratio = inliers.mean()

    # --- 5) Si on a redimensionné pour l’extraction, adapter H à la taille originale ---
    # load_image(resize=R) redimensionne l’image pour le modèle. Il faut donc
    # remettre H à l’échelle des images originales si RESIZE_LONG_EDGE est utilisé.
    def scale_from_to(src_shape, dst_shape):
        # src/dst shapes: (H,W)
        sh, sw = src_shape
        dh, dw = dst_shape
        S = np.eye(3, dtype=np.float32)
        S[0,0] = dw / sw
        S[1,1] = dh / sh
        return S

    if RESIZE_LONG_EDGE is not None:
        # tailles utilisées par LightGlue (tensors)
        H2, W2 = image0.shape[-2], image0.shape[-1]   # img2 resized
        H1, W1 = image1.shape[-2], image1.shape[-1]   # img1 resized
        S_img2_to_orig = scale_from_to((H2, W2), img2_bgr.shape[:2])
        S_img1_to_orig = scale_from_to((H1, W1), img1_bgr.shape[:2])
        # H acts from img2_resized -> img1_resized. Convert to original sizes:
        # H_orig = S1_orig * H * S2_orig^{-1}
        H = S_img1_to_orig @ H @ np.linalg.inv(S_img2_to_orig)

    # --- 6) Warp (img2 -> img1) en taille originale ---
    h1, w1 = img1_bgr.shape[:2]
    warped = cv2.warpPerspective(img2_bgr, H, (w1, h1))
    overlay = cv2.addWeighted(img1_bgr, 0.5, warped, 0.5, 0)

    # --- 7) Visu matches (inliers) sur images redimensionnées utilisées par LG ---
    # Pour un visuel rapide, on dessine sur les versions resized (plus petit).
    img2_viz = (image0.permute(1,2,0).cpu().numpy()*255).astype(np.uint8)[:, :, ::-1]  # RGB->BGR
    img1_viz = (image1.permute(1,2,0).cpu().numpy()*255).astype(np.uint8)[:, :, ::-1]
    vis = draw_matches_concat(img2_viz, img1_viz, pts2, pts1, inliers)

    # --- 8) Sauvegardes ---
    Path(".").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(f"{OUT_PREFIX}_matches.png", vis)
    cv2.imwrite(f"{OUT_PREFIX}_warped_img2_on_img1.png", warped)
    cv2.imwrite(f"{OUT_PREFIX}_overlay.png", overlay)

    # --- 9) Rapide métrique d’erreur (reprojection moyenne sur inliers) ---
    def reproj_err(H, p_src, p_dst):
        if H is None or len(p_src) == 0: return float('inf')
        ph = np.hstack([p_src, np.ones((len(p_src),1), np.float32)])
        proj = (H @ ph.T).T
        proj = proj[:, :2] / proj[:, 2:3]
        return float(np.mean(np.linalg.norm(proj - p_dst, axis=1)))

    # Pour la reprojection, on travaille en taille originale -> on doit retransformer pts2/pts1 si on avait resize
    if RESIZE_LONG_EDGE is not None:
        # remet pts des resized vers tailles originales pour la métrique
        pts2_orig = cv2.perspectiveTransform(pts2.reshape(-1,1,2),
                         np.linalg.inv(scale_from_to((image0.shape[-2], image0.shape[-1]), img2_bgr.shape[:2]))).reshape(-1,2)
        pts1_orig = cv2.perspectiveTransform(pts1.reshape(-1,1,2),
                         np.linalg.inv(scale_from_to((image1.shape[-2], image1.shape[-1]), img1_bgr.shape[:2]))).reshape(-1,2)
    else:
        pts2_orig, pts1_orig = pts2, pts1

    err_px = reproj_err(H, pts2_orig[inliers], pts1_orig[inliers])

    print("\n=== LightGlue Homography (img2 -> img1) ===")
    np.set_printoptions(precision=6, suppress=True)
    print(H)
    print(f"Matches: {len(pts1)} | Inliers: {int(inliers.sum())} ({inlier_ratio:.2%}) | Reproj err: {err_px:.3f}px")
    print("Écrits: "
          f"{OUT_PREFIX}_matches.png, "
          f"{OUT_PREFIX}_warped_img2_on_img1.png, "
          f"{OUT_PREFIX}_overlay.png")

if __name__ == "__main__":
    main()
