import numpy as np
from PIL import Image
from harris import get_harris_corners, dist2
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from parta import computeH
import os, shutil
from parta import make_mosaic, save_mosaic

def img2arr(path):
    img = Image.open(path).convert('L')
    if img.width > 1000:
        scale = 1000 / img.width
        img = img.resize((1000, int(img.height * scale)), Image.LANCZOS)
    out = np.array(img, dtype=np.float64)
    return out / 255.0

def load_rgb(path, max_w=1000):
    img = Image.open(path).convert('RGB')
    if img.width > max_w:
        s = max_w / img.width
        img = img.resize((max_w, int(img.height * s)), Image.LANCZOS)
    return np.array(img, dtype=np.float64)

def anms(h, coords, N):
    strenghts = h[coords[0], coords[1]]

    pts = coords.T

    dist = dist2(pts, pts)

    stronger = strenghts[:, None] < 0.9 * strenghts[None, :]

    dist = np.where(stronger, dist, np.inf)

    radii = dist.min(axis=1)

    idx = np.argsort(-radii)[:N] 

    return coords[:, idx]

def blur(img):
    blur = gaussian_filter(img, sigma=2)
    return blur

def patch(img, coords):  
    out = []
    for corner in zip(coords[0], coords[1]):
        y = corner[0]
        x = corner[1]
        bp = img[y-20:y+20, x-20:x+20]
        bpd = bp[::5, ::5]
        bpd = (bpd - bpd.mean()) / (bpd.std() + 1e-8)
        out.append(bpd.flatten())
    return np.array(out)

def extract_feature_desc(img):
    h, coords = get_harris_corners(img)
    anms_coords = anms(h, coords, 500)
    blurred = blur(img)
    desc = patch(blurred, anms_coords)
    return desc, anms_coords

def feature_match(desc1, desc2, threshold=0.7):
    d = dist2(desc1, desc2)
    d = np.sqrt(np.maximum(d, 0))
    ordered = np.sort(d, axis=1)
    best = np.argmin(d, axis=1)
    nn1 = ordered[:, 0]
    nn2 = ordered[:, 1]
    keep_mask = nn1 / nn2 < threshold
    stack1 = np.where(keep_mask)[0]
    stack2 = best[stack1]
    return np.stack([stack1, stack2], axis=1)

def ransac(gray1, gray2, iterations=2000):
    desc1, c1 = extract_feature_desc(gray1)
    desc2, c2 = extract_feature_desc(gray2)
    matches = feature_match(desc1, desc2)
    m = len(matches)
    i, j = matches[:, 0], matches[:, 1]
    pts1 = np.stack([c1[1][i], c1[0][i]], axis=1)
    pts2 = np.stack([c2[1][j], c2[0][j]], axis=1)

    best_sum = 0
    best_H = None
    best_mask = None

    for hi in range(iterations):
        idx = np.random.choice(m, 4, replace=False)
        H = computeH(pts1[idx], pts2[idx])
        ones = np.ones((pts1.shape[0], 1))
        p1h = np.hstack([pts1, ones])
        p1h = p1h @ H.T
        p1h = p1h[:, :2] / p1h[:, 2:3]
        error = np.linalg.norm(p1h - pts2, axis=1)
        inside = error < 3
        insum = inside.sum()
        if hi == 0:
            best_sum = insum
            best_H = H
            best_mask = inside
        else:
            if best_sum < insum:
                best_sum = insum
                best_H = H
                best_mask = inside
    H = computeH(pts1[best_mask], pts2[best_mask])
    return H, best_mask, pts1, pts2

def show_corners(gray, pts, path):
    plt.figure(figsize=(10, 7)); plt.imshow(gray, cmap='gray')
    plt.scatter(pts[1], pts[0], s=4, c='r'); plt.axis('off')
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()

def show_descriptors(blurred, coords, descs, path, k=8):
    fig, ax = plt.subplots(2, k, figsize=(2*k, 4))
    for n in range(k):
        y, x = coords[0][n], coords[1][n]
        ax[0, n].imshow(blurred[y-20:y+20, x-20:x+20], cmap='gray')
        ax[1, n].imshow(descs[n].reshape(8, 8), cmap='gray')
        ax[0, n].axis('off'); ax[1, n].axis('off')
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()

def show_matches(im1, im2, pts1, pts2, path):  # pts are (m, 2) as (x, y)
    H = max(im1.shape[0], im2.shape[0])
    canvas = np.zeros((H, im1.shape[1] + im2.shape[1]))
    canvas[:im1.shape[0], :im1.shape[1]] = im1
    canvas[:im2.shape[0], im1.shape[1]:] = im2
    off = im1.shape[1]
    plt.figure(figsize=(14, 7)); plt.imshow(canvas, cmap='gray')
    for (x1, y1), (x2, y2) in zip(pts1, pts2):
        plt.plot([x1, x2 + off], [y1, y2], lw=0.7)
    plt.axis('off'); plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()

def show_matched_patches(g1, g2, pts1, pts2, path, k=6):
    b1, b2 = blur(g1), blur(g2)
    sel = np.linspace(0, len(pts1) - 1, k).astype(int)  
    fig, ax = plt.subplots(2, k, figsize=(2*k, 4))
    for n, s in enumerate(sel):
        (x1, y1), (x2, y2) = pts1[s], pts2[s]
        ax[0, n].imshow(b1[y1-20:y1+20, x1-20:x1+20], cmap='gray')
        ax[1, n].imshow(b2[y2-20:y2+20, x2-20:x2+20], cmap='gray')
        ax[0, n].axis('off'); ax[1, n].axis('off')
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()

if __name__ == "__main__":
    g = img2arr('4/raw/sn_1.jpg')
    h, coords = get_harris_corners(g)
    desc, anms_c = extract_feature_desc(g)
    os.makedirs('4/img/b', exist_ok=True)
    show_corners(g, coords, '4/img/b/harris_sn_1.jpg')
    show_corners(g, anms_c, '4/img/b/anms_sn_1.jpg')
    show_descriptors(blur(g), anms_c, desc, '4/img/b/desc_sn_1.jpg')

    pairs = {'gw': ('gw_1', 'gw_2'), 'm': ('m_1', 'm_2'), 'sn': ('sn_1', 'sn_2')}
    for tag, (a, b) in pairs.items():
        g1, g2 = img2arr(f'4/raw/{a}.jpg'), img2arr(f'4/raw/{b}.jpg')
        H, mask, p1, p2 = ransac(g1, g2)
        print(tag, 'inliers:', mask.sum(), 'of', len(mask))

        show_matches(g1, g2, p1[mask], p2[mask], f'4/img/b/matches_{tag}.jpg')
        show_matched_patches(g1, g2, p1[mask], p2[mask], f'4/img/b/patches_{tag}.jpg')

        im1, im2 = load_rgb(f'4/raw/{a}.jpg'), load_rgb(f'4/raw/{b}.jpg')
        mosaic = make_mosaic(im1, im2, p1[mask], p2[mask])
        save_mosaic(mosaic, f'4/img/b/auto_{tag}.jpg')

        shutil.copy(f'4/results/mosaic_{a}_{b}.jpg', f'4/img/b/manual_{tag}.jpg')

gray1 = img2arr('4/img/a/sn_1.jpg')
gray2 = img2arr('4/img/a/sn_2.jpg')