import numpy as np
from PIL import Image

def img2arr(path):
    img = Image.open(path).convert('RGB')
    out = np.array(img, dtype=np.float64)
    return out

def find_matching_points(im1, im2):

    im1_pts = []
    im2_pts = []

    return im1_pts, im2_pts

def computeH(im1_pts, im2_pts):

    # A h = b
    
    pts1 = np.asarray(im1_pts, dtype=np.float64)
    pts2 = np.asarray(im2_pts, dtype=np.float64)
    n = pts1.shape[0]

    A = np.zeros((2 * n, 8))
    b = np.zeros(2 * n)

    for i in range(n):
        x, y = pts1[i]
        xp, yp = pts2[i]
        A[2 * i] = [x, y, 1, 0, 0, 0, -x * xp, -y * xp]
        A[2 * i + 1] = [0, 0, 0, x, y, 1, -x * yp, -y * yp]
        b[2 * i]  = xp
        b[2 * i + 1] = yp

    h, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    h_full = np.append(h, 1.0)
    H = h_full.reshape(3, 3)

    return H

def warpImageNearestNeighbor(im, H):
    height = im.shape[0]
    width = im.shape[1]
    box_finder = np.array([[0 , 0, 1], [0, height, 1], [width, 0, 1], [width, height, 1]])
    box = box_finder @ H.T
    box = box / box[:, 2:3]

    box_hmin = int(np.floor(box[:, 1].min()))
    box_wmin = int(np.floor(box[:, 0].min()))
    box_hmax = int(np.ceil(box[:, 1].max()))
    box_wmax = int(np.ceil(box[:, 0].max()))

    box_h = box_hmax - box_hmin
    box_w = box_wmax - box_wmin

    im_out = np.zeros((box_h, box_w) + im.shape[2:])
    alpha = np.zeros((box_h, box_w))

    inv_H = np.linalg.inv(H)

    X, Y = np.meshgrid(np.arange(box_w), np.arange(box_h))

    pts = np.stack([X.ravel() + box_wmin, Y.ravel() + box_hmin, np.ones(X.size)])

    pts_inv = inv_H @ pts
    pts_inv = pts_inv / pts_inv[2]

    og_x = np.round(pts_inv[0]).astype(int)
    og_y = np.round(pts_inv[1]).astype(int)

    valid = (og_x >= 0) & (og_x < width) & (og_y >= 0) & (og_y < height)

    im_out[Y.ravel()[valid], X.ravel()[valid]] = im[og_y[valid], og_x[valid]]
    alpha[Y.ravel()[valid], X.ravel()[valid]] = 1

    return im_out, alpha, box_wmin, box_hmin

def warpImageBilinear(im, H):
    height = im.shape[0]
    width = im.shape[1]
    box_finder = np.array([[0 , 0, 1], [0, height, 1], [width, 0, 1], [width, height, 1]])
    box = box_finder @ H.T
    box = box / box[:, 2:3]

    box_hmin = int(np.floor(box[:, 1].min()))
    box_wmin = int(np.floor(box[:, 0].min()))
    box_hmax = int(np.ceil(box[:, 1].max()))
    box_wmax = int(np.ceil(box[:, 0].max()))

    box_h = box_hmax - box_hmin
    box_w = box_wmax - box_wmin

    im_out = np.zeros((box_h, box_w) + im.shape[2:])
    alpha = np.zeros((box_h, box_w))

    inv_H = np.linalg.inv(H)

    X, Y = np.meshgrid(np.arange(box_w), np.arange(box_h))

    pts = np.stack([X.ravel() + box_wmin, Y.ravel() + box_hmin, np.ones(X.size)])

    pts_inv = inv_H @ pts
    pts_inv = pts_inv / pts_inv[2]

    urog_x = pts_inv[0]
    urog_y = pts_inv[1]

    x0 = np.floor(urog_x)
    y0 = np.floor(urog_y)
    dx = urog_x - x0
    dy = urog_y - y0
    x0 = x0.astype(int)
    y0 = y0.astype(int)
    x1 = x0 + 1
    y1 = y0 + 1

    valid = (x0 >= 0) & (x1 < width) & (y0 >= 0) & (y1 < height)

    im_out[Y.ravel()[valid], X.ravel()[valid]] = ((1 - dx[valid])[:, None] * (1 - dy[valid])[:, None] * im[y0[valid], x0[valid]] +dx[valid][:, None] * (1 - dy[valid])[:, None] * im[y0[valid], x1[valid]] + (1 - dx[valid])[:, None] * dy[valid][:, None]* im[y1[valid], x0[valid]] + dx[valid][:, None]* dy[valid][:, None] * im[y1[valid], x1[valid]])
    alpha[Y.ravel()[valid], X.ravel()[valid]] = 1

    return im_out, alpha, box_wmin, box_hmin

def feather(h, w):
    ry = np.minimum(np.arange(h), np.arange(h)[::-1]) + 1.0
    rx = np.minimum(np.arange(w), np.arange(w)[::-1]) + 1.0
    return ry[:, None] * rx[None, :]

def make_mosaic(im1, im2, im1_pts, im2_pts):
    H = computeH(im1_pts, im2_pts)

    w1, _, bmi1, bma1 = warpImageBilinear(im1, H)
    a1, _, _, _ = warpImageBilinear(feather(*im1.shape[:2])[..., None], H)
    a1 = a1[..., 0]
    w2 = im2
    a2 = feather(*im2.shape[:2])
    bmi2 = 0
    bma2 = 0

    minw = min(bmi1, bmi2)
    minh = min(bma1, bma2)
    maxw = max(bmi1 + w1.shape[1], bmi2 + w2.shape[1])
    maxh = max(bma1 + w1.shape[0], bma2 + w2.shape[0])

    c1 = np.zeros((maxh - minh, maxw - minw, 3))
    c2 = np.zeros_like(c1)
    wt1 = np.zeros(c1.shape[:2])
    wt2 = np.zeros(c1.shape[:2])

    r1, q1 = bma1 - minh, bmi1 - minw
    r2, q2 = bma2 - minh, bmi2 - minw

    c1[r1:r1 + w1.shape[0], q1:q1 + w1.shape[1]] = w1
    wt1[r1:r1 + w1.shape[0], q1:q1 + w1.shape[1]] = a1
    c2[r2:r2 + w2.shape[0], q2:q2 + w2.shape[1]] = w2
    wt2[r2:r2 + w2.shape[0], q2:q2 + w2.shape[1]] = a2

    wsum = wt1 + wt2
    mosaic = (wt1[..., None] * c1 + wt2[..., None] * c2) / np.maximum(wsum, 1e-8)[..., None]

    return mosaic

def rectify(im, im_pts, out_w, out_h, warp=warpImageBilinear):
    ref_pts = [[0, 0], [out_w, 0], [0, out_h], [out_w, out_h]]
    p = np.asarray(im_pts, dtype=np.float64)
    cx0, cy0 = np.maximum(np.floor(p.min(axis=0)).astype(int) - 2, 0)
    cx1, cy1 = np.ceil(p.max(axis=0)).astype(int) + 3
    im = im[cy0:cy1, cx0:cx1]
    H = computeH(p - [cx0, cy0], ref_pts)
    im_out, alpha, xmin, ymin = warp(im, H)
    return im_out[-ymin:-ymin + out_h, -xmin:-xmin + out_w], H

def save_mosaic(mosaic, path):
    Image.fromarray(np.clip(mosaic, 0, 255).astype(np.uint8)).save(path)