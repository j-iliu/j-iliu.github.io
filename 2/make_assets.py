"""Generates the few Project 2 results that were not already in the project2 folder.

Everything else on the page (2/index.html) is copied as-is from project2's own output
folders (1p2, 1p3, 2p1, 2p2, 2p3, 2p4); see REUSE at the bottom. The functions here are
the ones from the project2 scripts, copied without changes.

Generated here:
  1.1  box / Dx / Dy results on a portrait of me (me.jpg, cropped to just me)
  2.1  the cameraman sharpening set, and the blur-then-sharpen test on the portrait
  2.2  the high-pass / low-pass (spatial) images of the apple hybrid
  2.3  Gaussian and Laplacian stacks of the apple and the orange
  2.4  the standalone oraple
"""
import json
import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.signal import convolve2d

P = r'C:\Users\Diddl\OneDrive\Desktop\CS180_Projects\project2'
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'img')
os.makedirs(OUT, exist_ok=True)
stats = {}


def out(name):
    return os.path.join(OUT, name)


def png2arr(path):
    img = Image.open(path).convert('L')
    return np.array(img, dtype=np.float64)


# ===================================================== 1.1  (functions from part1point1.py)
def conv2d_4loops(arr, kernel):
    tkernel = np.flip(kernel).copy()
    out_ = np.zeros(arr.shape)

    x_pad = (tkernel.shape[0] - 1) // 2
    y_pad = (tkernel.shape[1] - 1) // 2
    arr_temp = np.zeros((arr.shape[0] + 2 * x_pad, arr.shape[1] + 2 * y_pad))
    arr_temp[x_pad:arr.shape[0] + x_pad, y_pad:arr.shape[1] + y_pad] = arr

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            for kx in range(tkernel.shape[0]):
                for ky in range(tkernel.shape[1]):
                    out_[i][j] += arr_temp[i + kx, j + ky] * tkernel[kx, ky]
    return out_


def conv2d_2loops(arr, kernel):
    tkernel = np.flip(kernel).copy()
    out_ = np.zeros(arr.shape)

    x_pad = (tkernel.shape[0] - 1) // 2
    y_pad = (tkernel.shape[1] - 1) // 2
    arr_temp = np.zeros((arr.shape[0] + 2 * x_pad, arr.shape[1] + 2 * y_pad))
    arr_temp[x_pad:arr.shape[0] + x_pad, y_pad:arr.shape[1] + y_pad] = arr

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            out_[i][j] += np.sum(tkernel * arr_temp[i: i + kernel.shape[0], j: j + kernel.shape[1]])
    return out_


kernel_x = np.array([[1, 0, -1]])
kernel_y = np.array([[1], [0], [-1]])
box_kernel = np.ones((9, 9)) / 81

# portrait: crop to just me (another person is in the original frame), grayscale, shrink
me_img = Image.open(os.path.join(P, 'me.jpg')).crop((100, 0, 2200, 2400)).convert('L')
me = np.array(me_img.resize((300, 343), Image.LANCZOS), dtype=np.float64)
Image.fromarray(me.astype(np.uint8)).save(out('me_gray.png'))

t = time.perf_counter(); out4 = conv2d_4loops(me, box_kernel); stats['time_4loop_s'] = time.perf_counter() - t
t = time.perf_counter(); out2 = conv2d_2loops(me, box_kernel); stats['time_2loop_s'] = time.perf_counter() - t
t = time.perf_counter(); outsp = convolve2d(me, box_kernel, mode='same'); stats['time_scipy_s'] = time.perf_counter() - t
stats['portrait_shape'] = list(me.shape)
stats['allclose_4_vs_scipy'] = bool(np.allclose(out4, outsp))
stats['allclose_2_vs_scipy'] = bool(np.allclose(out2, outsp))
stats['allclose_4_vs_2'] = bool(np.allclose(out4, out2))

kx = conv2d_2loops(me, kernel_x)
ky = conv2d_2loops(me, kernel_y)
stats['allclose_dx_vs_scipy'] = bool(np.allclose(kx, convolve2d(me, kernel_x, mode='same')))
stats['allclose_dy_vs_scipy'] = bool(np.allclose(ky, convolve2d(me, kernel_y, mode='same')))

for nm, arr in (('out2_box', out2), ('out4_box', out4), ('outsp_box', outsp),
                ('out2_kx', kx), ('out2_ky', ky)):
    plt.imsave(out('me_%s.png' % nm), arr, cmap='gray')

# ===================================================== 2.1  (from part2point1.py)
g1d = cv2.getGaussianKernel(9, 1)
gl2 = g1d @ g1d.T


def sharpfil(alpha, g=gl2):
    delta = np.zeros(g.shape)
    delta[g.shape[0] // 2, g.shape[1] // 2] = 1
    return (1 + alpha) * delta - alpha * g


man = png2arr(os.path.join(P, 'cameraman.png'))
Image.fromarray(man.astype(np.uint8)).save(out('cameraman.png'))
cam_blur = convolve2d(man, gl2, mode='same')
cam_sharp = man - cam_blur
plt.imsave(out('cam_blur.png'), cam_blur, cmap='gray')
plt.imsave(out('cam_sharp.png'), cam_sharp, cmap='gray')
plt.imsave(out('cam_sharpened.png'), man + cam_sharp, cmap='gray')
for a, nm in ((0.5, '0p5'), (1, '1'), (2, '2'), (10, '10')):
    plt.imsave(out('cam_alpha%s.png' % nm), convolve2d(man, sharpfil(a), mode='same'), cmap='gray')

# blur a sharp image (my portrait) and sharpen it again; same gray scale for all three
sig = 2
gb1 = cv2.getGaussianKernel(13, sig)
gb = gb1 @ gb1.T
me_blur = convolve2d(me, gb, mode='same')
me_re = convolve2d(me_blur, sharpfil(2, gb), mode='same')
for nm, arr in (('me_sharp_orig', me), ('me_sharp_blur', me_blur), ('me_sharp_resharp', me_re)):
    plt.imsave(out(nm + '.png'), arr, cmap='gray', vmin=0, vmax=255)
mse = lambda a, b: float(np.mean((a - b) ** 2))
psnr = lambda a, b: float(10 * np.log10(255 ** 2 / mse(a, b)))
stats['blurtest_psnr_blur'] = psnr(me, me_blur)
stats['blurtest_psnr_resharp'] = psnr(me, np.clip(me_re, 0, 255))

# ===================================================== 2.2  (hybrid_image from part2point2.py)
def hybrid_parts(im1, im2, sigma1, sigma2):
    im1g = cv2.cvtColor(im1.astype(np.float32), cv2.COLOR_RGB2GRAY)
    im2g = cv2.cvtColor(im2.astype(np.float32), cv2.COLOR_RGB2GRAY)

    g1d_l = cv2.getGaussianKernel(int(3 * sigma2 + 1), sigma2)
    img2_tmp = convolve2d(im2g, g1d_l, mode='same')
    im2_lpf = convolve2d(img2_tmp, g1d_l.T, mode='same')

    g1d_h = cv2.getGaussianKernel(int(3 * sigma1 + 1), sigma1)
    im1_hpfl_tmp = convolve2d(im1g, g1d_h, mode='same')
    im1_hpfl = convolve2d(im1_hpfl_tmp, g1d_h.T, mode='same')
    im1_hpf = im1g - im1_hpfl

    out_ = 0.5 * im1_hpf + 0.5 * im2_lpf
    out_ = (out_ - out_.min()) / (out_.max() - out_.min())
    return im1_hpf, im2_lpf, out_


goofy = plt.imread(os.path.join(P, '2p2', 'goofy_apple_aligned.png'))
sad = plt.imread(os.path.join(P, '2p2', 'sad_apple_aligned.png'))
hpf, lpf, hyb = hybrid_parts(goofy, sad, 1.5, 3)
plt.imsave(out('apple_hpf.png'), hpf, cmap='gray')
plt.imsave(out('apple_lpf.png'), lpf, cmap='gray')
ref = plt.imread(os.path.join(P, '2p2', 'goofysadapple.png'))[..., 0]
stats['apple_hybrid_matches_saved_max_diff'] = float(np.abs(hyb - ref).max())

# ===================================================== 2.3 / 2.4  (from part2point34.py)
def app_gauss(img, sigma):
    g1d_ = cv2.getGaussianKernel(int(6 * sigma + 1), sigma)
    img_tmp = convolve2d(img, g1d_, mode='same')
    return convolve2d(img_tmp, g1d_.T, mode='same')


def gaussian_stack(img, spline, levels):
    gstack_out = []
    lstack_out = []
    spline_out = []
    for level in range(-1, levels):
        if level != -1:
            spline_out.append(app_gauss(spline, 2 ** level))
            gstack_out.append(app_gauss(img, 2 ** level))
            lstack_out.append(gstack_out[len(gstack_out) - 2] - gstack_out[len(gstack_out) - 1])
        else:
            gstack_out.append(img)
            spline_out.append(spline)

    lstack_out.append(gstack_out[-1])
    return lstack_out, spline_out, gstack_out


def blend(im1, im2, mask, levels):
    mask1 = mask
    mask2 = np.ones(mask.shape) - mask
    out1 = gaussian_stack(im1, mask1, levels)
    out2 = gaussian_stack(im2, mask2, levels)
    final = np.zeros(im1.shape)
    for level in range(levels + 1):
        im1_l = out1[0][level] * out1[1][level]
        im2_l = out2[0][level] * out2[1][level]
        final += (im1_l + im2_l)
    return final


apple = plt.imread(os.path.join(P, 'apple.jpeg')) / 255.
orange = plt.imread(os.path.join(P, 'orange.jpeg')) / 255.
mask = np.zeros(apple.shape[:2])
mask[:, apple.shape[1] // 2:] = 1

oraple = np.dstack([blend(apple[:, :, c], orange[:, :, c], mask, 5) for c in range(3)])
plt.imsave(out('oraple.png'), np.clip(oraple, 0, 1))
plt.imsave(out('mask_step.png'), mask, cmap='gray')


def strips(img, tag):
    ch = [gaussian_stack(img[:, :, c], mask, 5) for c in range(3)]
    gl = [np.dstack([ch[c][2][k] for c in range(3)]) for k in range(1, 6)]    # sigma = 1, 2, 4, 8, 16
    ll = [np.dstack([ch[c][0][k] for c in range(3)]) for k in range(6)]       # 5 band-pass + base
    plt.imsave(out('%s_gstack.png' % tag), np.clip(np.hstack(gl), 0, 1))
    shown = [np.clip(x * 3 + 0.5, 0, 1) for x in ll[:-1]] + [np.clip(ll[-1], 0, 1)]
    plt.imsave(out('%s_lstack.png' % tag), np.hstack(shown))


strips(apple, 'apple')
strips(orange, 'orange')

# ===================================================== copied as-is from the project2 folders
def copy(src, dst, max_w=None):
    im = Image.open(os.path.join(P, src))
    im = im.convert('RGBA') if im.mode in ('RGBA', 'LA', 'P') else im.convert('RGB')
    if max_w and im.width > max_w:
        im = im.resize((max_w, round(im.height * max_w / im.width)), Image.LANCZOS)
    im.save(out(dst))


REUSE = {
    # 1.2
    '1p2/pd_x.png': 'pd_x.png', '1p2/pd_y.png': 'pd_y.png', '1p2/gmi.png': 'gmi_fd.png',
    '1p2/bins.png': 'bins.png',
    # 1.3
    '1p3/pd_g.png': 'pd_g.png', '1p3/gmi.png': 'gmi_gauss.png',
    '1p3/b20.png': 'b20.png', '1p3/b25.png': 'b25.png', '1p3/b30.png': 'b30.png',
    '1p3/DoGx.png': 'DoGx.png', '1p3/DoGy.png': 'DoGy.png',
    '1p3/manxDoGx.png': 'manxDoGx.png', '1p3/manxDoGy.png': 'manxDoGy.png',
    # 2.1 (Taj)
    '2p1/blurred_pd_g.png': 'taj_blur.png', '2p1/sharp.png': 'taj_sharp.png',
    '2p1/sharpened.png': 'taj_sharpened.png', '2p1/alpha0p5.png': 'taj_alpha0p5.png',
    '2p1/alpha1.png': 'taj_alpha1.png', '2p1/alpha2.png': 'taj_alpha2.png',
    '2p1/alpha10.png': 'taj_alpha10.png',
    # 2.2
    'apple_toungeout.webp': 'apple_goofy_orig.png', 'apple_sad.jpg': 'apple_sad_orig.png',
    '2p2/goofy_apple_aligned.png': 'apple_goofy_aligned.png',
    '2p2/sad_apple_aligned.png': 'apple_sad_aligned.png',
    '2p2/goof_appft_im1.png': 'ft_goofy.png', '2p2/sad_appft_im2.png': 'ft_sad.png',
    '2p2/goof_appft_im1_hpf.png': 'ft_goofy_hpf.png', '2p2/sad_appft_im2_lpf.png': 'ft_sad_lpf.png',
    '2p2/appft_hybrid.png': 'ft_apple_hybrid.png',
    '2p2/goofysadapple.png': 'apple_hybrid.png', '2p2/goofysadapple_small.png': 'apple_hybrid_small.png',
    'nutmeg.jpg': 'nutmeg_orig.png', 'DerekPicture.jpg': 'derek_orig.png',
    '2p2/derekcat.png': 'derek_hybrid.png', '2p2/derekkcat_small.png': 'derek_hybrid_small.png',
    'wetcat.png': 'wetcat_orig.png', 'drycat.png': 'drycat_orig.png',
    '2p2/wetdrycat.png': 'cat_hybrid.png', '2p2/wetdrycat_small.png': 'cat_hybrid_small.png',
    # 2.3 / 2.4
    '2p3/apple_orange.png': 'fig342.png',
    '2p4/frog_tp.png': 'frog_result.png', '2p4/frog_tp_stacks.png': 'frog_stacks.png',
    '2p4/snakeworm_small.png': 'snake_result.png', '2p4/snakeworm_stacks.png': 'snake_stacks.png',
    'apple.jpeg': 'apple.png', 'orange.jpeg': 'orange.png',
    'tadpole.png': 'tadpole.png', 'frog.png': 'frog.png', 'mask_frog_tadpole.png': 'mask_frog.png',
    'worm.png': 'worm.png', 'snake.png': 'snake.png', 'mask_sw.png': 'mask_snake.png',
}
BIG = {'fig342.png', 'frog_stacks.png', 'snake_stacks.png'}
for src, dst in REUSE.items():
    copy(src, dst, 1100 if dst in BIG else (700 if dst.endswith('_orig.png') else None))

with open(os.path.join(HERE, 'stats.json'), 'w') as f:
    json.dump(stats, f, indent=2)
print(json.dumps(stats, indent=2))
