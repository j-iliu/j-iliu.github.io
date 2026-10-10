import matplotlib.pyplot as plt
from align_image_code import align_images
import cv2
import numpy as np
from scipy.signal import convolve2d
from PIL import Image
import matplotlib.pyplot as plt
import cv2

im1 = plt.imread('project2/nutmeg.png')

im2 = plt.imread('project2/DerekPicture.png')

print(im1.shape, im2.shape)

def app_gauss(img, sigma):
    g1d = cv2.getGaussianKernel(int(6*sigma + 1), sigma)
    img_tmp = convolve2d(img, g1d, mode='same')
    out = convolve2d(img_tmp, g1d.T, mode='same')
    return out

def gaussian_stack(img, spline, levels):
    gstack_out = []
    lstack_out = []
    spline_out = []
    for level in range(-1, levels):
        if(level != -1):
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

    blend1 = []
    blend2 = []

    final = np.zeros(im1.shape)

    for level in range(levels + 1):
        im1_l = out1[0][level] * out1[1][level]
        im2_l = out2[0][level] * out2[1][level]

        blend1.append(im1_l)
        blend2.append(im2_l)

        final += (im1_l + im2_l)

    return final, blend1, blend2

#mask = np.zeros(im1.shape[:2])
#mask[:, im1.shape[1] // 2:] = 1 

mask = plt.imread('project2/mask_frog_tadpole.png')[:, :, 0]

channels = [blend(im1[:, :, c], im2[:, :, c], mask, 5) for c in range(3)]

outf = np.dstack([c[0] for c in channels]) 
b1 = [np.dstack([c[1][k] for c in channels]) for k in range(len(channels[0][1]))] 
b2 = [np.dstack([c[2][k] for c in channels]) for k in range(len(channels[0][2]))] 
b3 = [x + y for x, y in zip(b1, b2)]

def show_lstack(img):
    return np.clip(img * 3 + 0.5, 0, 1)

rows = []
for l in [0, 2, 4]:
    rows.append([show_lstack(b1[l]), show_lstack(b2[l]), show_lstack(b3[l])])

im1_t = sum(b1)
im2_t = sum(b2) 

rows.append([np.clip(im1_t, 0, 1), np.clip(im2_t, 0, 1), np.clip(outf, 0, 1)])


fig, axes = plt.subplots(len(rows), 3, figsize=(12, 16), constrained_layout=True)
letters = "abcdefghijklmnopqrstuv"
for r in range(len(rows)):
    for c in range(3):
        ax = axes[r,c]
        ax.imshow(rows[r][c])
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xlabel(f'({letters[r*3 + c]})')

fig.savefig('project2/2p4/frog_tp_stacks.png', dpi=150, bbox_inches='tight')
plt.imsave('project2/2p4/frog_tp.png', np.clip(outf, 0, 1), cmap='gray')

plt.show()