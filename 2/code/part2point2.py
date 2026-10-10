import matplotlib.pyplot as plt
from align_image_code import align_images
import cv2
import numpy as np
from scipy.signal import convolve2d
from PIL import Image
import matplotlib.pyplot as plt
import cv2

# First load images

# high sf
im1 = plt.imread('project2/nutmeg.jpg')
im2 = plt.imread('project2/DerekPicture.jpg')

im1_aligned = None
im2_aligned = None

realign = False

# Next align images (this code is provided, but may be improved)
if realign:
    im1_aligned, im2_aligned = align_images(im1, im2)

    plt.imsave('project2/2p2/temp_am1_aligned.png', im1_aligned, cmap='gray')
    plt.imsave('project2/2p2/temp_im2_aligned.png', im2_aligned, cmap='gray')
else:
    im1_aligned = plt.imread('project2/2p2/temp_am1_aligned.png')
    im2_aligned = plt.imread('project2/2p2/temp_im2_aligned.png')

## You will provide the code below. Sigma1 and sigma2 are arbitrary 
## cutoff values for the high and low frequencies

def hybrid_image(im1, im2, sigma1, sigma2):

    im1g =  cv2.cvtColor(im1.astype(np.float32), cv2.COLOR_RGB2GRAY)
    im2g =  cv2.cvtColor(im2.astype(np.float32), cv2.COLOR_RGB2GRAY)

    g1d_l = cv2.getGaussianKernel(int(3*sigma2 + 1), sigma2)
    img2_tmp = convolve2d(im2g, g1d_l, mode='same')
    im2_lpf = convolve2d(img2_tmp, g1d_l.T, mode='same')

    g1d_h = cv2.getGaussianKernel(int(3*sigma1 + 1), sigma1)

    im1_hpfl_tmp = convolve2d(im1g, g1d_h, mode='same')

    im1_hpfl = convolve2d(im1_hpfl_tmp, g1d_h.T, mode='same')

    im1_hpf = im1g - im1_hpfl

    out = 0.5 * im1_hpf + 0.5 * im2_lpf
    
    out = (out - out.min()) / (out.max() - out.min())

    ft = lambda x: np.log(np.abs(np.fft.fftshift(np.fft.fft2(x))) + 1e-8)
    plt.imsave('project2/2p2/ft_im1.png', ft(im1g), cmap='gray')
    plt.imsave('project2/2p2/ft_im2.png', ft(im2g), cmap='gray')
    plt.imsave('project2/2p2/ft_im1_hpf.png', ft(im1_hpf), cmap='gray')
    plt.imsave('project2/2p2/ft_im2_lpf.png', ft(im2_lpf), cmap='gray')
    plt.imsave('project2/2p2/ft_hybrid.png', ft(out), cmap='gray')

    return out

#sigma1 is blur u subtract, sigma2 is blur u add
sigma1 = 15
sigma2 = 20
hybrid = hybrid_image(im1_aligned, im2_aligned, sigma1, sigma2)
plt.imsave('project2/2p2/derekcat.png', hybrid, cmap='gray')

small = cv2.resize(hybrid, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
plt.imsave('project2/2p2/derekkcat_small.png', small, cmap='gray')
