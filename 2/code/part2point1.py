import numpy as np
from scipy.signal import convolve2d
from PIL import Image
import matplotlib.pyplot as plt
import cv2

def png2arr(path):
    img = Image.open(path).convert('L')
    out = np.array(img, dtype=np.float64)
    return out

taj = png2arr(r'project2\taj.jpg')

g1d = cv2.getGaussianKernel(9, 1)
gl2 = g1d @ g1d.T

pd_g = convolve2d(taj, gl2, mode='same')

sharp = taj - pd_g

plt.imsave('project2/2p1/blurred_pd_g.png', pd_g, cmap='gray')

plt.imsave('project2/2p1/sharp.png', sharp, cmap='gray')

sharpened = taj + sharp

plt.imsave('project2/2p1/sharpened.png', sharpened, cmap='gray')

def sharpfil(alpha):
    delta = np.zeros(gl2.shape)
    delta[gl2.shape[0]//2, gl2.shape[1]//2] = 1
    return (1 + alpha) * delta - alpha * gl2

alpha0p5 = convolve2d(taj, sharpfil(0.5), mode='same')
alpha1 = convolve2d(taj, sharpfil(1), mode='same')
alpha2 = convolve2d(taj, sharpfil(2), mode='same')
alpha10 = convolve2d(taj, sharpfil(10), mode='same')

plt.imsave('project2/2p1/alpha0p5.png', alpha0p5, cmap='gray')
plt.imsave('project2/2p1/alpha1.png', alpha1, cmap='gray')
plt.imsave('project2/2p1/alpha2.png', alpha2, cmap='gray')
plt.imsave('project2/2p1/alpha10.png', alpha10, cmap='gray')


print(np.allclose(sharpened, alpha1))

fig, axes = plt.subplots(1, 2, figsize=(16,4))
images = [taj, sharpened]
for i, ax in enumerate(axes):
    ax.imshow(images[i], cmap="gray")
    ax.axis('off')

plt.tight_layout()
plt.show()