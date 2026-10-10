import numpy as np
from scipy.signal import convolve2d
from PIL import Image
import matplotlib.pyplot as plt
import cv2

def png2arr(path):
    img = Image.open(path).convert('L')
    out = np.array(img, dtype=np.float64)
    return out

man = png2arr(r'project2\cameraman.png')

g1d = cv2.getGaussianKernel(9, 1)
gl2 = g1d @ g1d.T

pd_g = convolve2d(man, gl2, mode='same')

kernel_x = np.array([[1, 0, -1]])
kernel_y = np.array([[1], [0], [-1]])

pd_x = convolve2d(pd_g, kernel_x, mode='same')
pd_y = convolve2d(pd_g, kernel_y, mode='same')

DoGx = convolve2d(gl2, kernel_x) 
DoGy = convolve2d(gl2, kernel_y) 

plt.imsave('project2/1p3/DoGx.png', DoGx, cmap='gray')
plt.imsave('project2/1p3/DoGy.png', DoGy, cmap='gray')

plt.imsave('project2/1p3/manxDoGx.png', convolve2d(man, DoGx, mode='same'), cmap='gray')
plt.imsave('project2/1p3/manxDoGy.png', convolve2d(man, DoGy, mode='same'), cmap='gray')

gmi = np.sqrt(pd_x ** 2 + pd_y ** 2)

print(np.allclose(pd_x, convolve2d(man, DoGx, mode='same')))

plt.imsave('project2/1p3/gmi.png', gmi, cmap='gray')
plt.imsave('project2/1p3/pd_g.png', pd_g, cmap='gray')

def binthresh(threshold):
    return gmi > threshold

plt.imsave('project2/1p3/b20.png', binthresh(20), cmap='gray')
plt.imsave('project2/1p3/b25.png', binthresh(25), cmap='gray')
plt.imsave('project2/1p3/b30.png', binthresh(30), cmap='gray')

fig, axes = plt.subplots(1, 5, figsize=(16,4))
images = [pd_g, gmi, binthresh(20), binthresh(25), binthresh(30)]
for i, ax in enumerate(axes):
    ax.imshow(images[i], cmap="gray")
    ax.axis('off')

plt.tight_layout()
plt.show()