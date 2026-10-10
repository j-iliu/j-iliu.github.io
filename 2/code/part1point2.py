import numpy as np
from scipy.signal import convolve2d
from PIL import Image
import matplotlib.pyplot as plt

def png2arr(path):
    img = Image.open(path).convert('L')
    out = np.array(img, dtype=np.float64)
    return out

man = png2arr(r'project2\cameraman.png')

kernel_x = np.array([[1, 0, -1]])
kernel_y = np.array([[1], [0], [-1]])

pd_x = convolve2d(man, kernel_x, mode='same')
pd_y = convolve2d(man, kernel_y, mode='same')

gmi = np.sqrt(pd_x ** 2 + pd_y ** 2)

def binthresh(threshold):
    return gmi > threshold


plt.imsave('project2/1p2/pd_x.png', pd_x, cmap='gray')
plt.imsave('project2/1p2/pd_y.png', pd_y, cmap='gray')
plt.imsave('project2/1p2/gmi.png', gmi, cmap='gray')

fig, axes = plt.subplots(1, 3, figsize=(16,4))
images = [binthresh(50), binthresh(55), binthresh(60)]
titles = [50, 55, 60]
for i, ax in enumerate(axes):
    ax.imshow(images[i], cmap="gray")
    ax.set_title(titles[i])
    ax.axis('off')

plt.tight_layout()
fig.savefig('project2/1p2/bins.png', dpi=150, bbox_inches='tight')
plt.show()