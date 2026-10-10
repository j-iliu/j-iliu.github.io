import numpy as np
from scipy.signal import convolve2d
from PIL import Image
import matplotlib.pyplot as plt
import time


def conv2d_4loops(arr, kernel):
    tkernel = np.flip(kernel).copy()
    out = np.zeros(arr.shape)

    x_pad = (tkernel.shape[0] - 1) // 2
    y_pad = (tkernel.shape[1] - 1) // 2
    arr_temp = np.zeros((arr.shape[0] + 2 * x_pad, arr.shape[1] + 2 * y_pad))
    arr_temp[x_pad:arr.shape[0] + x_pad, y_pad:arr.shape[1] + y_pad] = arr

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            for kx in range(tkernel.shape[0]):
                for ky in range(tkernel.shape[1]):
                    out[i][j] += arr_temp[i + kx,j + ky] * tkernel[kx, ky]
    return out

def conv2d_2loops(arr, kernel):
    tkernel = np.flip(kernel).copy()
    out = np.zeros(arr.shape)
    
    x_pad = (tkernel.shape[0] - 1) // 2
    y_pad = (tkernel.shape[1] - 1) // 2
    arr_temp = np.zeros((arr.shape[0] + 2 * x_pad, arr.shape[1] + 2 * y_pad))
    arr_temp[x_pad:arr.shape[0] + x_pad, y_pad:arr.shape[1] + y_pad] = arr

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            out[i][j] += np.sum(tkernel * arr_temp[i : i + kernel.shape[0], j : j + kernel.shape[1]])
    
    return out

def png2arr(path):
    img = Image.open(path).convert('L')
    out = np.array(img, dtype=np.float64)
    return out

kernel_xy = np.array([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]])
kernel_x = np.array([[1, 0, -1]])
kernel_y = np.array([[1], [0], [-1]])

box_kernel = np.ones((9,9)) / 81

cat = png2arr(r'project2\cat.png')

start_time = time.perf_counter()
out4 = conv2d_4loops(cat, box_kernel)
end_time = time.perf_counter()
execution_time = end_time - start_time
print(f"4 Loop: {execution_time} ms")

start_time = time.perf_counter()
out2 = conv2d_2loops(cat, box_kernel)
end_time = time.perf_counter()
execution_time = end_time - start_time
print(f"2 Loop: {execution_time} ms")

start_time = time.perf_counter()
outsp = convolve2d(cat, box_kernel, mode='same')
end_time = time.perf_counter()
execution_time = end_time - start_time
print(f"scipy : {execution_time} ms")

print(np.allclose(out4, outsp))
print(np.allclose(out4, out2))
print(np.allclose(out2, outsp))

#4 Loop: 3.651648400002159 ms
#2 Loop: 0.5036876999947708 ms
#scipy : 0.022186099988175556 ms

plt.imsave('project2/1p1/out2_box.png', out2, cmap='gray')
plt.imsave('project2/1p1/out4_box.png', out4, cmap='gray')
plt.imsave('project2/1p1/outsp_box.png', outsp, cmap='gray')

plt.imsave('project2/1p1/out2_kx.png', conv2d_2loops(cat, kernel_x), cmap='gray')
plt.imsave('project2/1p1/out2_ky.png', conv2d_2loops(cat, kernel_y), cmap='gray')

fig, axes = plt.subplots(1, 4, figsize=(16,4))
images = [cat, out4, out2, outsp]
for i, ax in enumerate(axes):
    ax.imshow(images[i], cmap="gray")
    ax.axis('off')

plt.tight_layout()
fig.savefig("project2/1p1/norm.png")
plt.show()
