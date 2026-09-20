# CS180 (CS280A): Project 1 starter Python code

# these are just some suggested libraries
# instead of scikit-image you could use matplotlib and opencv to read, write, and display images

from math import sqrt

import numpy as np
import skimage as sk
import skimage.io as skio
import matplotlib.pyplot as plt
import time

def align(im1, im2, guess_x = 0, guess_y = 0):

    height = im1.shape[0]
    width = im1.shape[1]

    if height < 10 or width < 20:
        best_x, best_y = align_helper(im1, im2, guess_x, guess_y)
        #print("base", im1.shape, best_x, best_y)
        return im1, best_x, best_y
    else:
        im1_half = sk.transform.rescale(im1, 0.5, anti_aliasing=True)
        im2_half = sk.transform.rescale(im2, 0.5, anti_aliasing=True)
        #print("recursed", im1_half.shape)
        _, guess_xhalf, guess_yhalf = align(im1_half, im2_half, guess_x/2, guess_y/2)

        best_x, best_y = align_helper(im1, im2, guess_xhalf*2, guess_yhalf*2)
        #print(im1.shape, "best_x, best_y", best_x, best_y)
        return im1, best_x, best_y

def align_helper(im1, im2, guess_x, guess_y):
    best = np.inf
    best_x = 0
    best_y = 0
    guess_range = 3

    crop = 0.05

    my = int(crop*im1.shape[0])
    mx = int(crop*im1.shape[1])

    tmy = my + abs(int(guess_x)) + guess_range
    tmx = mx + abs(int(guess_y)) + guess_range

    no_margin = (slice(tmy, im1.shape[0]-tmy), slice(tmx, im1.shape[1]-tmx))

    for x in range(-guess_range + int(guess_x), guess_range + int(guess_x) + 1):
        for y in  range(-guess_range + int(guess_y), guess_range + int(guess_y) + 1):

            temp1 = np.roll(np.roll(im1, x, axis=0), y, axis=1)

            a = temp1[no_margin]- np.mean(temp1[no_margin])
            c = im2[no_margin] - np.mean(im2[no_margin])

            new = -np.sum(a*c) / (np.sqrt(np.sum(a**2)) * np.sqrt(np.sum(c**2)))

            #new = np.sqrt(np.sum((temp1[slice1]-im2[slice1])**2))
            
            if best > new:
                best = new
                best_x = x
                best_y = y

    return best_x, best_y


def align_rgb(im, imname):
    height = np.floor(im.shape[0] / 3.0).astype(int)
    b = im[:height]
    g = im[height: 2*height]
    r = im[2*height: 3*height]

    _, gx, gy = align(g, b)
    _, rx, ry = align(r, g)
    rx += gx
    ry += gy

    print("final:", "green: (", gx, gy, ")", "red: (", rx, ry, ")")

    ar = np.roll(np.roll(r, rx, axis=0), ry, axis=1)
    ag = np.roll(np.roll(g, gx, axis=0), gy, axis=1)

    # create a color image
    im_out = np.dstack([ar, ag, b])

    # save the image
    fname = 'out/' + imname.split('.')[0] + '_aligned.jpg'
    skio.imsave(fname, sk.img_as_ubyte(im_out))

    plt.imshow(im_out)
    plt.show()

list = ['cathedral.jpg',
               'church.tif',
               'emir.tif', 
               'harvesters.tif', 
               'icon.tif', 
               'ilemselga.tif', 
               'melons.tif', 
               'monastery.jpg', 
               'religous_painting.tif', 
               'self_portrait.tif', 
               'siren.tif', 
               'three_generations.tif', 
               'tobolsk.jpg', 
               'wharf.tif']

list = ['master-pnp-prok-00000-00087u.tif',
        'master-pnp-prok-00000-00090u.tif',
        'master-pnp-prok-01700-01769u.tif',
        'master-pnp-prok-01800-01841u.tif'
        ]

prefix = 'CS180_fa2026_proj1_data/'
prefix_additional = 'additional/'

for imn in list:
    im = skio.imread(prefix_additional + imn)
    im = sk.img_as_float(im).astype(np.float32)
    start = time.perf_counter()
    align_rgb(im, imn)
    end = time.perf_counter() - start
    print(f'{imn}: {end:0.4f} seconds')