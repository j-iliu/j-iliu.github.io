# CS180 (CS280A): Project 1 starter Python code

# these are just some suggested libraries
# instead of scikit-image you could use matplotlib and opencv to read, write, and display images

from math import sqrt

import numpy as np
import skimage as sk
import skimage.io as skio

# name of the input file
imname = 'CS180_fa2026_proj1_data/cathedral.jpg'

# read in the image
im = skio.imread(imname)

# convert to double (might want to do this later on to save memory)    
im = sk.img_as_float(im)
    
# compute the height of each part (just 1/3 of total)
height = np.floor(im.shape[0] / 3.0).astype(int)

print (im.shape)

# separate color channels
b = im[:height]
g = im[height: 2*height]
r = im[2*height: 3*height]

# align the images
# functions that might be useful for aligning the images include:
# np.roll, np.sum, sk.transform.rescale (for multiscale)

def align(im1, im2):
    best = np.inf
    best_x = 0
    best_y = 0

    crop = 0.1
    slice1 = (slice(int(crop*im1.shape[0]), int((1-crop)*im1.shape[0])), slice(int(crop*im1.shape[1]), int((1-crop)*im1.shape[1])))
    for x in range(-15, 15):
        for y in range(-15, 15):
            temp1 = np.roll(np.roll(im1, x, axis=0), y, axis=1)
            new = np.sqrt(np.sum((temp1[slice1]-im2[slice1])**2))
            if best > new:
                best = new
                best_x = x
                best_y = y
    print(best_x, best_y)
    print(best)
    return np.roll(np.roll(im1, best_x, axis=0), best_y, axis=1)

ag = align(g, b)
ar = align(r, b)
# create a color image
im_out = np.dstack([ar, ag, b])

# save the image
fname = 'CS180_fa2026_proj1_data/single_alignment.jpg'
skio.imsave(fname, sk.img_as_ubyte(im_out))

# display the image
#skio.imshow(im_out)
#skio.show()
