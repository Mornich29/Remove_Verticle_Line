import numpy as np
from skimage import io
from matplotlib import pyplot as plt
from scipy import ndimage as ndi
from skimage.color import gray2rgb
from skimage.restoration import inpaint_biharmonic

gray_image = io.imread("gray_line.jpg")

img = np.stack((gray_image, gray_image, gray_image), axis=-1)

r, c, k = img.shape

mask = np.zeros((r, c))

# Every pixel that has value in every color channel more than 205 
for i in range(r):
    for j in range(c):
        if img[i, j, 0] > 205 and img[i, j, 1] > 205 and img[i, j, 2] > 205:
            mask[i, j] = 1

se = np.ones((5, 5))
e_m = ndi.binary_dilation(mask, se)


mask_bl = ndi.gaussian_filter(mask.astype(float), 1)
mask_bl = mask_bl.astype(bool)

# Change into 2d shape
img_2d = img[..., 0]
remove_line_img_2d = inpaint_biharmonic(img_2d, mask_bl)
# Change back into 3d shape.
remove_line_img = np.stack((remove_line_img_2d, remove_line_img_2d, remove_line_img_2d), axis=-1)

# Change type from float to int8 before save
remove_line_img = (remove_line_img * 255).astype(np.uint8)

io.imsave('withoutline.jpg', remove_line_img)
