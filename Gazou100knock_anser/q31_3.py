import cv2
import numpy as np
import matplotlib.pyplot as plt


# Read image
img = cv2.imread("input/imori.jpg").astype(np.float32)

# Affine
H,W,C=img.shape
dy=30
dx=30

# Affine hyper parameters
a = 1.
b = dx / H
c = dy / W
d = 1.
tx = 0.
ty = 0.

# prepare temporary
_img = np.zeros((H+2, W+2, C), dtype=np.float32)

# insert image to center of temporary
_img[1:H+1, 1:W+1] = img

# prepare affine image temporary
H_new = np.ceil(dy + H).astype(np.int)
W_new = np.ceil(dx + W).astype(np.int)
out = np.zeros((H_new, W_new, C), dtype=np.float32)

# preprare assigned index
x_new = np.tile(np.arange(W_new), (H_new, 1))
y_new = np.arange(H_new).repeat(W_new).reshape(H_new, -1)

# prepare inverse matrix for affine
adbc = a * d - b * c
x = np.round((d * x_new  - b * y_new) / adbc).astype(np.int) - tx + 1
y = np.round((-c * x_new + a * y_new) / adbc).astype(np.int) - ty + 1

x = np.minimum(np.maximum(x, 0), W+1).astype(np.int)
y = np.minimum(np.maximum(y, 0), H+1).astype(np.int)

# assign value from original to affine image
out[y_new, x_new] = _img[y, x]
out = out.astype(np.uint8)


# out = affine(_img, a=1, b=0, c=A, d=1,tx=0, ty=0)
# out2 = affine(out, a=1, b=B, c=0, d=1,tx=0, ty=0)


# Save result
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.imwrite("output/q31_3.jpg", out)