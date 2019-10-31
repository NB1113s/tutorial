import cv2
import numpy as np
import function as func
import matplotlib.pyplot as plt

img = cv2.imread("input/imori.jpg")
img = func.RGB2GRAY(img)
out = img.copy()

# https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.ravel.html
flat=img.ravel()

plt.hist(flat, bins=255, rwidth=0.8, range=(0, 255))
plt.savefig("output/q20.jpg")
plt.show()