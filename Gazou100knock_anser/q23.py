import cv2
import numpy as np
import function as func
import matplotlib.pyplot as plt

"""
Histogram equalization
ヒストグラム平坦化を実装せよ。
ヒストグラム平坦化とはヒストグラムを平坦に変更する操作であり、
上記の平均値や標準偏差などを必要とせず、ヒストグラム値を均衡にする操作である。
これは次式で定義される。 ただし、S ... 画素値の総数、
Zmax ... 画素値の最大値、h(z) ... 濃度zの度数


Z'=Z_max/S * \sum_{i=0}^{z} h(i)
"""

img = cv2.imread("input/imori_dark.jpg")

flat=func.hist_equalization(img,z_max=255)

plt.hist(flat.ravel(), bins=255, rwidth=0.8, range=(0, 255))
plt.savefig("output/q23_hist.jpg")
plt.show()

# Save result
cv2.imshow("result", flat)
cv2.waitKey(0)
cv2.imwrite("output/q23_img.jpg", flat)
