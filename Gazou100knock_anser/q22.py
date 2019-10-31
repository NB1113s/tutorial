import cv2
import numpy as np
import function as func
import matplotlib.pyplot as plt

"""

http://ipr20.cs.ehime-u.ac.jp/column/gazo_syori/chapter2.html
ヒストグラムの平均値をm0=128、標準偏差をs0=52になるように操作せよ。
これはヒストグラムのダイナミックレンジを変更するのではなく、
ヒストグラムを平坦に変更する操作である。
平均値m、標準偏差s、のヒストグラムを平均値m0,
 標準偏差s0に変更するには、次式によって変換する。

x_out = s0/s * (x_in-m)+m0

"""

img = cv2.imread("input/imori_dark.jpg")
out = img.copy()

# flat=func.hist_normalization(out)
# print(np.mean(flat))#120.45355224609375
# print(np.std(flat))#43.71755818189537

#変更前：平均値mb、標準偏差sb->変更後：平均値ma、標準偏差sa
# mb=np.mean(flat)
# sb=np.std(flat)
# ma=128
# sa=52

# flat=(sa/sa)*(flat-mb)+mb
# flat[flat <0]=0
# flat[flat>255]=255
# flat=flat.astype(np.uint8)


flat=func.hist_manipulate(img,m0=128,s0=52)
plt.hist(flat.ravel(), bins=255, rwidth=0.8, range=(0, 255))
plt.savefig("output/q22_hist.jpg")
plt.show()

# Save result
cv2.imshow("result", flat)
cv2.waitKey(0)
cv2.imwrite("output/q22_img.jpg", flat)
