import cv2
import numpy as np
import function as func
import matplotlib.pyplot as plt

"""
ヒストグラム正規化を実装せよ。

ヒストグラムは偏りを持っていることが伺える。 
例えば、0に近い画素が多ければ画像は全体的に暗く、255に近い画素が多ければ画像は明るくなる。 
ヒストグラムが局所的に偏っていることをダイナミックレンジが狭いなどと表現する。 そのため画像を人の目に見やすくするために、ヒストグラムを正規化したり平坦化したりなどの処理が必要である。

このヒストグラム正規化は濃度階調変換(gray-scale transformation) と呼ばれ、
[c,d]の画素値を持つ画像を[a,b]のレンジに変換する場合は次式で実現できる。 今回はimori_dark.jpgを[0, 255]のレンジにそれぞれ変換する。

"""

img = cv2.imread("input/imori_dark.jpg")
out = img.copy()


# flat=img.ravel()
# c=np.min(flat)#62
# d=np.max(flat)#131
# a=0.
# b=255.

# const=(b-a)/(d-c)

# c=0
# for i in range(len(flat)):
#     if flat[i] < c:
#         flat[i]=a
#     elif c <= flat[i] < d:
#         flat[i]=const*(flat[i]-c)+a
#     else:
#         flat[i]=b

flat=func.hist_normalization(out)
plt.hist(flat.ravel(), bins=255, rwidth=0.8, range=(0, 255))
plt.savefig("output/q21_hist.jpg")
plt.show()

# Save result
cv2.imshow("result", flat)
cv2.waitKey(0)
cv2.imwrite("output/q21_img.jpg", flat)
