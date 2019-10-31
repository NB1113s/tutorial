import cv2
import numpy as np
import function as func
import matplotlib.pyplot as plt

"""
# Nearest neighbor interpolation

最近傍補間により画像を1.5倍に拡大せよ。

最近傍補間(Nearest Neighbor)は画像の拡大時に最近傍にある画素をそのまま使う手法である。
 シンプルで処理速度が速いが、画質の劣化は著しい。

次式で補間される。
 I' ... 拡大後の画像、 I ... 拡大前の画像、a ... 拡大率、[ ] ... 四捨五入

I'(x,y)=I(round(x/ax),round(y/ay))

"""
#入力
img = cv2.imread("input/imori.jpg")


#処理
# flat=func.Gamma_correction(img,c=1.,gamma=2.2)
out=img.copy().astype(np.float32)
alpha=1.5

out=func.nn_interpolate(out,alpha=1.5)

# #確認
cv2.imshow("result", out)
cv2.waitKey(0)

#出力
cv2.imwrite("output/q25.jpg", out)