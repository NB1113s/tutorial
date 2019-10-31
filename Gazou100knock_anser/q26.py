import cv2
import numpy as np
import function as func
import matplotlib.pyplot as plt

"""
Bi-linear補間により画像を1.5倍に拡大せよ。

Bi-linear補間とは周辺の４画素に距離に応じた重みをつけることで補完する手法である。 計算量が多いだけ処理時間がかかるが、画質の劣化を抑えることができる。

拡大画像の座標(x', y')を拡大率aで割り、floor(x'/a, y'/a)を求める。
元画像の(x'/a, y'/a)の周囲4画素、I(x,y), I(x+1,y), I(x,y+1), I(x+1, y+1)を求める
それぞれの画素と(x'/a, y'/a)との距離dを求め、重み付けする。 w = d / Sum d
次式によって拡大画像の画素(x',y')を求める。 dx = x'/a - x , dy = y'/a - y

I'(x',y')=(1-dx)(1-dy)*I(x,y)+dx(1-dy)*I(x+1,y)+(1-dx)dy*I(x,y+1)+dxdy*I(x+1,y+1)

"""
#入力
img = cv2.imread("input/imori.jpg")

#処理
out=func.bl_interpolate(img,ax=1.5,ay=1.5)

# #確認
cv2.imshow("result", out)
cv2.waitKey(0)

#出力
cv2.imwrite("output/q26.jpg", out)