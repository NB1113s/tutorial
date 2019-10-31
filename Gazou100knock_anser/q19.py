import cv2
import numpy as np
import function as func

# LoG filter エッジ抽出
# LoGフィルタ(sigma=3、カーネルサイズ=5)を実装し、imori_noise.jpgのエッジを検出せよ。
# LoGフィルタとはLaplacian of Gaussianであり、ガウシアンフィルタで画像を平滑化した後にラプラシアンフィルタで輪郭を取り出すフィルタである。
# Laplcianフィルタは二次微分をとるのでノイズが強調されるのを防ぐために、予めGaussianフィルタでノイズを抑える。
# LoGフィルタは次式で定義される。


img = cv2.imread("input/imori.jpg")
img = func.RGB2GRAY(img)
out = img.copy()

print(out.shape)
out = func.LoG_filter(out,size=3)

cv2.imwrite("output/q19.jpg",out)
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.destroyAllWindows()

