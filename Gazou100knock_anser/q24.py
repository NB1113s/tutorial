import cv2
import numpy as np
import function as func
import matplotlib.pyplot as plt

"""
Gamma correction
imori_gamma.jpgに対してガンマ補正(c=1, g=2.2)を実行せよ。
ガンマ補正とは、カメラなどの媒体の経由によって画素値が非線形的に変換された場合の補正である。
 ディスプレイなどで画像をそのまま表示すると画面が暗くなってしまうため、RGBの値を予め大きくすることで、ディスプレイの特性を排除した画像表示を行うことがガンマ補正の目的である。
非線形変換は次式で起こるとされる。 
ただしxは[0,1]に正規化されている。 c ... 定数、g ... ガンマ特性(通常は2.2)

x' = c*I_in^g
そこで、ガンマ補正は次式で行われる。
I_out=(1/c * I_{in})^{1/g}
"""

img = cv2.imread("input/imori_gamma.jpg")
flat=func.Gamma_correction(img,c=1.,gamma=2.2)

# Save result
cv2.imshow("result", flat)
cv2.waitKey(0)
cv2.imwrite("output/q24.jpg", flat)