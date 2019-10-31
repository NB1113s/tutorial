import cv2
import matplotlib.pyplot as plt
import numpy as np
import function as func


"""
q1 code RGB->BGR
"""
img = cv2.imread("input/imori.jpg")
img = func.BGR2RGB(img)
# plt.imshow(img)
# plt.show()

cv2.imwrite("output/q1.jpg", img)
# cv2.imshow("result", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# ====================================================================
"""
q2 code グレースケール化
"""
img = cv2.imread("input/imori.jpg")
img = func.BGR2RGB(img)
out = func.RGB2GRAY(img)
cv2.imwrite("output/q2.jpg", out)
# cv2.imshow("result", out)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# ====================================================================
"""
q3 code 二値かせよ
"""
img = cv2.imread("input/imori.jpg")
img = func.RGB2GRAY(img)#gray scale
out = func.binarization(img,th=128)
cv2.imwrite("outout/q3.jpg", out)
# cv2.imshow("result", out)
# cv2.destroyAllWindows()
# cv2.waitKey(0)

# ====================================================================
"""
q4.大津の二値か
"""
img = cv2.imread("input/imori.jpg")
out = func.RGB2GRAY(img)

out = func.Obinarization(out)

cv2.imwrite("output/q4.jpg", out)
# cv2.imshow("result", out)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# ====================================================================
"""
q5.HSV変換 Hue：色相、Saturation：彩度、Value：明度
"""
img = cv2.imread("input/imori.jpg")
img = func.BGR2RGB(img)

# RGB > HSV
hsv = func.BGR2HSV(img)

# Transpose Hue
hsv[..., 0] = (hsv[..., 0] + 180) % 360

# HSV > RGB
out = func.HSV2BGR(img, hsv)

cv2.imwrite("q5.jpg", out)
# cv2.imshow("result", out)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# ====================================================================
"""
q6.減色処理
"""
img = cv2.imread("input/imori.jpg")
img = func.BGR2RGB(img)

# img2=img.copy().astype(np.float32)
# img2=img2//64*32
# out=img2.astype(np.uint8)

out=func.tone_reduction(img)

cv2.imwrite("output/q6.jpg", out)
# cv2.imshow("result", out)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# ====================================================================
# """
# q7.平均プーリング
# """

img = cv2.imread("input/imori.jpg")
# img = func.BGR2RGB(img)

#全体操作を行う際は配列自体に演算を施すだけでよい
# img 128x128x3
# print(img.shape)

# out=img.copy()
# print(out.shape)

# out[0:16,0:16]=np.mean(out[0:16,0:16])
# n=out.shape[0]/8

# for i in range(int(img.shape[0]/8)):
    # print(i)
# n=8
# for i in range(int(img.shape[0]/n)):
#     for j in range(int(img.shape[1]/n)):
#         for k in range(img.shape[2]):
#             out[n*i:n*i+n,n*j:n*j+n,k]=np.mean(out[n*i:n*i+n,n*j:n*j+n,k])

# out=out.astype(np.uint8)


out=func.average_pooling(img,gx=2,gy=10)

cv2.imwrite("output/q7.jpg", out)
# cv2.imshow("result", out)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# # ====================================================================
# """
# q8.最大プーリング
# """

img = cv2.imread("input/imori.jpg")

out=img.copy()
out=func.max_pooling(out,gx=2,gy=10)

cv2.imwrite("output/q8.jpg", out)
# cv2.imshow("result", out)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# ====================================================================
# """
# q9.ガウシアンフィルタ
# 画像の平滑化を行うフィルタの一種：ノイズ除去
# 他にはメディアンフィルタ、平準化フィルタ、LoGフィルタがある
# """

img=cv2.imread("input/imori_noise.jpg")
out=img.copy()

out=func.gaussian_filter(out,size=3,sigma=1.3)
cv2.imwrite("output/q9.jpg", out)
# cv2.imshow("result", out)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# ====================================================================
# """
# q10.メディアンフィルタ
# 画像の平滑化を行うフィルタの一種：ノイズ除去
# 他にはメディアンフィルタ、平準化フィルタ、LoGフィルタがある
# """

img=cv2.imread("input/imori_noise.jpg")
out=img.copy()

out=func.median_filter(out,size=3,sigma=1.3)
cv2.imwrite("output/q10.jpg", out)
# cv2.imshow("result", out)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# ====================================================================
# """
# extra.Laplacian Of Gaussian Filter
# ガウシアンフィルタを少しいじっただけ
# 画像の平滑化を行うフィルタの一種：ノイズ除去
# 他にはメディアンフィルタ、平準化フィルタ、LoGフィルタがある
# """

img=cv2.imread("input/imori_noise.jpg")
out=img.copy()

out=func.LoG_filter(out,size=3,sigma=1.3)
cv2.imwrite("output/q10_LoG.jpg", out)
# cv2.imshow("result", out)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# ====================================================================
"""
extra.mean Filter
ガウシアンフィルタを少しいじっただけ
画像の平滑化を行うフィルタの一種：ノイズ除去
他にはメディアンフィルタ、平準化フィルタ、LoGフィルタがある
"""

img=cv2.imread("input/imori.jpg")
out=img.copy()

out=func.mean_filter(out,size=3)
cv2.imwrite("output/q10_mean.jpg", out)
# cv2.imshow("result", out)
# cv2.waitKey(0)
# cv2.destroyAllWindows()