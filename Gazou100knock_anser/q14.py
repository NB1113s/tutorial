import cv2
import numpy as np
import function as func

img = cv2.imread("input/imori.jpg")
img = func.RGB2GRAY(img)
out = img.copy()
out = func.differential_filter(out,size=3,axis=1)#axis=0 tate 1 yoko

cv2.imwrite("output/q14_yoko.jpg",out)
cv2.imshow("result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()



