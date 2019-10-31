import cv2
import numpy as np
import function as func

# エンボス加工
img = cv2.imread("input/imori.jpg")
img = func.RGB2GRAY(img)
out = img.copy()
out = func.Emboss_filter(out,size=3)

cv2.imwrite("output/q18.jpg",out)
cv2.imshow("result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

