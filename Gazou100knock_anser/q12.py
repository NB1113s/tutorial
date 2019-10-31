import cv2
import numpy as np
import function as func

img = cv2.imread("input/imori.jpg")
out = img.copy()
out = func.motion_filter(out,size=3)

cv2.imwrite("output/q12.jpg",out)
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.destroyAllWindows()



