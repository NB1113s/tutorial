import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import function as func

img = cv2.imread("input/imori.jpg")
out = img.copy()
out = func.mean_filter(out,size=3)

cv2.imwrite("output/q11.jpg",out)
cv2.imshow("",out)
cv2.waitKey(0)
cv2.destroyAllWindows()