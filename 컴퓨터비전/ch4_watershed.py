# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 10:49:25 2023

@author: BigData
"""

import cv2 as cv
import numpy as np

#%%
img=cv.imread("group.jpg")
gray=cv.cvtColor(img, cv.COLOR_BGR2GRAY)
t, binary=cv.threshold(gray, 0,255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
print(t)

cv.imshow("Color", img); cv.imshow("Gray", gray); cv.imshow("Binary", binary)
cv.waitKey(); cv.destroyAllWindows()
#%%
kernel=cv.getStructuringElement(cv.MORPH_RECT, (3,3))
fg=cv.erode(binary,kernel, iterations= 8)
cv.imshow("Foreground", fg); cv.waitKey()

bg=cv.dilate(binary, kernel, iterations=6)
cv.imshow("bg", bg); 
cv.waitKey()
t, bg=cv.threshold(bg, 1, 128, cv.THRESH_BINARY_INV)
cv.imshow("Background", bg); cv.waitKey()

markers=fg+bg
cv.imshow("Markers", markers)
cv.waitKey()

markers=np.int32(markers)
markers=cv.watershed(img, markers)
np.max(markers)
np.min(markers)
#%%
dst=np.ones(img.shape, np.uint8)*255
dst[markers==-1] = 0
cv.imshow("Watershed", dst)
img[markers==-1]=[255,0,0]
cv.imshow("Color&Boundry", img)
cv.waitKey()

markers=np.uint8(np.clip(markers, 0, 255))
cv.imshow("Segmentation", markers)
cv.waitKey() 

cv.destroyAllWindows()
