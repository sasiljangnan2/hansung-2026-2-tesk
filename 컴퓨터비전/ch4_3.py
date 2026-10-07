# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 11:27:12 2023

@author: BigData
"""

import cv2 as cv
import numpy as np
#%%

img=cv.imread('soccer.jpg')
gray=cv.cvtColor(img, cv.COLOR_BGR2GRAY)
canny=cv.Canny(gray, 100, 200)
#%%

contour, hierarchy = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

print(contour[0])
len(contour[0])

lcontour=[ ]
for i in range(len(contour)):
    if contour[i].shape[0] >100:
        lcontour.append(contour[i])
        
cv.drawContours(img, lcontour, -1, (0,255,0), 3)

cv.imshow('Original', img)
cv.waitKey()

cv.imshow('Canny', canny)
cv.waitKey()

cv.destroyAllWindows()
