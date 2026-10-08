# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 12:32:47 2023

@author: BigData
"""

import cv2 as cv
import numpy as np
#%%

img=cv.imread('apples.jpg')
cv.imshow('Apple', img)

cv.waitKey()
#%%
gray=cv.cvtColor(img, cv.COLOR_BGR2GRAY)

apples=cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, 200, param1=150, param2=20, minRadius=50, maxRadius=120)
print(apples.shape)
print(apples)
len(apples[0])
#%%

for i in apples[0]:
    cv.circle(img, (int(i[0]), int(i[1])), int(i[2]), (255,0,0), 2)
    
cv.imshow('Apple detection', img)

cv.waitKey()
cv.destroyAllWindows()

              
  