# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 09:23:04 2023

@author: BigData
"""

import cv2 as cv
import numpy as np

img=cv.imread('soccer.jpg')
gray=cv.cvtColor(img, cv.COLOR_BGR2GRAY)

grad_x=cv.Sobel(gray, cv.CV_32F, 1, 0, ksize=3)
grad_y=cv.Sobel(gray, cv.CV_32F, 0, 1, ksize=3)

sobel_x=cv.convertScaleAbs(grad_x)
sobel_y=cv.convertScaleAbs(grad_y)

edge_strength=np.sqrt(grad_x**2+grad_y**2)
max_strength=np.max(edge_strength)
edge_strength=np.uint8(edge_strength/max_strength*255)

#%%

cv.imshow('Original', gray)
cv.waitKey()

cv.imshow('sobel_x', sobel_x)
cv.waitKey()

cv.imshow('sobel_y', sobel_y)

cv.waitKey()

cv.imshow('sobel_strength', edge_strength)

cv.waitKey()
cv.destroyAllWindows()
