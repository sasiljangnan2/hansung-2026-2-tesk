# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 11:20:02 2023

@author: BigData
"""

import cv2 as cv

img=cv.imread('soccer.jpg')

gray=cv.cvtColor(img, cv.COLOR_BGR2GRAY)

canny1=cv.Canny(gray, 50, 150)
canny2=cv.Canny(gray, 100, 200)

cv.imshow('Original', gray)
cv.waitKey()

cv.imshow('Canny1', canny1)
cv.waitKey()

cv.imshow('Canny2', canny2)


cv.waitKey()
cv.destroyAllWindows()