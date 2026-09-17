# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 20:54:33 2023

@author: BigData
"""

import cv2 as cv
import sys
#%%
img=cv.imread('soccer.jpg')\

if img is None:
    sys.exit("No File exists.")
#%%    
#t, bin = cv.threshold(img[:,:,2], 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
t, bin = cv.threshold(img[:,:,2], 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

print("threshold", t)
cv.imshow('R', img[:,:,2])
cv.imshow('R binary', bin)

cv.waitKey()
cv.destroyAllWindows()
