# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 15:54:21 2023

@author: BigData
"""

import cv2 as cv
import sys

#%%
img=cv.imread('soccer.jpg')

if img is None:
    sys.exit("No File exists.")

print(type(img))
print(img.shape)
type(img)
#pixel elements
print(img[0,0,0], img[0,0,1], img[0,0,2])   
print(img[0,1,0], img[0,1,1], img[0,1,2])   
print(img[0:10,0:10,0], img[0:10,0:10,1], img[0:10,0:10,2])   

cv.imshow('Img Display', img)
cv.imshow("Partial", img[0:10,0:10,:])

cv.waitKey()
cv.destroyAllWindows()
