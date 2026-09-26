# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 13:56:50 2023

@author: BigData
"""

import cv2 as cv
import numpy as np
import sys

#%%
img = cv.imread("soccer.jpg")

if img is None:
    sys.exit("No File exists.")
 #%%   
a=np.array([-3,-2,-1,0,254,255,256,257,258], dtype=np.int16).astype(np.uint8)
print(type(a))
print(type(a[0]))
print(a)

#%%
print(type(img))
print(type(img[0,0,0]))

#%%
img=cv.resize(img, dsize=(0,0), fx=0.4, fy=0.4)
gray=cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.putText(gray, 'soccer', (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255),2 )
cv.imshow('Original', gray)
cv.waitKey()

#%%
smooth=np.hstack((cv.GaussianBlur(gray, (5,5), 0.0), 
                 cv.GaussianBlur(gray, (9,9), 0.0), cv.GaussianBlur(gray, (15,15), 0.0))) 
cv.imshow('Smooth', smooth)
cv.waitKey()
#%%Embossing

femboss=np.array([[-1.0, 0.0, 0.0], [0.0,0.0,0.0], [0.0,0.0,1.0]])
gray16=np.uint16(gray)
cv.waitKey()
cv.destroyAllWindows()

emboss=np.uint8(np.clip(cv.filter2D(gray16, -1, femboss)+128, 0, 255))
emboss_bad=np.uint8(cv.filter2D(gray16, -1, femboss)+128)
emboss_worse=cv.filter2D(gray, -1, femboss)

cv.imshow('Emboss', emboss)
cv.imshow('Emboss_bad', emboss_bad)
cv.imshow('Emboss_worse', emboss_worse)

cv.waitKey()
cv.destroyAllWindows()
