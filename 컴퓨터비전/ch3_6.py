# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 12:16:58 2023

@author: BigData
"""

import cv2 as cv
import matplotlib.pyplot as plt
import sys
#%%

img = cv.imread("mistyroad.jpg")

if img is None:
    sys.exit("No File exists.")
    
cv.imshow('Original', img)
cv.waitKey()

print(img.shape)
#%%
gray=cv.cvtColor(img, cv.COLOR_BGR2GRAY)
print(gray.shape)

plt.imshow(gray, cmap='gray'), plt.xticks([]), plt.yticks([])
#%%
h=cv.calcHist([gray], [0], None, [256], [0, 255])
plt.plot(h, color='r', linewidth=1) 
plt.show()
#%%
equal=cv.equalizeHist(gray)
plt.imshow(equal, cmap='gray'), plt.xticks([]), plt.yticks([])
cv.imshow('Equal', equal)
cv.waitKey()
cv.destroyAllWindows()
#%%
h=cv.calcHist([equal], [0], None, [256], [0, 255])
plt.plot(h, color='r', linewidth=1) 
plt.show()
