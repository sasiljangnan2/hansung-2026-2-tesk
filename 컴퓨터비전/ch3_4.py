# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 14:04:24 2023

@author: BigData
"""

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import sys

#%%
img = cv.imread("JohnHancocksSignature.png", cv.IMREAD_UNCHANGED)

imgc = cv.imread("JohnHancocksSignature.png")

if img is None:
    sys.exit("No File exists.")
    
print(type(img))

print(img.shape)
print(imgc.shape)
print(img)
print(imgc)
print(np.max(imgc))
print(np.max(img[:,:,3]))
print(np.min(img[:,:,3]))

cv.imshow("unchanged", img[:,:,3]) #graylevel image
cv.waitKey()
cv.imshow("color", imgc)
cv.waitKey()
cv.destroyAllWindows()
#%%
t, binimg=cv.threshold(img[:,:,3], 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
cv.imshow("Binary", binimg)
cv.waitKey()
cv.destroyAllWindows()
#%%
plt.imshow(binimg, cmap='gray') ;  plt.xticks([]) ; plt.yticks([])
plt.show()
b=binimg[binimg.shape[0]//2: binimg.shape[0],0: binimg.shape[0]//2+1]
plt.imshow(b, cmap='gray') ; plt.xticks([]) ; plt.yticks([])
plt.show()
#%%
se=np.uint8([[0,0,1,0,0],[0,1,1,1,0],
             [1,1,1,1,1], [0,1,1,1,0],[0,0,1,0,0]])
print(se.shape)
b_dilation = cv.dilate(b, se, iterations=1)
plt.imshow(b_dilation, cmap='gray'), plt.xticks([]) ; plt.yticks([])
plt.show()

b_eroision = cv.erode(b, se, iterations=1)
plt.imshow(b_eroision, cmap='gray'), plt.xticks([]) ; plt.yticks([])
plt.show()

b_closing = cv.erode(cv.dilate(b, se, iterations=1), se, iterations=1)
plt.imshow(b_closing, cmap='gray'), plt.xticks([]) ; plt.yticks([])
plt.show()



