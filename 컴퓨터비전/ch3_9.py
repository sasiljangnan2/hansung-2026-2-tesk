# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 16:38:26 2023

@author: BigData
"""

import cv2 as cv
import numpy as np
import sys
import time

#%%

def my_cvtGray1(bar_img):
    g = np.zeros([bar_img.shape[0], bar_img.shape[1]])
    for r in range(bar_img.shape[0]):
        for c in range(bar_img.shape[1]):
            g[r,c]= 0.114*bar_img[r,c,0]+0.587*bar_img[r,c,1]+0.299*bar_img[r,c,2]
    return np.uint8(g)

def my_cvtGray2(bar_img):
    g = np.zeros([bar_img.shape[0], bar_img.shape[1]])
    g = 0.114*bar_img[:,:,0]+0.587*bar_img[:,:,1]+0.299*bar_img[:,:,2]
    return np.uint8(g)
        
#%%
img = cv.imread("girl_laughing.jpg")
if img is None:
    sys.exit("No File exists.")

start=time.time()
my_cvtGray1(img)
print('My time1: ', time.time()-start)

start=time.time()
my_cvtGray2(img)
print('My time2: ', time.time()-start)

start=time.time()
cv.cvtColor(img, cv.COLOR_BGR2GRAY)
print('MOpenCV time1: ', time.time()-start)
