# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 11:04:25 2023

@author: admin
"""
import cv2 as cv
import numpy as np
#%%
bin=cv.imread('binaryGroup.bmp',0)
img=cv.imread('group.jpg')

cv.imshow("binary", bin); cv.imshow("Original", img)
cv.waitKey()

contours, hierarchy=cv.findContours(bin, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

img1=cv.resize(img, (0,0), fx=0.7, fy=0.7);
img2=cv.cvtColor(bin, cv.COLOR_GRAY2BGR)
img3=img1.copy()
img4=img1.copy()
img5=img1.copy()

cv.drawContours(img2, contours, -1, (255, 0, 255), 2)
cv.drawContours(img1, contours, -1, (255, 0, 255), 2)
cv.imshow("Contours_gray", img2)
cv.imshow("Contours_original", img1)
cv.waitKey()

lcontours=[]
for i in range(len(contours)):
    if len(contours[i])>50 and len(contours[i])<1000:
        lcontours.append(contours[i])
        
print(len(lcontours)) 
cv.drawContours(img3, lcontours, -1, (0, 255, 255), 2)
cv.imshow("Contours_animal", img3)

cv.waitKey()
cv.destroyAllWindows()

#%% use img4
#1. moment, center
#2. contour length, contour area, roundness
#3 bounding box(contours[0]), minimum enclosing circle(contours[1]), 
#polylines(contours[2]), convex Hull(contours[3]) on each contour 
#%%
#1. moment, center
m=cv.moments(lcontours[0])
print(cv.arcLength(lcontours[0], True))
print("Area:", cv.contourArea(lcontours[0])); 
print("center of x, y", m['m10']/m['m00'], m['m01']/m['m00'])
print("mom_Area", m['m00']); print("sum of x, y", m['m10'], m['m01'])

r0=cv.boundingRect(contours[0])
cv.rectangle(img4, r0, (0,0,255),2)

cv.imshow("Draw_Contours", img4)
cv.waitKey()

## 다른 contour 이 밑에 그려 볼 것

##실습

#%%
cv.imshow("Draw on Contour", img4)
cv.waitKey()
cv.destroyAllWindows()

#%%
cv.drawContours(img5, lcontours, -1, 255, thickness=-1)
cv.imshow("Draw_Contours_filled", img5)
cv.waitKey()
cv.destroyAllWindows()