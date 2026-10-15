# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 10:00:53 2023

@author: BigData
"""

import skimage
import cv2 as cv
import numpy as np
import sys
import math
#%%

org=skimage.data.horse()

if org is None:
    sys.exit(' No file')
    
print(org.shape); print(type(org)); print(np.dtype(org[0,0]))
print(np.max(np.uint8(org))); print(np.min(np.uint8(org)))

cv.imshow("Original", np.uint8(org)*255)
img= 255-np.uint8(org)*255
cv.imshow("Horse", img)
cv.waitKey()
cv.destroyAllWindows()
#%%
contours, hierarchy = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

img2=cv.cvtColor(img, cv.COLOR_GRAY2BGR)
cv.drawContours(img2, contours, -1, (255, 0, 255), 2)
cv.imshow("Contours", img2)
cv.waitKey()

m=cv.moments(contours[0]) #dictionary
print(m)
area=cv.contourArea(contours[0]) # 면적
cx, cy = m['m10']/m['m00'], m['m01']/m['m00']
perimeter = cv.arcLength(contours[0], True)
roundness = 4.0*np.pi*area/(perimeter*perimeter)
print('면적= ', area, '/n중점=(', cx, cy, ')', '/n둘레= ', perimeter, '/n둥근정도= ', roundness)
#%%
img3=cv.cvtColor(img, cv.COLOR_GRAY2BGR)
contour_approxP = cv.approxPolyDP(contours[0],8, True)
print(type(contour_approxP)); print(type([contour_approxP]));
print(len(contour_approxP))
cv.drawContours(img3, [contour_approxP], -1, (0, 255,0),2)

hull = cv.convexHull(contours[0])
print(hull.shape); print(hull);print(type(hull)) 
print(len(hull))
hull=hull.reshape(1, hull.shape[0], hull.shape[2])
cv.drawContours(img3, hull, -1, (0, 0, 255),2)

rect=cv.boundingRect(contours[0])
print(rect);print(type(rect)) 
cv.rectangle(img3, rect, (255, 0,0), 2)

center, radius = cv.minEnclosingCircle(contours[0])
print('center= ', center, '\nRadius= ', radius)
cv.circle(img3, np.uint8(center), np.uint8(radius), (0, 255, 255), 2)


cv.imshow("Horse with Contours", img3)
cv.waitKey()

cv.destroyAllWindows()
