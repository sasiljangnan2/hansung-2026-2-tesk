# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 12:32:47 2023

@author: BigData
"""

import cv2 as cv
import numpy as np
#%%

img=cv.imread('apples.jpg')
cv.imshow('Apple', img)

cv.waitKey()
#%%
gray=cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# 원 허프 변환 [에지 검출_1 PPT 39, 43~44쪽]
# 결과는 중심 x,y와 반지름이다. dp=1은 누적 배열과 원본의 해상도 비율, minDist=200은 중심 간 최소 거리다.
# HOUGH_GRADIENT의 param1은 Canny 상한, param2는 중심 누적 임계값이다. 반지름 범위도 후보를 제한한다.
# apples.jpg가 필요하고, 원이 검출되지 않으면 반환값 None을 확인한 뒤 접근해야 한다.
apples=cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, 200, param1=150, param2=20, minRadius=50, maxRadius=120)
print(apples.shape)
print(apples)
len(apples[0])
#%%

for i in apples[0]:
    cv.circle(img, (int(i[0]), int(i[1])), int(i[2]), (255,0,0), 2)
    
cv.imshow('Apple detection', img)

cv.waitKey()
cv.destroyAllWindows()

              
  