# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 11:27:12 2023

@author: BigData
"""

import cv2 as cv
import numpy as np
#%%

img=cv.imread('soccer.jpg')
gray=cv.cvtColor(img, cv.COLOR_BGR2GRAY)
canny=cv.Canny(gray, 100, 200)
#%%

# 에지 맵에서 연결된 경계 추적 [에지 검출_1 PPT 30~36쪽]
# RETR_LIST는 외곽선들을 계층 없이 모으고 CHAIN_APPROX_NONE은 모든 경계점을 보관한다.
# 반환점 좌표는 (x,y)이다. hierarchy의 항목 순서는 다음·이전·자식·부모이다.
contour, hierarchy = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

print(contour[0])
len(contour[0])

# 경계 필터링과 표시 [에지 검출_1 PPT 32쪽]
# shape[0]>100은 경계점 개수 조건이며 실제 둘레>100과는 다르다(둘레는 arcLength).
# drawContours의 -1은 전달된 모든 외곽선을 그리라는 뜻이다.
lcontour=[ ]
for i in range(len(contour)):
    if contour[i].shape[0] >100:
        lcontour.append(contour[i])
        
cv.drawContours(img, lcontour, -1, (0,255,0), 3)

cv.imshow('Original', img)
cv.waitKey()

cv.imshow('Canny', canny)
cv.waitKey()

cv.destroyAllWindows()
