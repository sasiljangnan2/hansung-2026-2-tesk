# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 11:20:02 2023

@author: BigData
"""

import cv2 as cv

img=cv.imread('soccer.jpg')

gray=cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Canny와 이중 임계값 [에지 검출_1 PPT 21~28쪽]
# 큰 임계값 이상에서 에지를 시작하고, 작은 임계값 이상인 연결된 약한 에지도 유지한다.
# 이론 흐름은 잡음 완화→미분→비최대 억제→이력 추적이다. 비최대 억제는 그레디언트 방향으로 비교한다.
# 이 파일은 GaussianBlur를 별도로 호출하지 않는다. 명시적 전처리는 food_edge.py에서 비교한다.
canny1=cv.Canny(gray, 50, 150)
canny2=cv.Canny(gray, 100, 200)

cv.imshow('Original', gray)
cv.waitKey()

cv.imshow('Canny1', canny1)
cv.waitKey()

cv.imshow('Canny2', canny2)


cv.waitKey()
cv.destroyAllWindows()