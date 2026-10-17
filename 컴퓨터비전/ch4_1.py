# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 09:23:04 2023

@author: BigData
"""

import cv2 as cv
import numpy as np

img=cv.imread('soccer.jpg')
gray=cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# 미분 영상 [지역특징 PPT 14~20쪽과 연결]
# 해리스도 같은 dx, dy에서 출발한다. CV_32F로 음수 미분값을 보존한다.
# 구조 행렬은 dx^2, dy^2, dx*dy를 이웃에서 모아 만든다. 실제 해리스 계산은 ch5_1.py 참고.
# 이 파일 자체는 Sobel 에지까지만 계산한다.
# Sobel의 방향별 1차 미분 [에지 검출_1 PPT 5~14쪽]
# dx=1,dy=0은 x 변화, dx=0,dy=1은 y 변화를 구한다. 영상 y축은 아래쪽이 양수이다.
# signed float 결과로 부호를 보존한 뒤 강도 sqrt(dx^2+dy^2)를 계산한다.
grad_x=cv.Sobel(gray, cv.CV_32F, 1, 0, ksize=3)
grad_y=cv.Sobel(gray, cv.CV_32F, 0, 1, ksize=3)

# 절댓값/uint8 변환은 화면 표시용이다. 해리스 계산에는 부호가 남은 grad_x, grad_y가 필요하다.
sobel_x=cv.convertScaleAbs(grad_x)
sobel_y=cv.convertScaleAbs(grad_y)

edge_strength=np.sqrt(grad_x**2+grad_y**2)
# 표시용 범위 변환 [에지 검출_1 PPT 15~16쪽과 비교]
# 현재 코드는 절댓값 두 장의 가중합 대신 유클리드 강도를 최댓값으로 나눠 표시한다.
# 정규화된 화면값과 원래 미분 강도는 다르다. 상수 영상에서는 최댓값이 0인지 확인이 필요하다.
max_strength=np.max(edge_strength)
edge_strength=np.uint8(edge_strength/max_strength*255)

#%%

cv.imshow('Original', gray)
cv.waitKey()

cv.imshow('sobel_x', sobel_x)
cv.waitKey()

cv.imshow('sobel_y', sobel_y)

cv.waitKey()

cv.imshow('sobel_strength', edge_strength)

cv.waitKey()
cv.destroyAllWindows()
