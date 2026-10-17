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

# 이진 영역 준비 [영역 특징 PPT 24쪽]
# bool 배열을 uint8의 0/255로 바꾸고 반전해 물체를 흰색으로 만든다.
# findContours는 0이 배경이고 0이 아닌 픽셀이 물체인 단일 채널 영상을 받는다.
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
# 외곽선으로 물체 표현 [PPT 3, 24쪽]
# RETR_EXTERNAL은 바깥 외곽선만, CHAIN_APPROX_NONE은 경계점을 생략하지 않고 저장한다.
# contours[0]이 가장 크거나 맨 위 물체라고 보장되지는 않는다. 반환 순서를 확인한다.
contours, hierarchy = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

img2=cv.cvtColor(img, cv.COLOR_GRAY2BGR)
cv.drawContours(img2, contours, -1, (255, 0, 255), 2)
cv.imshow("Contours", img2)
cv.waitKey()

# 모멘트와 모양 특징 [PPT 4~9, 14, 25쪽]
# m00은 외곽선이 둘러싼 면적, m10/m00과 m01/m00은 OpenCV 순서 (x, y)의 중심이다.
# 영상 배열 접근은 [y, x]이다. m00이 0인 퇴화 외곽선에서는 중심을 나눌 수 없다.
# arcLength는 점 사이 거리의 합으로 len(contour)와 다르다. 둥근 정도는 4*pi*면적/둘레^2.
# 중심 모멘트(mu)는 이동, 정규화 중심 모멘트(nu)는 스케일 영향을 줄인다.
# 회전도 다루는 Hu 모멘트는 별도 이론 문서 참고. 아래 코드는 Hu를 계산하지 않는다.
m=cv.moments(contours[0]) #dictionary
print(m)
area=cv.contourArea(contours[0]) # 면적
cx, cy = m['m10']/m['m00'], m['m01']/m['m00']
perimeter = cv.arcLength(contours[0], True)
roundness = 4.0*np.pi*area/(perimeter*perimeter)
print('면적= ', area, '/n중점=(', cx, cy, ')', '/n둘레= ', perimeter, '/n둥근정도= ', roundness)
#%%
img3=cv.cvtColor(img, cv.COLOR_GRAY2BGR)
# 다각형 근사 [PPT 12, 25쪽]
# epsilon=8은 원래 곡선과 근사 곡선 사이의 최대 허용 거리이다.
# 값을 키우면 보통 꼭짓점 수가 줄어든다. True는 닫힌 곡선이라는 뜻이다.
contour_approxP = cv.approxPolyDP(contours[0],8, True)
print(type(contour_approxP)); print(type([contour_approxP]));
print(len(contour_approxP))
cv.drawContours(img3, [contour_approxP], -1, (0, 255,0),2)

# 볼록 껍질 [PPT 13, 25쪽]
# 물체 전체를 고무줄로 감싼 모양으로 오목한 부분을 건너뛴다. 다각형 근사와 구분한다.
# 아래 reshape는 그리기 함수에 전달할 배열 구조를 조정하는 것이며 좌표를 바꾸지 않는다.
hull = cv.convexHull(contours[0])
print(hull.shape); print(hull);print(type(hull)) 
print(len(hull))
hull=hull.reshape(1, hull.shape[0], hull.shape[2])
cv.drawContours(img3, hull, -1, (0, 0, 255),2)

# 경계 상자 [PPT 10쪽]: 축에 평행한 최소 사각형 (x, y, 폭, 높이)을 반환한다.
rect=cv.boundingRect(contours[0])
print(rect);print(type(rect)) 
cv.rectangle(img3, rect, (255, 0,0), 2)

# 최소 둘러싼 원 [PPT 11쪽]: 모든 경계점을 포함하는 가장 작은 원이다.
# 반환한 중심/반지름은 실수이다. 기존 uint8 변환은 255 초과 값을 순환시킬 수 있어 int가 적절하다.
# 아래 원래 실행문은 유지했다. 안전한 좌표 변환은 시험 정리 문서에 따로 설명했다.
center, radius = cv.minEnclosingCircle(contours[0])
print('center= ', center, '\nRadius= ', radius)
cv.circle(img3, np.uint8(center), np.uint8(radius), (0, 255, 255), 2)


cv.imshow("Horse with Contours", img3)
cv.waitKey()

cv.destroyAllWindows()
