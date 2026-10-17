# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 11:04:25 2023

@author: admin
"""
import cv2 as cv
import numpy as np
#%%
# 실습 입력 [영역 특징 PPT 27쪽]
# 외곽선 좌표는 binaryGroup.bmp에서 얻는다. 원본에 그릴 때 두 영상의 크기가 같아야 한다.
bin=cv.imread('binaryGroup.bmp',0)
img=cv.imread('group.jpg')

cv.imshow("binary", bin); cv.imshow("Original", img)
cv.waitKey()

# 외곽선 추출 [PPT 3, 24쪽]
# RETR_EXTERNAL은 바깥 경계만 반환하고 내부 구멍을 별도 외곽선으로 반환하지 않는다.
# CHAIN_APPROX_NONE은 모든 경계점을 남긴다. 반환 순서가 위에서부터라고 가정하지 않는다.
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

# 작은 잡음과 큰 물체 제외 [PPT 27쪽]
# 아래 기존 코드는 경계점 개수로 고른다. len(contour)는 점 개수, arcLength는 둘레 길이이다.
# PPT의 '둘레 50 이하/1000 이상 제외'를 정확히 구현하려면 arcLength로 판단해야 한다.
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
# 모멘트와 중심 [PPT 4, 14쪽]
# OpenCV의 m10은 x좌표, m01은 y좌표에 대한 합이다. m00은 면적이다.
# m00이 0이면 중심을 나눌 수 없으므로 퇴화한 외곽선은 별도 확인이 필요하다.
m=cv.moments(lcontours[0])
print(cv.arcLength(lcontours[0], True))
print("Area:", cv.contourArea(lcontours[0])); 
print("center of x, y", m['m10']/m['m00'], m['m01']/m['m00'])
print("mom_Area", m['m00']); print("sum of x, y", m['m10'], m['m01'])

# 경계 상자 [PPT 10쪽]: (x, y, 폭, 높이)를 구해 축에 평행한 사각형으로 표현한다.
# 위 모멘트는 필터링한 lcontours[0], 아래 상자는 원래 contours[0]이므로 다른 물체일 수 있다.
# 기존 실행문을 유지하고 차이를 학습 주석으로 표시했다.
r0=cv.boundingRect(contours[0])
cv.rectangle(img4, r0, (0,0,255),2)

cv.imshow("Draw_Contours", img4)
cv.waitKey()

# 실습 연결 [PPT 11~13, 27쪽]
# 남은 외곽선에 minEnclosingCircle, approxPolyDP, convexHull을 적용하는 부분은 아직 빈칸이다.
# ch4_8.py에 각 함수 예제가 있다. 이 파일은 완성 답안이 아니라 기존 실습 틀이다.
## 다른 contour 이 밑에 그려 볼 것

##실습

#%%
cv.imshow("Draw on Contour", img4)
cv.waitKey()
cv.destroyAllWindows()

#%%
# thickness=-1은 외곽선 내부를 칠한다. 모멘트 계산이나 볼록 껍질 계산과는 다른 표시 작업이다.
cv.drawContours(img5, lcontours, -1, 255, thickness=-1)
cv.imshow("Draw_Contours_filled", img5)
cv.waitKey()
cv.destroyAllWindows()