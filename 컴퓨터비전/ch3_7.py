# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 13:56:50 2023

@author: BigData
"""

import cv2 as cv
import numpy as np
import sys

#%%
img = cv.imread("soccer.jpg")

if img is None:
    sys.exit("No File exists.")
 #%%   
a=np.array([-3,-2,-1,0,254,255,256,257,258], dtype=np.int16).astype(np.uint8)
print(type(a))
print(type(a[0]))
print(a)

#%%
print(type(img))
print(type(img[0,0,0]))

#%%
img=cv.resize(img, dsize=(0,0), fx=0.4, fy=0.4)
gray=cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.putText(gray, 'soccer', (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255),2 )
cv.imshow('Original', gray)
cv.waitKey()

#%%
# 가우시안 가중치 [지역특징 PPT 14~18, 28쪽과 연결]
# 가까운 이웃에 큰 가중치를 주어 잡음을 완화한다. 해리스는 미분값의 곱을 모을 때 사용한다.
# ch5_1.py의 g도 같은 원리지만, 이 코드는 원영상 평활화 결과를 비교한다.
# 가우시안 영역 연산 [영상처리_2 PPT 23~27쪽]
# 이웃에 가우시안 가중치를 적용해 잡음을 완화한다. 큰 범위로 평균하면 세부 경계도 흐려진다.
# sigma=0은 커널 크기로 표준편차를 자동 추정한다. 메디안은 이웃의 중앙값을 고르는 비선형 필터이다.
smooth=np.hstack((cv.GaussianBlur(gray, (5,5), 0.0), 
                 cv.GaussianBlur(gray, (9,9), 0.0), cv.GaussianBlur(gray, (15,15), 0.0))) 
cv.imshow('Smooth', smooth)
cv.waitKey()
#%%Embossing

femboss=np.array([[-1.0, 0.0, 0.0], [0.0,0.0,0.0], [0.0,0.0,1.0]])
# 자료형 확인 [영상처리_2 PPT 28~31쪽]
# PPT 예제는 부호 있는 int16인데 현재 파일은 uint16이다. ddepth=-1이면 이 자료형을 유지하여
# 음수 필터 응답이 0으로 잘린다. 뒤의 +128이나 clip으로 이미 잃은 음수를 되살릴 수는 없다.
# 원래 실행문은 유지했다. uint8 단순 형변환의 순환과 clip의 범위 제한도 구분한다.
gray16=np.uint16(gray)
cv.waitKey()
cv.destroyAllWindows()

# filter2D 활용 [지역특징 PPT 29쪽]
# 같은 함수라도 커널에 따라 엠보싱/미분/가중 합이 된다. 여기서는 엠보싱 커널이다.
# 해리스 예제는 미분/가우시안 커널을 넣고 결과를 CV_32F로 보존한다.
emboss=np.uint8(np.clip(cv.filter2D(gray16, -1, femboss)+128, 0, 255))
emboss_bad=np.uint8(cv.filter2D(gray16, -1, femboss)+128)
emboss_worse=cv.filter2D(gray, -1, femboss)

cv.imshow('Emboss', emboss)
cv.imshow('Emboss_bad', emboss_bad)
cv.imshow('Emboss_worse', emboss_worse)

cv.waitKey()
cv.destroyAllWindows()
