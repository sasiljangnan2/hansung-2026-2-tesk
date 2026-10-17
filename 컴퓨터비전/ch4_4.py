# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 12:32:47 2023

@author: BigData
"""

# ==================================================================
# [선행 이론] 허프 투표·직선 표현·원 검출의 배경
# 근거: 4장 에지 검출_1 37~46쪽
# 읽는 순서: 선행 이론 → 관련 코드 앞의 서술형 요약 → 계산·출력 주석.
# PPT의 배경 이론과 현재 코드의 구현 범위를 구분하여 공부한다.
#
# 1. 점을 연결하지 않고 같은 모델을 지지하는지 모으는 발상
# 경계점이 가능한 직선 후보에 투표하고 많은 표가 모이는 후보를 선으로 선택한다.
# 일부 경계가 끊겨 있어도 여러 점이 같은 모델을 지지하면 찾을 수 있다. 무늬·잡음도 후보에 투표할 수 있다.
# y=a*x+b를 쓰면 한 영상 점 (x,y)이 b=-x*a+y라는 매개변수 공간의 직선이 된다.
# 같은 영상 직선 위 점들은 공통 a,b 위치에 모인다. 하지만 수직선은 기울기 a가 무한대라 표현이 어렵다.
# 이를 rho=x*cos(theta)+y*sin(theta)로 표현한다. theta는 직선의 접선이 아니라 법선 방향이다.
# 누적 공간을 유한 간격으로 나누므로 rho/theta 해상도와 투표 기준이 결과·계산량에 영향을 준다.
#
# 2. 허프 함수의 결과와 모델을 구분
# HoughLines는 (rho,theta)의 무한 직선, HoughLinesP는 (x1,y1,x2,y2)의 선분을 반환한다.
# minLineLength는 최소 선분 길이, maxLineGap은 한 선분으로 연결할 수 있는 틈이다.
# 원의 일반 매개변수는 (중심a,중심b,반지름r)이고 (x-a)**2+(y-b)**2=r**2를 만족한다.
# 직선의 두 매개변수와 원의 세 매개변수 공간을 혼동하지 않는다.
#
# 3. 현재 원 검출 예제의 설정
# HoughCircles의 HOUGH_GRADIENT는 기울기 정보를 이용하는 원 검출 방식이다.
# dp는 중심 누적 배열의 해상도 비율, minDist는 검출 중심 사이 최소 거리이다.
# param1은 내부 Canny의 높은 기준, param2는 원 중심 누적 기준이다. 반지름 범위도 제한한다.
# param2를 올리면 약한 후보는 줄어들 수 있지만 실제 원도 놓칠 수 있다. 크기와 단위를 함께 본다.
# apples.jpg가 없거나 검출 결과가 None이면 실제 숫자를 출력할 수 없다. 아래 형식 설명과 실행 확인을 구분한다.
# ==================================================================


import cv2 as cv
import numpy as np
#%%

img=cv.imread('apples.jpg')
cv.imshow('Apple', img)

cv.waitKey()
#%%
gray=cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# [서술형] 허프 변환을 이용한 원 분석
# 관련 PPT: 에지 검출_1 37~44쪽
# 특징: 경계점이 지지하는 중심·반지름 후보에 투표하여 원 모델을 찾는다.
# 장점: 원 전체가 완벽히 이어지지 않아도 여러 점의 지지를 모아 검출할 수 있다.
# 단점: 반지름·해상도·임계값에 민감하고 다른 곡선/잡음이 가짜 후보를 만들 수 있다.
# 활용: 사과처럼 원형인 대상의 중심과 반지름 검출.
# 한 문장 답안: 허프 원 검출은 부분 경계의 투표를 모을 수 있지만 대상 크기와 검출 임계값을 맞춰야 한다.
# [계산·코드 연결]
# 원 식은 (x-a)**2+(y-b)**2=r**2. 매개변수는 a,b,r 세 개이다.
# dp=누적 배열 대비 해상도 비율, minDist=중심 간 최소 거리, min/maxRadius=반지름 범위.
# HOUGH_GRADIENT의 param1=Canny 상한, param2=중심 누적 기준. None이면 뒤의 shape 접근은 오류.
# 현재 apples.jpg가 없어 실제 원 개수/좌표는 미확인. 출력 형식 주석과 구분한다.
apples=cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, 200, param1=150, param2=20, minRadius=50, maxRadius=120)
# [출력 형식: 원이 N개 검출되면] shape=(1,N,3), 각 원은 [중심x,중심y,반지름] 실수값이다.
# 예를 들어 원 2개라면 (1,2,3)이다. 이는 형식 설명이며 현재 영상에서 검출한 실제 N은 아니다.
# apples[0]에는 N개의 원이 들어 있다. len(apples[0])=N이지만 그 줄은 print가 없어 출력되지 않는다.
# 현재 apples.jpg가 없어 실제 숫자는 확인하지 못했다. 검출 실패 None에는 shape가 없어 오류가 난다.
print(apples.shape)
print(apples)
len(apples[0])
#%%

for i in apples[0]:
    cv.circle(img, (int(i[0]), int(i[1])), int(i[2]), (255,0,0), 2)
    
cv.imshow('Apple detection', img)

cv.waitKey()
cv.destroyAllWindows()

              
  