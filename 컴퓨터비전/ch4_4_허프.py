# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 12:32:47 2023

@author: BigData
"""

#%% 개념 요약 — 정의·장점·단점, 함수 사용법은 해당 코드 옆 참고

#%% [개념] 허프 변환을 이용한 원 분석
# 정의: 경계점이 지지하는 원의 중심·반지름 후보에 투표하여 원을 찾는 방법이다.
# 장점: 원 전체가 완벽히 이어지지 않아도 여러 점의 지지를 모아 검출할 수 있다.
# 단점: 반지름·해상도·임계값에 민감하고 다른 곡선/잡음이 가짜 후보를 만들 수 있다.
# PPT: 에지 검출_1 37~44쪽

#region 핵심 이론·함수 참고
# [핵심 이론] 허프 투표·직선 표현·원 검출의 배경
# 근거: 4장 에지 검출_1 37~46쪽
# 1. 점을 연결하지 않고 같은 모델을 지지하는지 모으는 발상
# 경계점이 가능한 직선 후보에 투표하고 많은 표가 모이는 후보를 선으로 선택한다.
# y=a*x+b를 쓰면 한 영상 점 (x,y)이 b=-x*a+y라는 매개변수 공간의 직선이 된다.
# 이를 rho=x*cos(theta)+y*sin(theta)로 표현한다. theta는 직선의 접선이 아니라 법선 방향이다.
# 2. 허프 함수의 결과와 모델을 구분
# HoughLines는 (rho,theta)의 무한 직선, HoughLinesP는 (x1,y1,x2,y2)의 선분을 반환한다.
# 원의 일반 매개변수는 (중심a,중심b,반지름r)이고 (x-a)**2+(y-b)**2=r**2를 만족한다.
# 3. 현재 원 검출 예제의 설정
# HoughCircles의 HOUGH_GRADIENT는 기울기 정보를 이용하는 원 검출 방식이다.
# apples.jpg가 없거나 검출 결과가 None이면 실제 숫자를 출력할 수 없다. 아래 형식 설명과 실행 확인을 구분한다.


#%% [PPT 함수 보충] 일반·확률 허프 직선과 원 비교 (현재 원 검출 실행 유지)
# 관련 PPT: 에지 검출 37~50쪽. 아래 직선 예시는 주석이며 현재 원 검출 실행에 섞지 않는다.
# [함수 사용] lines=cv.HoughLines(edge,rho,theta,threshold,srn=0,stn=0,min_theta=0,max_theta=np.pi)
# edge=uint8 이진 에지, rho=거리 해상도(픽셀), theta=각도 해상도(라디안), threshold=최소 누적 투표 기준.
# 주석 예: lines=cv.HoughLines(edge,1,np.pi/180,90)
# 기본 반환은 N개의 (rho,theta), 보통 shape=(N,1,2). 검출이 없으면 None이다.
# rho=x*cos(theta)+y*sin(theta). theta는 직선 자체의 접선이 아니라 법선 방향이며 수직선도 유한하게 표현한다.
# srn/stn=0이면 표준 허프, 양수이면 거리/각도 해상도를 더 세분하는 다중 스케일 방식. min/max_theta는 탐색 각도 범위이다.
# 선 그리기: (x0,y0)=(rho*cosθ,rho*sinθ), 그 점에서 접선 (-sinθ,cosθ)의 양쪽으로 L만큼 연장해 끝점을 만든다.
# [확률 허프 비교] cv.HoughLinesP(edge,1,np.pi/180,90,minLineLength=100,maxLineGap=10)
# 반환은 끝점 (x1,y1,x2,y2), 보통 (N,1,4). minLineLength=최소 길이, maxLineGap=연결 허용 간격이다.
# Python의 다섯 번째 위치 인자는 lines 출력 버퍼이다. PPT 예시의 90,100,10 형태를 길이·간격으로 옮길 때는 위 키워드를 사용한다.
# [PPT 샘플 입력] img=skimage.data.checkerboard(): (200,200) uint8 명암 체커보드. 별도 사진 없이 직선 검출을 연습할 수 있다.
# [원 검출] HOUGH_GRADIENT의 dp는 입력에 대한 누적 배열 해상도 역비율(1은 동일), minDist는 원 중심 사이 최소 거리.
# dp를 키우면 중심 해상도가 거칠어질 수 있고, minDist가 너무 작으면 중복 원, 너무 크면 가까운 원 누락이 생긴다.

#endregion

import cv2 as cv
from cv_assets import read_image
import numpy as np
#%%

# [함수 사용: 입출력·창]
# img=cv.imread(path, flags=cv.IMREAD_COLOR): 파일→배열. 기본 BGR, 읽기 실패 시 None인지 확인한다.
# cv.imshow(winname,img): 이름이 winname인 창에 배열을 표시한다. 저장 함수는 아니다.
# key=cv.waitKey(delay=0): GUI 이벤트·키 처리. 0은 계속 대기, 양수는 ms, 입력 없으면 보통 -1이다.
# cv.destroyAllWindows(): OpenCV 창을 닫는다. 캡처 자원 해제는 cap.release()이다.
img=read_image('apples.jpg')
cv.imshow('Apple', img)

cv.waitKey()
#%%
# [함수 사용] dst=cv.cvtColor(src, code)
# src=입력 영상, code=색 변환 상수. 예: BGR2GRAY·BGR2HSV·HSV2BGR. 변환 영상을 반환한다.
# [PPT 인자 보충] 영상처리_1 16쪽: dstCn=0은 출력 채널 수 자동 결정. 출력 높이·너비와 깊이는 같고 채널 수는 변환에 따라 달라진다.
gray=cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# 원 식은 (x-a)**2+(y-b)**2=r**2. 매개변수는 a,b,r 세 개이다.
# dp=누적 배열 대비 해상도 비율, minDist=중심 간 최소 거리, min/maxRadius=반지름 범위.
# HOUGH_GRADIENT의 param1=Canny 상한, param2=중심 누적 기준. None이면 뒤의 shape 접근은 오류.
# 현재 apples.jpg가 없어 실제 원 개수/좌표는 미확인. 출력 형식 주석과 구분한다.
# [함수 사용] circles=cv.HoughCircles(gray, method, dp, minDist, param1=100, param2=100, minRadius=0, maxRadius=0)
# dp=누적판 해상도 비, minDist=중심 간격, param1/2=방법별 기준, min/maxRadius=반지름 범위. GRADIENT 결과는 (x,y,r)이며 없으면 None이다.
# [PPT 인자 보충] 에지 검출 44쪽: HOUGH_GRADIENT에서 param1은 내부 Canny의 높은 T(낮은 T는 절반), param2는 중심 누적 임계값. minRadius/maxRadius는 반지름 범위이다.
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
    # [함수 사용] cv.circle(img, center, radius, color, thickness=1)
    # center=(x,y), radius=반지름, color=BGR 색. img에 직접 그리며 thickness=-1이면 원 내부를 채운다.
    # [PPT 인자 보충] OpenCV 기본 37쪽: lineType=LINE_4/LINE_8/LINE_AA는 선 연결·안티앨리어싱 방식. shift는 좌표의 소수 비트 수이며 thickness<0이면 채운다.
    cv.circle(img, (int(i[0]), int(i[1])), int(i[2]), (255,0,0), 2)
    
cv.imshow('Apple detection', img)

cv.waitKey()
cv.destroyAllWindows()

              
  
