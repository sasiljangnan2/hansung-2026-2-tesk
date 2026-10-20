# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 09:23:04 2023

@author: BigData
"""

# ==================================================================
# [선행 이론] 미분·에지·그레디언트의 배경
# 근거: 4장 에지 검출_1 2,4~20쪽
# 읽는 순서: 선행 이론 → 관련 코드 앞의 서술형 요약 → 계산·출력 주석.
# PPT의 배경 이론과 현재 코드의 구현 범위를 구분하여 공부한다.
#
# 1. 에지와 영역은 같은 영상을 다른 관점으로 본다
# 에지는 명암·색·텍스처가 급격히 달라지는 곳, 영역은 특성이 비슷한 픽셀을 묶은 집합이다.
# 물체 내부보다 경계에서 변화가 크다는 가정으로 에지를 찾지만 그림자·질감도 큰 변화를 만든다.
# 따라서 에지 검출 결과가 의미상 물체의 외곽선 정답과 같다고 보장되지 않는다.
# 사람은 물체 모델·문맥을 이용하지만 여기의 고전적 연산은 픽셀 특성만으로 처리한다.
#
# 2. 디지털 미분과 경계 모델
# 1차 차분은 f(x+1)-f(x), 중심 차분은 (f(x+1)-f(x-1))/2로 생각할 수 있다.
# 2차 차분은 f(x-1)-2*f(x)+f(x+1). 1차 응답의 큰 값과 2차 응답의 영교차를 구분한다.
# 이상적인 계단 경계와 실제 여러 픽셀에 걸친 램프 경계는 응답의 폭·형태가 다르다.
# 미분은 잡음 변화도 크게 만들므로 평활화와 함께 고려한다.
#
# 3. 크기와 방향
# 그레디언트=(dx,dy), 유클리드 강도=sqrt(dx**2+dy**2), 방향=atan2(dy,dx)이다.
# 그레디언트는 밝기가 증가하는 법선 방향이고 경계 접선은 이에 수직이다.
# dx=3,dy=4면 크기는 5, L1 방식 abs(dx)+abs(dy)는 7로 서로 다르다.
# 영상은 y가 아래로 증가한다. 각도의 축과 단위를 확인하고 180도 주기의 경계 방향과 구별한다.
# 음수 미분을 unsigned로 잘라 버리면 방향 정보를 잃으므로 표시용 절댓값과 계산용 미분을 분리한다.
#
# 4. 연산자와 표시의 차이
# Prewitt와 Sobel은 x/y 방향 미분 커널이다. Sobel은 가운데 행/열에 2 가중치를 사용한다.
# x 미분이 크다는 것은 좌우 밝기 차이가 커 세로 방향 경계가 보일 수 있다는 뜻이다.
# Laplacian은 x/y 2차 미분을 합한 스칼라 응답으로 방향 벡터를 그대로 주지는 않는다.
# convertScaleAbs는 절댓값·스케일·포화 변환으로 화면에 보이게 한다. 원래 signed 미분은 남겨야 한다.
# 현재 sqrt 강도를 최댓값으로 나누는 표시는 원래 강도값 자체가 아니다. 최대가 0이면 나눗셈도 확인해야 한다.
# ==================================================================



#%% [PPT 함수 보충] 라플라시안·극좌표 변환 (현재 실행 없음)
# 관련 PPT: 에지 검출 6~8·17·20쪽. 기존 Sobel 실행문에 아래 호출을 추가한 것은 아니다.
# [함수 사용] dst=cv.Laplacian(src,ddepth,ksize=1,scale=1,delta=0,borderType=cv.BORDER_DEFAULT)
# 2차 미분 Ixx+Iyy를 계산한다. ddepth=CV_32F 등으로 부호를 보존하며 ksize는 양의 홀수이다.
# 주석 예: lap=cv.Laplacian(gray,cv.CV_32F,ksize=3)
# 평탄한 영역은 거의 0, 급격한 변화 주변은 양/음 반응. 영교차는 부호가 바뀌는 위치이며 단순히 값이 큰 곳만 고르는 Sobel과 구분한다.
# 2차 미분도 잡음에 민감하므로 평활화와 함께 사용한다. 라플라시안 값만으로 물체 의미를 인식하지는 않는다.
#
# [함수 사용] magnitude,angle=cv.cartToPolar(dx,dy,angleInDegrees=False)
# 같은 크기·깊이의 float32/float64 x,y 성분에서 크기 sqrt(dx²+dy²)와 방향 atan2(dy,dx)를 계산한다.
# 기본 방향은 0~2π 라디안, angleInDegrees=True면 0~360도이다. (0,0)의 방향은 0으로 둔다.
# 주석 예: mag,ang=cv.cartToPolar(np.array([[3]],np.float32),np.array([[4]],np.float32),angleInDegrees=True)
# 예상 mag=[[5.]], ang≈[[53.13]]. 방향 계산은 근사 오차가 있고, 0/360도 경계는 같은 방향이다.
# PPT의 angleInDegree 표기와 달리 Python 키워드는 복수형 angleInDegrees이다.

import cv2 as cv
import numpy as np

# [함수 사용: 입출력·창]
# img=cv.imread(path, flags=cv.IMREAD_COLOR): 파일→배열. 기본 BGR, 읽기 실패 시 None인지 확인한다.
# cv.imshow(winname,img): 이름이 winname인 창에 배열을 표시한다. 저장 함수는 아니다.
# key=cv.waitKey(delay=0): GUI 이벤트·키 처리. 0은 계속 대기, 양수는 ms, 입력 없으면 보통 -1이다.
# cv.destroyAllWindows(): OpenCV 창을 닫는다. 캡처 자원 해제는 cap.release()이다.
img=cv.imread('soccer.jpg')
# [함수 사용] dst=cv.cvtColor(src, code)
# src=입력 영상, code=색 변환 상수. 예: BGR2GRAY·BGR2HSV·HSV2BGR. 변환 영상을 반환한다.
# [PPT 인자 보충] 영상처리_1 16쪽: dstCn=0은 출력 채널 수 자동 결정. 출력 높이·너비와 깊이는 같고 채널 수는 변환에 따라 달라진다.
gray=cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# [서술형] Sobel 1차 미분 에지 분석
# 관련 PPT: 에지 검출_1 5~14쪽
# 특징: 차분과 수직 방향의 가중 평균으로 x/y 미분을 구해 밝기 변화의 크기·방향을 표현한다.
# 장점: 작은 커널로 간단히 계산하고 부호 있는 x/y 응답에서 방향을 구할 수 있다.
# 단점: 잡음·그림자·질감도 경계로 반응하며 얇게 정리된 에지나 물체 의미를 직접 주지 않는다.
# 활용: 기울기 크기/방향 측정, Canny와 Harris의 기초 미분, 에지 기반 특징 계산.
# 한 문장 답안: Sobel은 밝기 변화의 크기와 방향을 간단히 구하지만 잡음에 반응하고 경계 정리가 추가로 필요하다.
# [계산·코드 연결]
# x 커널=[[-1,0,1],[-2,0,2],[-1,0,1]], y 커널=[[-1,-2,-1],[0,0,0],[1,2,1]].
# 일정 밝기 패치는 커널 합이 0이라 응답 0. 오른쪽으로 밝아지면 dx+, 아래로 밝아지면 dy+.
# CV_32F는 음수 보존. gray/미분 shape=(948,1434), 미분 float32, 표시 절댓값은 uint8.
# Harris는 같은 미분의 곱을 모은다. 실제 특징점 계산은 ch5_1_해리스직접.py에서 한다.
# [함수 사용] dst=cv.Sobel(src, ddepth, dx, dy, ksize=3)
# dx·dy=각 축의 미분 차수, ksize=커널 크기. 예: CV_32F,1,0은 음수를 보존한 x 미분 결과이다.
# [PPT 인자 보충] 에지 검출 14쪽: dx/dy는 미분 차수. ksize는 보통 1,3,5,7, scale=1은 배율, delta=0은 더할 값, borderType은 가장자리 처리이다.
grad_x=cv.Sobel(gray, cv.CV_32F, 1, 0, ksize=3)
grad_y=cv.Sobel(gray, cv.CV_32F, 0, 1, ksize=3)

# 절댓값/uint8 변환은 화면 표시용이다. 해리스 계산에는 부호가 남은 grad_x, grad_y가 필요하다.
# [함수 사용] dst=cv.convertScaleAbs(src, alpha=1, beta=0)
# abs(alpha*src+beta)를 uint8로 포화 변환하여 반환한다. 계산용 부호는 사라지므로 주로 표시에 쓴다.
sobel_x=cv.convertScaleAbs(grad_x)
sobel_y=cv.convertScaleAbs(grad_y)

# [강도와 표시 문제: 에지 검출_1 PPT 10~16쪽]
# dx=3,dy=4라면 sqrt(3**2+4**2)=5이다. abs(dx)+abs(dy)=7과 다른 크기 정의이다.
# atan2(dy,dx)는 밝기 증가 방향이며 실제 에지 접선은 이에 수직이다.
# 아래 정규화는 가장 큰 강도를 255로 만드는 표시 처리이다. 원래 강도 단위는 유지되지 않는다.
# convertScaleAbs는 abs(alpha*src+beta)를 8비트로 포화 변환한다. 부호 복원은 불가능하다.
# [함수 사용] dst=np.sqrt(a)
# 원소별 제곱근을 반환한다. np.sqrt(dx**2+dy**2)는 그레디언트 크기 계산이다.
edge_strength=np.sqrt(grad_x**2+grad_y**2)
# 표시용 범위 변환 [에지 검출_1 PPT 15~16쪽과 비교]
# 현재 코드는 절댓값 두 장의 가중합 대신 유클리드 강도를 최댓값으로 나눠 표시한다.
# 정규화된 화면값과 원래 미분 강도는 다르다. 상수 영상에서는 최댓값이 0인지 확인이 필요하다.
# [함수 사용] value=np.max(a, axis=None)
# axis 생략 시 전체 원소 중 최댓값, 지정 시 해당 축의 최댓값을 반환한다.
max_strength=np.max(edge_strength)
# [함수 사용] dst=np.uint8(values)
# 값을 8비트 부호 없는 정수로 변환한다. 안전한 범위 표시에는 먼저 clip을 고려하며 좌표값을 8비트로 제한하지 않도록 주의한다.
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

# [서술형] Laplacian 2차 미분 — Sobel과 비교하는 관련 이론(미구현)
# 관련 PPT: 에지 검출_1 6~8,17쪽
# 특징: x/y의 2차 미분을 합치며, 경계 위치를 영교차와 연결해 해석할 수 있다.
# 장점: 하나의 스칼라 응답으로 여러 방향의 밝기 변화와 가는 구조를 표현한다.
# 단점: 2차 미분은 잡음에 민감하고 그레디언트처럼 밝기 증가 방향을 직접 주지 않는다.
# 활용: 에지의 영교차 원리, 평활화와 미분 순서 비교.
# 한 문장 답안: Laplacian은 2차 밝기 변화로 경계를 분석하지만 잡음에 민감해 평활화와 함께 고려한다.
# [계산·코드 연결]
# ksize=1 기본 커널은 [[0,1,0],[1,-4,1],[0,1,0]]. 이 파일은 Sobel만 실행한다.
