# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 13:56:50 2023

@author: BigData
"""

# ==================================================================
# [선행 이론] 이웃 연산·가우시안·자료형을 연결해서 이해하기
# 근거: 3장 영상처리_2 22~31쪽 / 지역특징_1 14~18,28~29쪽
# 읽는 순서: 선행 이론 → 관련 코드 앞의 서술형 요약 → 계산·출력 주석.
# PPT의 배경 이론과 현재 코드의 구현 범위를 구분하여 공부한다.
#
# 1. 필터와 연산은 구분한다
# 필터 커널 u는 어떤 이웃에 어떤 가중치를 줄지 정한다. 연산은 그 가중치를 곱해 더하는 계산이다.
# g(y,x)=sum(u(j,i)*f(y+j,x+i))는 커널을 뒤집지 않는 상관 계산이다.
# PPT는 이를 컨볼루션으로 설명하지만 filter2D 자체는 상관이다. 엄밀한 컨볼루션은 커널을 뒤집는다.
# 대칭 가우시안은 뒤집어도 같지만 엠보싱처럼 비대칭인 커널은 결과가 다를 수 있다.
# 평균처럼 커널 합이 1이면 일정 밝기를 유지하고, 미분처럼 합이 0이면 일정 영역 응답은 0이다.
#
# 2. 가우시안의 의미
# G(x,y)=exp(-(x*x+y*y)/(2*sigma*sigma))/(2*pi*sigma*sigma).
# 중심에서 멀수록 가중치가 작고 이산 가중치의 합을 1로 정규화한다.
# 2차원 가우시안은 1차원 두 커널의 외적으로 분리할 수 있다. k @ k.T와 외적 발상은 Harris에도 쓰인다.
# sigma는 퍼짐, 커널 크기는 실제 참조 범위이다. sigma=0은 크기에서 자동 추정한다는 뜻이다.
# 잡음은 완화되지만 경계·가는 무늬도 흐려질 수 있다. 큰 sigma/큰 커널이 무조건 더 정확하지는 않다.
# Harris는 원영상 표시가 아니라 미분 곱의 가중 집계에 가우시안을 쓰므로 적용 대상도 확인한다.
#
# 3. 메디안과의 비교
# 가우시안은 가중 평균인 선형 연산, 메디안은 정렬한 중앙값을 고르는 비선형 연산이다.
# 메디안은 소금·후추 잡음에 유리하고 기존 값을 골라 경계를 보존하는 경향이 있지만 작은 형상은 변할 수 있다.
# 마스크의 중앙값은 0/255를 유지하지만 평균은 중간 값을 만들 수 있다. 둘 다 그림자 전용 제거기가 아니다.
#
# 4. 음수 응답·표시·자료형
# 미분/엠보싱은 음수 응답도 의미를 가진다. CV_32F나 부호 있는 중간 결과에서 먼저 계산한다.
# uint8 형변환의 순환, OpenCV unsigned 출력의 포화, np.clip의 범위 제한은 서로 다른 처리이다.
# 부호를 잃은 뒤 +128이나 clip을 해도 원래 음수는 복원되지 않는다.
# 현재 gray16은 uint16이라 PPT의 int16과 다르다. 기존 실행문은 유지하고 차이를 설명해 둔다.
# 영상 밖에 커널이 걸칠 때는 상수·복제·반사 등의 경계 확장 방법도 결과에 영향을 준다.
# ==================================================================



#%% [PPT 전체 확인 보충] 경계 확장·자료형·필터 결과
# 관련 PPT: 영상처리_2 23~31쪽 / 에지 검출 18~19쪽.
# [borderType] BORDER_CONSTANT=상수, REPLICATE=끝값 반복, REFLECT=끝값 포함 반사, REFLECT_101=끝값 제외 반사.
# 예: 1D abcdefgh의 양쪽을 3개 확장하면 REFLECT는 cba|abcdefgh|hgf,
# REFLECT_101은 dcb|abcdefgh|gfe, REPLICATE는 aaa|abcdefgh|hhh이다. BORDER_DEFAULT는 REFLECT_101이다.
# 모든 함수가 모든 경계 방식을 지원하는 것은 아니다. 예를 들어 filter2D는 BORDER_WRAP을 지원하지 않는다.
# [자료형 표기] CV_8UC3는 8비트 unsigned 3채널, CV_16SC1는 signed 16비트 1채널,
# CV_32FC1/CV_64FC1는 float32/float64 1채널이다. NumPy에서는 dtype와 shape의 채널 축을 함께 확인한다.
# ddepth=-1은 입력 깊이 유지이다. CV_32F로 미분을 보존한 뒤 convertScaleAbs로 표시하면 음수·실수 계산과 화면 표시를 구분할 수 있다.
# uint8 영상에서 OpenCV의 많은 산술 연산은 포화 처리를 한다. NumPy 정수 배열의 산술·형변환은 같은 규칙이라고 가정하지 않는다.
# PPT의 범위 초과 직접 정수 생성은 NumPy 버전에 따라 오류가 날 수 있어, 넓은 dtype에서 계산한 뒤 clip/astype의 목적을 구분한다.

import cv2 as cv
from cv_assets import read_image
import numpy as np
import sys

#%%
# [함수 사용: 입출력·창]
# img=cv.imread(path, flags=cv.IMREAD_COLOR): 파일→배열. 기본 BGR, 읽기 실패 시 None인지 확인한다.
# cv.imshow(winname,img): 이름이 winname인 창에 배열을 표시한다. 저장 함수는 아니다.
# key=cv.waitKey(delay=0): GUI 이벤트·키 처리. 0은 계속 대기, 양수는 ms, 입력 없으면 보통 -1이다.
# cv.destroyAllWindows(): OpenCV 창을 닫는다. 캡처 자원 해제는 cap.release()이다.
img = read_image("soccer.jpg")

if img is None:
    sys.exit("No File exists.")
 #%%   
# [자료형 출력 문제: 영상처리_2 PPT 28~31쪽]
# int16은 음수를 표현하지만 uint8은 0~255이다. astype의 정수 변환은 범위 밖에서 순환한다.
# 여기서는 각 값을 256으로 나눈 나머지처럼 바뀐다: -3->253, -1->255, 256->0, 258->2.
# a는 원소 9개의 배열이고 a[0]은 배열에서 꺼낸 스칼라이다. 두 type 출력이 다른 이유이다.
# clip(a,0,255)는 범위 제한이며 이런 순환과 다르다. 원래 값에 먼저 적용해야 한다.
# [함수 사용] a=np.array(values, dtype=None)
# values=리스트 등의 입력, dtype=원소 자료형(생략하면 추론). 지정한 값을 가진 ndarray를 만든다.
a=np.array([-3,-2,-1,0,254,255,256,257,258], dtype=np.int16).astype(np.uint8)
print(type(a))  # 출력: <class 'numpy.ndarray'>
print(type(a[0]))  # 출력: <class 'numpy.uint8'>
print(a)  # 출력: [253 254 255   0 254 255   0   1   2]

#%%
print(type(img))  # 출력: <class 'numpy.ndarray'>
print(type(img[0,0,0]))  # 출력: <class 'numpy.uint8'>

#%%
# [함수 사용] dst=cv.resize(src, dsize, fx=0, fy=0, interpolation=cv.INTER_LINEAR)
# dsize=(너비,높이). (0,0)이면 fx·fy 배율로 크기를 정하며, interpolation은 보간법이다. 결과 영상을 반환한다.
img=cv.resize(img, dsize=(0,0), fx=0.4, fy=0.4)
# [함수 사용] dst=cv.cvtColor(src, code)
# src=입력 영상, code=색 변환 상수. 예: BGR2GRAY·BGR2HSV·HSV2BGR. 변환 영상을 반환한다.
# [PPT 인자 보충] 영상처리_1 16쪽: dstCn=0은 출력 채널 수 자동 결정. 출력 높이·너비와 깊이는 같고 채널 수는 변환에 따라 달라진다.
gray=cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# [함수 사용] cv.putText(img, text, org, fontFace, fontScale, color, thickness=1)
# text=문자열, org=(x,y) 기준선 시작점, fontFace=글꼴, fontScale=배율. 지정한 BGR 색으로 img에 쓴다.
# [PPT 인자 보충] OpenCV 기본 28쪽: org는 글자의 왼쪽 아래 기준점. lineType은 선 모양, bottomLeftOrigin=False는 영상 원점을 왼쪽 위로 해석한다.
cv.putText(gray, 'soccer', (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255),2 )
cv.imshow('Original', gray)
cv.waitKey()

#%%
# [서술형] 가우시안 평활화
# 관련 PPT: 영상처리_2 23~27쪽 / 지역특징_1 14~18,28쪽
# 특징: 가까운 이웃에 큰 가중치를 주는 선형 평균 필터이며 가중치 합을 1로 정규화한다.
# 장점: 미분 전에 작은 잡음 응답을 완화하고, 대칭·분리 가능한 커널로 효율적으로 계산할 수 있다.
# 단점: 가늘거나 약한 경계도 흐려지고 넓은 그림자만 골라 제거하지 못한다.
# 활용: Canny 전처리, 영상 평활화, Harris에서 미분 곱을 공간적으로 모으는 가중치.
# 한 문장 답안: 가우시안은 잡음을 부드럽게 줄이지만 경계도 흐려지므로 필요한 세부와 잡음 사이의 절충이 필요하다.
# [계산·코드 연결]
# G=exp(-(x*x+y*y)/(2*sigma*sigma))/(2*pi*sigma*sigma). 대칭이어서 상관/컨볼루션 결과가 같다.
# sigma=0은 생략이 아니라 커널 크기로 자동 추정. 큰 커널은 세부도 더 약화시킬 수 있다.
# 현재 gray=(379,574), 가로 비교 smooth=(379,1722). 해리스 예제는 원영상 대신 미분 곱에 적용한다.
# 메디안은 이웃 중앙값을 고르는 비선형 필터이다. 자세한 장단점은 ch4_손워터셰드.py와 비교한다.
# [함수 사용] dst=np.hstack((a,b))
# 영상은 너비 축으로 연결하므로 높이·채널 등 다른 축이 같아야 한다. 새 연결 배열을 반환한다.
# [함수 사용] dst=cv.GaussianBlur(src, ksize, sigmaX, sigmaY=0)
# ksize=(너비,높이) 커널, sigmaX/Y=표준편차. sigmaY=0이면 sigmaX를 사용하며 흐린 영상을 반환한다.
# [PPT 인자 보충] 영상처리_2 27쪽: ksize의 각 값은 양의 홀수 또는 sigma로 자동 결정하는 0. sigmaX=0은 커널 크기로 추정, borderType은 가장자리 확장 방식이다.
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
# [함수 사용] dst=np.uint16(values)
# 16비트 부호 없는 정수로 변환한다. 음수 미분을 보존하는 int16과는 다르다.
gray16=np.uint16(gray)
cv.waitKey()
cv.destroyAllWindows()

# [서술형] 커널 기반 영역 연산과 엠보싱
# 관련 PPT: 영상처리_2 22~24,28~31쪽
# 특징: 이웃에 커널을 곱해 더하며, 엠보싱 커널은 방향성 밝기 차이를 강조한다.
# 장점: 같은 filter2D 틀에서 평활화·미분·질감 강조를 커널 교체로 표현할 수 있다.
# 단점: 잡음·경계 확장·자료형에 민감하고 부호가 잘리면 방향성 응답을 잃는다.
# 활용: 선형 필터 비교, 방향성 경계·엠보싱 표시, 미분과 자료형 문제 학습.
# 한 문장 답안: 커널 연산은 이웃 관계를 유연하게 처리하지만 커널 목적과 중간 자료형을 맞춰야 한다.
# [계산·코드 연결]
# 현재 응답은 오른쪽 아래 - 왼쪽 위 밝기. +128은 0 응답을 중간 회색으로 표시하려는 값이다.
# filter2D는 커널을 뒤집지 않는 상관. 비대칭 커널의 컨볼루션은 180도 뒤집어 적용한다.
# 현재 gray16은 uint16이고 ddepth=-1이므로 음수가 먼저 0으로 잘린다. PPT는 signed int16이다.
# clip은 범위 제한일 뿐 손실한 부호를 복원하지 못한다. bad는 clip 없음, worse는 +128도 없음.
# [함수 사용] dst=np.uint8(values)
# 값을 8비트 부호 없는 정수로 변환한다. 안전한 범위 표시에는 먼저 clip을 고려하며 좌표값을 8비트로 제한하지 않도록 주의한다.
# [함수 사용] dst=cv.filter2D(src, ddepth, kernel)
# ddepth=출력 깊이(-1은 입력과 같음), kernel=2D 가중치. 커널을 뒤집지 않는 상관 계산 결과를 반환한다.
# [PPT 인자 보충] 영상처리_2 29~31쪽·지역특징 29쪽: anchor=(-1,-1)은 중심, delta=0은 결과에 더할 상수, borderType은 가장자리 처리. ddepth는 깊이만 지정하며 채널 수는 유지한다.
# [함수 사용] dst=np.clip(a, a_min, a_max)
# a_min 미만을 하한, a_max 초과를 상한으로 잘라 반환한다. dtype 변환은 별도이며 자동으로 uint8이 되지는 않는다.
emboss=np.uint8(np.clip(cv.filter2D(gray16, -1, femboss)+128, 0, 255))
emboss_bad=np.uint8(cv.filter2D(gray16, -1, femboss)+128)
emboss_worse=cv.filter2D(gray, -1, femboss)

cv.imshow('Emboss', emboss)
cv.imshow('Emboss_bad', emboss_bad)
cv.imshow('Emboss_worse', emboss_worse)

cv.waitKey()
cv.destroyAllWindows()
