# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 16:38:26 2023

@author: BigData
"""

# ==================================================================
# [선행 이론] 정확도·계산 효율·실시간 처리의 관계
# 근거: 3장 영상처리_2 43~45쪽 / 5장 지역특징_1 7쪽
# 읽는 순서: 선행 이론 → 관련 코드 앞의 서술형 요약 → 계산·출력 주석.
# PPT의 배경 이론과 현재 코드의 구현 범위를 구분하여 공부한다.
#
# 1. 결과가 맞는 것과 제시간에 나오는 것은 별개의 조건
# 컴퓨터 비전은 인식/측정의 정확성뿐 아니라 필요한 시간 안에 계산하는 효율도 중요하다.
# 실시간 추적은 프레임마다 제한된 시간 안에 결과를 내야 하고, 오프라인 분석은 시간을 더 쓸 수 있다.
# 정교한 방법이 더 많은 계산을 요구할 수 있으므로 응용의 정확도·속도·메모리 요구를 함께 본다.
# 모든 특성을 동시에 최대화하는 하나의 설정이 항상 존재하는 것은 아니다.
#
# 2. Python 반복과 배열 연산
# Python 이중 반복은 픽셀마다 인터프리터 연산을 반복한다. 벡터 연산은 배열 전체를 내부 루틴으로 처리한다.
# OpenCV도 최적화된 내부 구현을 사용하므로 같은 작업을 직접 반복하는 것보다 유리한 경우가 많다.
# 배열 연산도 중간 배열·메모리 비용이 들 수 있고, 작은 영상/큰 영상의 상대 속도는 달라질 수 있다.
# 속도 비교는 같은 입력·같은 작업 범위를 기준으로 해야 한다. 파일 읽기 포함 여부도 구분한다.
#
# 3. 출력 시간과 픽셀 결과의 해석
# 아래 print는 변환 전후 시각의 차이를 초 단위 실수로 출력한다. 시간 값은 매번 달라진다.
# 한 번의 측정값을 고정 정답이나 모든 상황에서의 속도 순위로 외우지 않는다.
# 가중합 0.114B+0.587G+0.299R의 uint8 절삭과 cvtColor의 반올림은 미세한 값 차이를 만들 수 있다.
# 빠름을 비교할 때 같은 결과를 얻는지도 함께 봐야 한다는 예이다.
# ==================================================================


import cv2 as cv
import numpy as np
import sys
import time

#%%

# [서술형] 반복문·배열 연산·OpenCV의 처리 시간 비교
# 관련 PPT: 영상처리_2 43~45쪽
# 특징: 같은 BGR 가중합을 Python 이중 반복문, NumPy 벡터 연산, cvtColor로 계산해 시간을 잰다.
# 장점: 벡터 연산과 라이브러리는 픽셀별 Python 반복 비용을 줄이고 최적화된 내부 계산을 사용한다.
# 단점: 배열 중간 메모리가 필요할 수 있고 한 번의 시간 측정은 환경·영상 크기의 영향을 받는다.
# 활용: 영상 처리 코드의 병목 찾기와 같은 작업을 구현하는 방법의 효율 비교.
# 한 문장 답안: 배열 연산은 Python 반복 비용을 줄이지만 실제 속도는 입력·메모리·측정 환경을 함께 봐야 한다.
# [계산·코드 연결]
# gray=0.114B+0.587G+0.299R. B=G=R=v이면 gray=v. zeros 기본 dtype은 float64이다.
# uint8 절삭과 cvtColor 반올림 차이가 있어 픽셀 값이 미세하게 다를 수 있다.
# 세 print의 시간은 초 단위 실수로 매번 달라진다. 고정 값이나 모든 상황의 속도 순서로 외우지 않는다.
def my_cvtGray1(bar_img):
    g = np.zeros([bar_img.shape[0], bar_img.shape[1]])
    for r in range(bar_img.shape[0]):
        for c in range(bar_img.shape[1]):
            g[r,c]= 0.114*bar_img[r,c,0]+0.587*bar_img[r,c,1]+0.299*bar_img[r,c,2]
    return np.uint8(g)

def my_cvtGray2(bar_img):
    g = np.zeros([bar_img.shape[0], bar_img.shape[1]])
    g = 0.114*bar_img[:,:,0]+0.587*bar_img[:,:,1]+0.299*bar_img[:,:,2]
    return np.uint8(g)
        
#%%
img = cv.imread("girl_laughing.jpg")
if img is None:
    sys.exit("No File exists.")

# 처리 시간 측정 [영상처리_2 PPT 44~45쪽]
# 영상 읽기를 제외하고 각 변환의 시간을 잰다. 한 번의 측정값은 실행 환경의 영향을 받으므로
# 모든 크기의 영상에서 같은 배수로 빠르다는 뜻은 아니다.
start=time.time()
my_cvtGray1(img)
print('My time1: ', time.time()-start)

start=time.time()
my_cvtGray2(img)
print('My time2: ', time.time()-start)

start=time.time()
cv.cvtColor(img, cv.COLOR_BGR2GRAY)
print('MOpenCV time1: ', time.time()-start)
