# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 20:54:33 2023

@author: BigData
"""

#%% 개념 요약 — 정의·장점·단점, 함수 사용법은 해당 코드 옆 참고

#%% [개념] 전역 임계값과 Otsu 이진화
# 정의: 이진화는 임계값으로 픽셀을 두 집단으로 나누는 방법이다. Otsu는 두 집단의 클래스 내 분산을 최소화하는
#   전역 임계값을 자동으로 정한다.
# 장점: 직접 임계값을 정하는 부담이 적고, 물체·배경의 밝기가 잘 나뉜 영상에 유리하다.
# 단점: 위치별 조명 변화나 두 밝기 분포의 겹침에 약하다. 물체 의미를 인식하는 방법은 아니다.
# PPT: 영상처리_1 24~30쪽

#%% [개념] 적응 임계값
# 정의: 주변 픽셀의 밝기를 이용해 위치마다 서로 다른 임계값을 정하는 이진화 방법이다.
# 장점: 조명이 고르지 않아 하나의 임계값을 쓰기 어려운 영상에 대응할 수 있다.
# 단점: 이웃 크기와 C에 민감하고 배경 무늬·작은 잡음까지 전경으로 남을 수 있다.
# PPT: 영역 분할_2 4~6쪽
# 참고 이론이며 이 파일에서 실행하지 않음.

#region 핵심 이론·함수 참고
# [핵심 이론] 이진화·Otsu·연결 성분의 기초
# 근거: 3장 영상처리_1 24~31쪽 / 4장 영역 분할_2 4~6쪽
# 1. 명암 영상에서 이진 영역으로
# 명암값은 밝기이고 이진값은 특정 기준에 따른 두 클래스의 표시이다. 전경은 보통 255, 배경은 0이다.
# 전경/배경 중 어느 쪽이 흰색인지는 과제와 threshold 종류가 정한다. 흰색이 항상 밝은 물체인 것은 아니다.
# 2. 히스토그램으로 임계값을 정하는 발상
# 두 집단의 밝기가 잘 떨어져 있으면 히스토그램 두 봉우리 사이의 계곡을 T로 생각할 수 있다.
# p(k)=밝기 k의 비율, w0=sum(p(k),k<=T), w1=sum(p(k),k>T)이다.
# 각 집단 평균 mu0,mu1과 분산 var0,var1을 구해 within=w0*var0+w1*var1을 최소화한다.
# 동일한 기준을 클래스 간 분산 w0*w1*(mu0-mu1)**2의 최대화로 볼 수도 있다.
# 3. 전역과 지역 임계값의 차이
# 전역 방법은 모든 위치에 같은 T를 사용한다. Otsu가 자동이라고 위치별 T를 쓰는 것은 아니다.
# 4. 이진화 이후 연결 성분
# 연결 성분은 이웃으로 이어진 전경 픽셀 집합이다. 4-연결은 상하좌우, 8-연결은 대각선도 포함한다.
# 레이블링은 각 영역에 서로 다른 번호를 준다. 이진화의 0/255와 물체별 레이블 번호는 다르다.


#%% [PPT 함수 보충] 이진화 종류·연결요소·적응 임계값 (현재 실행 없음)
# 관련 PPT: 영상처리_1 27·29·31쪽 / 영역 분할_2 4~6쪽.
# [출력 문제] src=[50,100,150], T=100, maxval=255일 때:
# BINARY -> [0,0,255], BINARY_INV -> [255,255,0], TRUNC -> [50,100,100].
# TOZERO -> [0,0,150], TOZERO_INV -> [50,100,0]. 같은 값 100은 src>T가 아니므로 이하 분기에 들어간다.
# Otsu는 클래스 내 분산 최소화와 클래스 간 분산 최대화가 동치이다.
# 각 T마다 모든 밝기를 다시 합산하는 단순 탐색은 밝기 단계 L에 O(L²), 누적합을 활용하면 O(L)으로 계산할 수 있다.
#
# [함수 사용] n,labels=cv.connectedComponents(binary,connectivity=8,ltype=cv.CV_32S)
# binary는 uint8 단일 채널(0=배경, 비제로=전경), labels는 같은 높이·너비의 정수 레이블 영상이다.
# n은 배경 번호 0을 포함하므로 전경 연결요소 수는 n-1이다. connectivity=4는 상하좌우, 8은 대각선도 연결한다.
# 주석 예: b=np.array([[255,0],[0,255]],np.uint8)
# cv.connectedComponents(b,connectivity=4)[0] -> 3, connectivity=8이면 -> 2 (각각 배경 1개 포함).
#
# [함수 사용] dst=cv.adaptiveThreshold(src,maxValue,adaptiveMethod,thresholdType,blockSize,C)
# src=uint8 명암, maxValue=전경 출력값. adaptiveMethod=ADAPTIVE_THRESH_MEAN_C 또는 GAUSSIAN_C.
# thresholdType은 THRESH_BINARY/BINARY_INV, blockSize는 1보다 큰 홀수, C는 이웃 평균에서 뺄 상수이다.
# 주석 예: dst=cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,21,10)
# 지역 평균이 100이고 C=10이면 T=90, src=95는 BINARY에서 255가 된다. C 증가→T 감소→흰 전경이 늘어나는 경향.
# 장점은 불균일한 조명 대응, 한계는 이웃 크기·C에 대한 민감도이다. 위 예제를 추가 실행하는 파일은 아니다.

#endregion

import cv2 as cv
from cv_assets import read_image
import sys
#%%
# [함수 사용: 입출력·창]
# img=cv.imread(path, flags=cv.IMREAD_COLOR): 파일→배열. 기본 BGR, 읽기 실패 시 None인지 확인한다.
# cv.imshow(winname,img): 이름이 winname인 창에 배열을 표시한다. 저장 함수는 아니다.
# key=cv.waitKey(delay=0): GUI 이벤트·키 처리. 0은 계속 대기, 양수는 ms, 입력 없으면 보통 -1이다.
# cv.destroyAllWindows(): OpenCV 창을 닫는다. 캡처 자원 해제는 cap.release()이다.
img=read_image('soccer.jpg')\

if img is None:
    sys.exit("No File exists.")
#%%    
# 기준: w0*var0+w1*var1 최소화. 동치로 클래스 간 분산을 최대화한다.
# t는 실수 임계값, bin은 (948,1434) uint8 영상. THRESH_OTSU 사용 시 전달한 0은 무시된다.
# 현재 R 채널에서 T=113.0. BINARY는 R>113이면 255, R<=113이면 0이고 INV는 반대이다.
# R 채널과 cvtColor의 gray는 다르므로 입력을 바꾸면 같은 T가 나온다고 보장되지 않는다.
# [함수 사용] t,dst=cv.threshold(src, thresh, maxval, type)
# thresh=기준값, maxval=이진화 출력값, type=방법. t는 사용한 임계값, dst는 결과이며 OTSU는 기준을 자동 계산한다.
# [PPT 인자 보충] 영상처리_1 29쪽: 비교는 src>thresh이다(같으면 이하 쪽). THRESH_OTSU는 자동 T를 반환하며 초기 thresh는 무시한다.
t, bin = cv.threshold(img[:,:,2], 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

print("threshold", t)  # 출력: threshold 113.0 (현재 soccer.jpg의 R 채널)
cv.imshow('R', img[:,:,2])
cv.imshow('R binary', bin)

cv.waitKey()
cv.destroyAllWindows()

# cv.adaptiveThreshold의 blockSize는 1보다 큰 홀수. BINARY에서 C가 커지면 T는 낮아진다.
# Otsu는 전체 분포에서 하나의 T를, 적응 방법은 위치별 T를 정한다. 아래 실행문 추가는 없다.
