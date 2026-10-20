# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 11:27:12 2023

@author: BigData
"""

# ==================================================================
# [선행 이론] 에지 맵에서 경계 표현으로 넘어가는 과정
# 근거: 4장 에지 검출_1 30~36쪽 / 영역 특징 3쪽
# 읽는 순서: 선행 이론 → 관련 코드 앞의 서술형 요약 → 계산·출력 주석.
# PPT의 배경 이론과 현재 코드의 구현 범위를 구분하여 공부한다.
#
# 1. 에지 픽셀과 외곽선은 같은 표현이 아니다
# 에지 맵은 어느 픽셀이 에지인지 담은 이진 영상이다. 외곽선은 경계를 따라 나열한 좌표 배열이다.
# 좌표 표현으로 바꾸면 둘레·면적·중심·다각형·볼록 껍질처럼 후속 기하 계산을 하기 쉽다.
# 외부 경계만으로는 내부의 색·명암·텍스처까지 모두 표현하지 못하므로 목적에 맞춰 표현을 고른다.
#
# 2. 어떤 경계를 가져올지: retrieval mode
# RETR_EXTERNAL: 가장 바깥 경계만. 내부 구멍은 별도 경계로 반환하지 않는다.
# RETR_LIST: 경계를 모두 가져오지만 부모/자식 계층으로 연결하지 않는다.
# RETR_CCOMP: 외곽과 구멍을 두 단계로 정리한다.
# RETR_TREE: 중첩된 전체 부모/자식 계층을 만든다.
# hierarchy의 각 항목은 [다음,이전,첫 자식,부모]이고 없으면 -1이다.
#
# 3. 얼마나 많은 점을 남길지: approximation method
# CHAIN_APPROX_NONE은 연결 경계점을 모두 남기고 SIMPLE은 직선의 중간점을 줄인다.
# 같은 모양도 점 수는 달라질 수 있다. len(contour)는 점 개수이고 arcLength는 좌표 간 거리 합이다.
# 상하좌우 한 칸은 거리 1, 대각 한 칸은 sqrt(2)이므로 점 수와 유클리드 둘레는 일반적으로 다르다.
# 외곽선 좌표는 (x,y), NumPy 영상 접근은 [y,x]이다. 반환 순서를 물체 이름·크기 순으로 단정하지 않는다.
# 현재 파일은 점 개수 조건으로 긴 경계를 고른다. 물체 의미를 알아서 분류하는 코드는 아니다.
# ==================================================================



#%% [PPT 전체 확인 보충] 외곽선 옵션과 버전 차이
# 관련 PPT: 에지 검출 33~35쪽.
# RETR_EXTERNAL=최외곽만, RETR_LIST=모두 나열(부모/자식 없음), RETR_CCOMP=외곽/구멍 2계층, RETR_TREE=전체 중첩 계층.
# CHAIN_APPROX_NONE=인접 경계점 모두, SIMPLE=직선 구간 끝점 중심 압축, TC89_L1/TC89_KCOS=Teh-Chin 근사 방식.
# 이 옵션은 검출 결과의 표현·계층을 바꾸므로 len(contour)는 실제 둘레와 같지 않다. 길이는 arcLength로 계산한다.
# PPT의 '입력을 수정한다'는 구버전 설명과 달리 OpenCV 3.2 이후 findContours는 원본을 수정하지 않는다.
# drawContours는 표시 대상 img를 직접 바꾼다. 검색 함수와 그리기 함수의 원본 변경 여부를 구분한다.

import cv2 as cv
from cv_assets import read_image
import numpy as np
#%%

# [함수 사용: 입출력·창]
# img=cv.imread(path, flags=cv.IMREAD_COLOR): 파일→배열. 기본 BGR, 읽기 실패 시 None인지 확인한다.
# cv.imshow(winname,img): 이름이 winname인 창에 배열을 표시한다. 저장 함수는 아니다.
# key=cv.waitKey(delay=0): GUI 이벤트·키 처리. 0은 계속 대기, 양수는 ms, 입력 없으면 보통 -1이다.
# cv.destroyAllWindows(): OpenCV 창을 닫는다. 캡처 자원 해제는 cap.release()이다.
img=read_image('soccer.jpg')
# [함수 사용] dst=cv.cvtColor(src, code)
# src=입력 영상, code=색 변환 상수. 예: BGR2GRAY·BGR2HSV·HSV2BGR. 변환 영상을 반환한다.
# [PPT 인자 보충] 영상처리_1 16쪽: dstCn=0은 출력 채널 수 자동 결정. 출력 높이·너비와 깊이는 같고 채널 수는 변환에 따라 달라진다.
gray=cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# [함수 사용] edges=cv.Canny(src, threshold1, threshold2, apertureSize=3, L2gradient=False)
# 수업 예제의 src=uint8 명암 영상, 두 threshold=히스테리시스 기준. 단일 채널 에지 맵을 반환한다.
# [PPT 인자 보충] 에지 검출 28쪽: apertureSize는 Sobel 크기(3/5/7). L2gradient=True면 sqrt(dx²+dy²), False면 |dx|+|dy|로 강도를 계산한다. 반환은 0/255의 uint8 에지 맵이다.
canny=cv.Canny(gray, 100, 200)
#%%

# [서술형] 외곽선 기반 경계 분석
# 관련 PPT: 에지 검출_1 30~36쪽 / 영역 특징 3쪽
# 특징: 이진 전경 경계를 순서 있는 점 배열로 표현하고 필요하면 중첩 계층도 저장한다.
# 장점: 전체 픽셀보다 경계 중심으로 모양·둘레·다각형 등 기하 특징을 다루기 쉽다.
# 단점: 앞 단계 분할 오류에 민감하고 경계만으로는 물체 내부의 색·텍스처를 표현하기 어렵다.
# 활용: 면적·중심·둘레·모양 기술자 계산, 물체 외곽선 표시.
# 한 문장 답안: 외곽선 분석은 모양을 점 배열로 다루기 편하지만 정확한 전경 마스크와 경계 추출이 먼저 필요하다.
# [계산·코드 연결]
# EXTERNAL=바깥만, LIST=모두 계층 없이, CCOMP=2단계, TREE=전체 중첩 계층.
# NONE은 모든 점, SIMPLE은 직선 중간점을 줄인다. 점 수와 실제 둘레는 다르다.
# 좌표는 (x,y), hierarchy 항목은 [다음,이전,첫 자식,부모], 없는 번호는 -1.
# [함수 사용] contours,hierarchy=cv.findContours(binary, mode, method)
# binary=단일 채널 이진 영상, mode=검색 방식, method=점 보존/근사 방식. 외곽선 목록과 계층 배열을 반환한다.
# [PPT 인자 보충] 에지 검출 33~34쪽: offset=(x,y)는 ROI 외곽선 좌표에 더할 이동량. hierarchy의 각 항은 [next,prev,first_child,parent], 없는 관계는 -1이다.
contour, hierarchy = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

# [출력 확인: 현재 soccer.jpg, 현재 OpenCV]
# 첫 경계 shape=(16,1,2), dtype=int32. 처음 세 점은 [[[1312,862]],[[1313,862]],[[1314,862]]]이다.
# 마지막 점은 [[1313,862]]이며 중간에는 x=1320까지 갔다가 되돌아오는 점들도 있다.
# 아래 print는 이 좌표 16개 전체를 출력한다. CHAIN_APPROX_NONE이라 직선 위 점도 모두 보관한다.
# 그 다음 len(contour[0])의 값은 16이지만 단독 표현식이므로 .py 실행 시 출력되지 않는다.
# 경계 반환 순서와 좌표는 영상·OpenCV 버전에 따라 달라질 수 있어 첫 점 자체를 일반 규칙으로 외우지 않는다.
print(contour[0])
len(contour[0])

# 경계 필터링과 표시 [에지 검출_1 PPT 32쪽]
# shape[0]>100은 경계점 개수 조건이며 실제 둘레>100과는 다르다(둘레는 arcLength).
# drawContours의 -1은 전달된 모든 외곽선을 그리라는 뜻이다.
lcontour=[ ]
for i in range(len(contour)):
    if contour[i].shape[0] >100:
        lcontour.append(contour[i])
        
# [함수 사용] cv.drawContours(img, contours, contourIdx, color, thickness=1)
# contours=외곽선 목록, contourIdx=-1이면 전체, color=BGR 색. img에 직접 그리며 thickness<0이면 내부를 채운다.
# [PPT 인자 보충] 에지 검출 35쪽: thickness<0은 내부 채움, lineType은 선 모양, hierarchy/maxLevel은 계층 범위, offset은 그릴 좌표의 이동량이다. img를 직접 수정한다.
cv.drawContours(img, lcontour, -1, (0,255,0), 3)

cv.imshow('Original', img)
cv.waitKey()

cv.imshow('Canny', canny)
cv.waitKey()

cv.destroyAllWindows()
