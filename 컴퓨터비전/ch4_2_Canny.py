# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 11:20:02 2023

@author: BigData
"""

# ==================================================================
# [선행 이론] Canny의 목표·조건·네 단계
# 근거: 4장 에지 검출_1 21~28쪽
# 읽는 순서: 선행 이론 → 관련 코드 앞의 서술형 요약 → 계산·출력 주석.
# PPT의 배경 이론과 현재 코드의 구현 범위를 구분하여 공부한다.
#
# 1. 좋은 에지 검출기가 만족해야 할 세 기준
# 낮은 오류율: 실제 경계를 놓치는 누락과 비경계를 경계로 만드는 오검출을 줄인다.
# 정확한 위치: 검출한 경계가 실제 변화 위치에 가까워야 한다.
# 단일 응답: 하나의 실제 경계를 여러 줄의 중복 응답으로 만들지 않아야 한다.
# 이 목표를 위해 잡음 완화·미분·위치 정리·연결 판단을 함께 사용한다.
#
# 2. 단계별로 해결하는 문제
# 평활화: 잡음으로 생긴 빠른 변화를 줄인다. 너무 강하면 실제 약한 경계도 약해진다.
# 미분: 그레디언트의 크기와 방향을 얻는다. 밝기 자체가 아니라 밝기의 변화량을 평가한다.
# 비최대 억제(NMS): 그레디언트 방향 양쪽과 비교해 최대가 아닌 응답을 제거하여 얇게 만든다.
# 이력 추적(hysteresis): 높은 기준을 통과한 강한 에지를 시작으로 연결된 약한 후보를 유지한다.
# PPT의 '에지 방향 비교'는 NMS에서 경계 접선이 아니라 그레디언트 방향으로 해석한다.
#
# 3. 이중 임계값은 단순한 밝기 범위 필터가 아니다
# 낮은 값보다 작은 반응은 버리고 강한 반응은 시작점으로 사용한다.
# 낮은 값과 높은 값 사이의 약한 반응은 강한 에지에 이어져야 최종 에지로 남는다.
# 강한 A에 연결된 약한 C는 살고 고립된 약한 B는 사라지는 것이 PPT 23쪽의 예이다.
# 높음:낮음의 2:1~3:1은 강의의 출발점 제안이지 모든 영상의 고정 정답이 아니다.
# L2gradient 기본 False는 L1 강도, True는 유클리드 강도를 사용한다. 결과는 방향이 아닌 이진 에지 맵이다.
#
# 4. 현재 코드가 하는 것과 하지 않는 것
# 현재 ch4_2_Canny.py는 GaussianBlur를 별도로 호출하지 않고 gray에 Canny를 적용한다.
# 명시적 평활화를 먼저 하는 ch4_1_음식에지.py/차선 예제와 처리 흐름을 비교한다.
# 얇고 연결된 에지라도 그림자와 실제 물체 경계를 의미적으로 분리했다는 뜻은 아니다.
# ==================================================================



#%% [PPT 함수 보충] Canny의 두 호출 형식
# 관련 PPT: 에지 검출 28쪽. 아래 두 번째 형식은 사용법 주석이며 기존 Canny 실행은 유지한다.
# 1) edges=cv.Canny(gray,low,high,apertureSize=3,L2gradient=False): 내부에서 미분을 계산한다.
# 2) edges=cv.Canny(dx,dy,low,high,L2gradient=False): 이미 계산한 미분을 입력하는 오버로드이다.
# 두 번째 형식의 dx/dy는 같은 크기의 CV_16S 미분이다. 기존 float32 Sobel 결과를 그대로 넘기는 형식은 아니다.
# 주석 예: dx16=cv.Sobel(gray,cv.CV_16S,1,0,ksize=3); dy16=cv.Sobel(gray,cv.CV_16S,0,1,ksize=3)
# 주석 예: edge2=cv.Canny(dx16,dy16,100,200,L2gradient=True)
# 첫 형식도 가우시안 평활화를 자동으로 대신한다고 가정하지 않는다. 강의 흐름처럼 필요 시 입력을 먼저 평활화한다.

import cv2 as cv

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

# [서술형] Canny 에지 검출
# 관련 PPT: 에지 검출_1 21~28쪽
# 특징: 평활화→미분→비최대 억제→이중 임계값/이력 추적으로 얇고 연결된 에지를 만든다.
# 장점: 강한 에지와 연결된 약한 에지를 살리고 고립된 약한 반응을 억제한다.
# 단점: 여러 단계와 임계값 설정이 필요하고 그림자와 실제 물체 경계를 의미적으로 구별하지 못한다.
# 활용: 외곽선 추적, 허프 직선 검출, 차선 검출의 입력 에지 맵.
# 한 문장 답안: Canny는 비최대 억제와 이력 추적으로 얇은 경계를 얻지만 임계값·잡음·그림자의 영향을 받는다.
# [계산·코드 연결]
# 목표는 낮은 오류율·정확한 위치·한 경계의 한 응답. NMS 비교 방향은 그레디언트 방향이다.
# 약한 후보는 강한 에지와 연결되어야 유지된다. 두 임계값이 단순한 밝기 상·하한은 아니다.
# 이 파일은 GaussianBlur를 별도 호출하지 않는다. 명시적 전처리는 ch4_1_음식에지.py에 있다.
# L2gradient=False 기본은 |dx|+|dy|, True는 sqrt(dx**2+dy**2). 출력은 (948,1434) 0/255 uint8.
# [함수 사용] edges=cv.Canny(src, threshold1, threshold2, apertureSize=3, L2gradient=False)
# 수업 예제의 src=uint8 명암 영상, 두 threshold=히스테리시스 기준. 단일 채널 에지 맵을 반환한다.
# [PPT 인자 보충] 에지 검출 28쪽: apertureSize는 Sobel 크기(3/5/7). L2gradient=True면 sqrt(dx²+dy²), False면 |dx|+|dy|로 강도를 계산한다. 반환은 0/255의 uint8 에지 맵이다.
canny1=cv.Canny(gray, 50, 150)
canny2=cv.Canny(gray, 100, 200)

cv.imshow('Original', gray)
cv.waitKey()

cv.imshow('Canny1', canny1)
cv.waitKey()

cv.imshow('Canny2', canny2)


cv.waitKey()
cv.destroyAllWindows()
