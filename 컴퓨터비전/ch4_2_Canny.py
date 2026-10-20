# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 11:20:02 2023

@author: BigData
"""

#%% 개념 요약 — 정의·장점·단점, 함수 사용법은 해당 코드 옆 참고

#%% [개념] Canny 에지 검출
# 정의: 잡음 완화·미분·비최대 억제·이력 추적을 결합해 얇고 연결된 에지를 찾는 방법이다.
# 장점: 강한 에지와 연결된 약한 에지를 살리고 고립된 약한 반응을 억제한다.
# 단점: 여러 단계와 임계값 설정이 필요하고 그림자와 실제 물체 경계를 의미적으로 구별하지 못한다.
# PPT: 에지 검출_1 21~28쪽

#region 핵심 이론·함수 참고
# [핵심 이론] Canny의 목표·조건·네 단계
# 근거: 4장 에지 검출_1 21~28쪽
# 1. 좋은 에지 검출기가 만족해야 할 세 기준
# 낮은 오류율: 실제 경계를 놓치는 누락과 비경계를 경계로 만드는 오검출을 줄인다.
# 2. 단계별로 해결하는 문제
# 평활화: 잡음으로 생긴 빠른 변화를 줄인다. 너무 강하면 실제 약한 경계도 약해진다.
# 3. 이중 임계값은 단순한 밝기 범위 필터가 아니다
# 낮은 값보다 작은 반응은 버리고 강한 반응은 시작점으로 사용한다.
# 높음:낮음의 2:1~3:1은 강의의 출발점 제안이지 모든 영상의 고정 정답이 아니다.
# 4. 현재 코드가 하는 것과 하지 않는 것
# 현재 ch4_2_Canny.py는 GaussianBlur를 별도로 호출하지 않고 gray에 Canny를 적용한다.
# 얇고 연결된 에지라도 그림자와 실제 물체 경계를 의미적으로 분리했다는 뜻은 아니다.


#%% [PPT 함수 보충] Canny의 두 호출 형식
# 관련 PPT: 에지 검출 28쪽. 아래 두 번째 형식은 사용법 주석이며 기존 Canny 실행은 유지한다.
# 1) edges=cv.Canny(gray,low,high,apertureSize=3,L2gradient=False): 내부에서 미분을 계산한다.
# 2) edges=cv.Canny(dx,dy,low,high,L2gradient=False): 이미 계산한 미분을 입력하는 오버로드이다.
# 두 번째 형식의 dx/dy는 같은 크기의 CV_16S 미분이다. 기존 float32 Sobel 결과를 그대로 넘기는 형식은 아니다.
# 주석 예: dx16=cv.Sobel(gray,cv.CV_16S,1,0,ksize=3); dy16=cv.Sobel(gray,cv.CV_16S,0,1,ksize=3)
# 주석 예: edge2=cv.Canny(dx16,dy16,100,200,L2gradient=True)
# 첫 형식도 가우시안 평활화를 자동으로 대신한다고 가정하지 않는다. 강의 흐름처럼 필요 시 입력을 먼저 평활화한다.

#endregion

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
