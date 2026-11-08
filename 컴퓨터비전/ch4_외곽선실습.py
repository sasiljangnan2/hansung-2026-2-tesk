# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 11:04:25 2023

@author: admin
"""

#%% 개념 요약 — 정의·장점·단점, 함수 사용법은 해당 코드 옆 참고

#%% [개념] 외곽선 필터링과 모양 분석 실습
# 정의: 외곽선의 둘레 등 조건으로 물체 후보를 고른 뒤 면적·중심·경계 상자를 분석하는 실습이다.
# 장점: 작은 잡음과 큰 배경 후보를 제외하고 관심 물체의 기하 특징에 집중할 수 있다.
# 단점: 분할·조건에 따라 물체가 빠질 수 있고 점 개수를 둘레로 오해하면 필터 기준이 달라진다.
# PPT: 영역 특징 3,10~14,24~27쪽

#region 핵심 이론·함수 참고
# [핵심 이론] 여러 물체의 외곽선·모멘트·도형 표현을 비교하는 배경
# 근거: 4장 영역 특징_3 (1) 2~14,27쪽
# 1. 영역 표현과 특징은 무엇이 다른가?
# 표현(representation)은 영역을 경계점 또는 내부 픽셀로 어떻게 저장할지 정하는 것이다.
# 경계 표현은 모양에, 내부 표현은 밝기·색·무늬에 유리하다. 목적에 따라 필요한 정보가 다르다.
# 2. 불변성과 등변성
# 불변성(invariance)은 변환 뒤에도 특징값이 유지되는 성질, 등변성(equivariance)은 변환에 맞춰 값도 변하는 성질이다.
# 이상적인 평면 영역의 면적은 이동/회전에서 유지되지만 길이를 s배 하면 면적은 s**2배, 둘레는 s배이다.
# 물체의 중심 좌표는 이동하면 같은 양만큼 이동한다. 방향/좌표 자체가 유지되는 것은 아니다.
# 3. 모멘트의 단계
# 영상 m_pq=sum(x**p*y**q*f(x,y)), m00=질량, cx=m10/m00, cy=m01/m00이다.
# moments(mask,binaryImage=True)는 비제로 값을 1로 본다. 외곽선 입력은 다각형의 면적 모멘트를 구한다.
# 이상적 불변성에도 실제 이산화·분할 잡음의 수치 오차가 있으며 모든 원근/비강체 변형에 불변은 아니다.
# 4. 서로 다른 모양 기술자의 목적
# 원형도=4*pi*A/P**2는 면적·둘레를 조합해 둥근 정도를 본다. 하나의 숫자로 모든 모양을 구별할 수는 없다.
# 5. 이 실습에서 먼저 점검할 조건
# 전경 이진화가 틀리면 뒤의 모든 모양 값도 바뀐다. 점 수 조건과 실제 둘레 조건도 서로 다르다.

#endregion

import cv2 as cv
from cv_assets import read_image
import numpy as np
#%%
# 실습 입력 [영역 특징 PPT 27쪽]
# 외곽선 좌표는 binaryGroup.bmp에서 얻는다. 원본에 그릴 때 두 영상의 크기가 같아야 한다.
# [함수 사용: 입출력·창]
# img=cv.imread(path, flags=cv.IMREAD_COLOR): 파일→배열. 기본 BGR, 읽기 실패 시 None인지 확인한다.
# cv.imshow(winname,img): 이름이 winname인 창에 배열을 표시한다. 저장 함수는 아니다.
# key=cv.waitKey(delay=0): GUI 이벤트·키 처리. 0은 계속 대기, 양수는 ms, 입력 없으면 보통 -1이다.
# cv.destroyAllWindows(): OpenCV 창을 닫는다. 캡처 자원 해제는 cap.release()이다.
bin=read_image('binaryGroup.bmp',0)
img=read_image('group.jpg')

cv.imshow("binary", bin); cv.imshow("Original", img)
cv.waitKey()

# EXTERNAL은 바깥만, NONE은 모든 경계점을 보존. 반환 순서가 위에서부터/크기 순이라고 보장되지 않는다.
# 실제 둘레는 arcLength, 점 개수는 len. 나머지 모양 기술자의 장단점은 ch4_8_모멘트.py에 정리했다.
# [함수 사용] contours,hierarchy=cv.findContours(binary, mode, method)
# binary=단일 채널 이진 영상, mode=검색 방식, method=점 보존/근사 방식. 외곽선 목록과 계층 배열을 반환한다.
# [PPT 인자 보충] 에지 검출 33~34쪽: offset=(x,y)는 ROI 외곽선 좌표에 더할 이동량. hierarchy의 각 항은 [next,prev,first_child,parent], 없는 관계는 -1이다.
contours, hierarchy=cv.findContours(bin, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

# [함수 사용] dst=cv.resize(src, dsize, fx=0, fy=0, interpolation=cv.INTER_LINEAR)
# dsize=(너비,높이). (0,0)이면 fx·fy 배율로 크기를 정하며, interpolation은 보간법이다. 결과 영상을 반환한다.
img1=cv.resize(img, (0,0), fx=0.7, fy=0.7);
# [함수 사용] dst=cv.cvtColor(src, code)
# src=입력 영상, code=색 변환 상수. 예: BGR2GRAY·BGR2HSV·HSV2BGR. 변환 영상을 반환한다.
# [PPT 인자 보충] 영상처리_1 16쪽: dstCn=0은 출력 채널 수 자동 결정. 출력 높이·너비와 깊이는 같고 채널 수는 변환에 따라 달라진다.
img2=cv.cvtColor(bin, cv.COLOR_GRAY2BGR)
img3=img1.copy()
img4=img1.copy()
img5=img1.copy()

# [함수 사용] cv.drawContours(img, contours, contourIdx, color, thickness=1)
# contours=외곽선 목록, contourIdx=-1이면 전체, color=BGR 색. img에 직접 그리며 thickness<0이면 내부를 채운다.
# [PPT 인자 보충] 에지 검출 35쪽: thickness<0은 내부 채움, lineType은 선 모양, hierarchy/maxLevel은 계층 범위, offset은 그릴 좌표의 이동량이다. img를 직접 수정한다.
cv.drawContours(img2, contours, -1, (255, 0, 255), 2)
cv.drawContours(img1, contours, -1, (255, 0, 255), 2)
cv.imshow("Contours_gray", img2)
cv.imshow("Contours_original", img1)
cv.waitKey()

# 작은 잡음과 큰 물체 제외 [PPT 27쪽]
# 아래 기존 코드는 경계점 개수로 고른다. len(contour)는 점 개수, arcLength는 둘레 길이이다.
# PPT의 '둘레 50 이하/1000 이상 제외'를 정확히 구현하려면 arcLength로 판단해야 한다.
lcontours=[]
for i in range(len(contours)):
    if len(contours[i])>50 and len(contours[i])<1000:
        lcontours.append(contours[i])
        
# [출력 확인: 현재 binaryGroup.bmp, 현재 OpenCV] 필터를 통과한 외곽선 수는 4이다.
# 원래 바깥 외곽선은 6개이고 점 개수 50초과/1000미만 조건으로 고른다. 경계점 개수는 둘레와 다르다.
# img 원본=(319,480,3), bin=(224,336), 0.7배 img1=(223,336,3)으로 한 행 차이가 있다.
# 반환 외곽선의 좌표는 bin 기준이다. 다른 크기의 컬러 영상에 겹칠 때 좌표계를 확인한다.
print(len(lcontours))  # 출력: 4 (현재 입력/점 개수 조건)
cv.drawContours(img3, lcontours, -1, (0, 255, 255), 2)
cv.imshow("Contours_animal", img3)

cv.waitKey()
cv.destroyAllWindows()

#%% use img4
#1. moment, center
#2. contour length, contour area, roundness
#3 bounding box(contours[0]), minimum enclosing circle(contours[1]), 
#polylines(contours[2]), convex Hull(contours[3]) on each contour 
#%%
#1. moment, center
# 모멘트와 중심 [PPT 4, 14쪽]
# OpenCV의 m10은 x좌표, m01은 y좌표에 대한 합이다. m00은 면적이다.
# m00이 0이면 중심을 나눌 수 없으므로 퇴화한 외곽선은 별도 확인이 필요하다.
# [함수 사용] m=cv.moments(array, binaryImage=False)
# array=외곽선 또는 단일 채널 영상. 모멘트 딕셔너리를 반환한다. 영상에서 binaryImage=True면 0이 아닌 픽셀을 1로 센다.
m=cv.moments(lcontours[0])
# [출력 확인: 현재 첫 번째 lcontours, 영상/버전 변경 시 번호·수치는 달라질 수 있음]
# arcLength = 170.6690467596054
# Area: 1005.5
# center of x, y 235.0043096303663 170.04293054864908
# mom_Area 1005.5
# sum of x, y 236296.8333333333 170978.16666666666
# m00=면적, m10=sum(x*질량), m01=sum(y*질량). 중심 계산은 각각 m00으로 나눈 결과이다.
# 외곽선으로 구한 면적이므로 .5 같은 소수도 가능하며 픽셀 개수와 동일한 양이 아니다.
# [함수 사용] length=cv.arcLength(curve, closed)
# curve=점 배열, closed=True이면 끝점과 시작점도 연결한다. 점 개수가 아니라 거리 합인 실수 길이를 반환한다.
print(cv.arcLength(lcontours[0], True))
# [함수 사용] area=cv.contourArea(contour, oriented=False)
# contour=외곽선 점 배열. 둘러싼 다각형 면적을 실수로 반환하며 oriented=True이면 방향에 따른 부호를 유지한다.
print("Area:", cv.contourArea(lcontours[0])); 
print("center of x, y", m['m10']/m['m00'], m['m01']/m['m00'])
print("mom_Area", m['m00']); print("sum of x, y", m['m10'], m['m01'])

# 경계 상자 [PPT 10쪽]: (x, y, 폭, 높이)를 구해 축에 평행한 사각형으로 표현한다.
# 위 모멘트는 필터링한 lcontours[0], 아래 상자는 원래 contours[0]이므로 다른 물체일 수 있다.
# 기존 실행문을 유지하고 차이를 학습 주석으로 표시했다.
# [함수 사용] x,y,w,h=cv.boundingRect(points)
# 점 배열 또는 0이 아닌 픽셀을 감싸는 축에 평행한 사각형을 반환한다. w·h는 끝 좌표가 아니라 너비·높이이다.
r0=cv.boundingRect(contours[0])
# [함수 사용] cv.rectangle(img, pt1, pt2, color, thickness=1)
# pt1·pt2=(x,y) 꼭짓점, color=BGR 색. img에 그리며 -1은 채우기이다. (x,y,w,h) 사각형 인자 형식도 있다.
# [PPT 인자 보충] OpenCV 기본 30쪽: lineType은 선 연결·안티앨리어싱, shift는 좌표의 소수 비트 수. thickness<0이면 채우며 img를 직접 수정한다.
cv.rectangle(img4, r0, (0,0,255),2)

cv.imshow("Draw_Contours", img4)
cv.waitKey()

# 실습 연결 [PPT 11~13, 27쪽]
# 남은 외곽선에 minEnclosingCircle, approxPolyDP, convexHull을 적용하는 부분은 아직 빈칸이다.
# ch4_8_모멘트.py에 각 함수 예제가 있다. 이 파일은 완성 답안이 아니라 기존 실습 틀이다.
## 다른 contour 이 밑에 그려 볼 것

##실습

#%%
cv.imshow("Draw on Contour", img4)
cv.waitKey()
cv.destroyAllWindows()

#%%
# thickness=-1은 외곽선 내부를 칠한다. 모멘트 계산이나 볼록 껍질 계산과는 다른 표시 작업이다.
cv.drawContours(img5, lcontours, -1, 255, thickness=-1)
cv.imshow("Draw_Contours_filled", img5)
cv.waitKey()
cv.destroyAllWindows()
