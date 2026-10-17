# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 11:04:25 2023

@author: admin
"""

# ==================================================================
# [선행 이론] 여러 물체의 외곽선·모멘트·도형 표현을 비교하는 배경
# 근거: 4장 영역 특징_3 (1) 2~14,27쪽
# 읽는 순서: 선행 이론 → 관련 코드 앞의 서술형 요약 → 계산·출력 주석.
# PPT의 배경 이론과 현재 코드의 구현 범위를 구분하여 공부한다.
#
# 1. 영역 표현과 특징은 무엇이 다른가?
# 표현(representation)은 영역을 경계점 또는 내부 픽셀로 어떻게 저장할지 정하는 것이다.
# 묘사(description)는 그 표현에서 둘레·면적·모멘트·색·텍스처 같은 비교 가능한 수치를 얻는 것이다.
# 경계 표현은 모양에, 내부 표현은 밝기·색·무늬에 유리하다. 목적에 따라 필요한 정보가 다르다.
# 특징의 분별력이 높다는 것은 다른 물체를 잘 구별한다는 뜻이지 값이 크다는 뜻이 아니다.
#
# 2. 불변성과 등변성
# 불변성(invariance)은 변환 뒤에도 특징값이 유지되는 성질, 등변성(equivariance)은 변환에 맞춰 값도 변하는 성질이다.
# 이상적인 평면 영역의 면적은 이동/회전에서 유지되지만 길이를 s배 하면 면적은 s**2배, 둘레는 s배이다.
# 물체의 중심 좌표는 이동하면 같은 양만큼 이동한다. 방향/좌표 자체가 유지되는 것은 아니다.
# 인식에서는 회전 영향이 적은 값이 편리할 수 있고, 로봇이 집는 방향에는 회전을 따라 변하는 방향값도 필요하다.
# 광도 불변성은 밝기 변화에 대한 강건성이다. 기하 변환 불변성과 동일한 성질은 아니다.
#
# 3. 모멘트의 단계
# 영상 m_pq=sum(x**p*y**q*f(x,y)), m00=질량, cx=m10/m00, cy=m01/m00이다.
# 이진 0/1 영상의 질량은 픽셀 수이고, 0/255 영상 기본 모멘트는 255의 밝기 질량을 합친다.
# moments(mask,binaryImage=True)는 비제로 값을 1로 본다. 외곽선 입력은 다각형의 면적 모멘트를 구한다.
# 따라서 같은 물체라도 경계 다각형 면적과 마스크의 전경 픽셀 개수가 항상 같지는 않다.
# 중심 모멘트 mu는 x-cx,y-cy를 사용해 이동 영향을 제거하고, 정규화 nu는 균일 스케일도 보정한다.
# Hu는 정규화 중심 모멘트를 조합한 7개 값으로 회전을 다룬다. moments 결과에 자동 포함되지는 않는다.
# 이상적 불변성에도 실제 이산화·분할 잡음의 수치 오차가 있으며 모든 원근/비강체 변형에 불변은 아니다.
#
# 4. 서로 다른 모양 기술자의 목적
# 원형도=4*pi*A/P**2는 면적·둘레를 조합해 둥근 정도를 본다. 하나의 숫자로 모든 모양을 구별할 수는 없다.
# boundingRect는 축에 평행한 상자, minEnclosingCircle은 전체를 감싼 최소 원이다.
# approxPolyDP는 오차 허용으로 점 수를 줄이고 convexHull은 오목함을 건너뛰는 볼록한 외곽을 만든다.
# 면적 중심과 최소 원 중심은 같다는 보장이 없다. 같은 이름 '중심'이라도 계산 목적을 확인한다.
#
# 5. 이 실습에서 먼저 점검할 조건
# 전경 이진화가 틀리면 뒤의 모든 모양 값도 바뀐다. 점 수 조건과 실제 둘레 조건도 서로 다르다.
# 현재 len 조건은 점 개수를 제한하고, 강의의 '둘레 길이'를 그대로 계산한 조건은 아니다.
# 필터 후 lcontours의 인덱스와 원래 contours의 인덱스를 섞으면 서로 다른 물체의 특징을 볼 수 있다.
# 입력 이진/컬러의 크기와 좌표계를 맞춰야 겹쳐 그린 도형이 올바른 위치에 놓인다.
# 아래는 원래 실습 틀과 빈칸을 유지하며, 출력 확인과 미구현 응용 부분을 구분해 둔다.
# ==================================================================

import cv2 as cv
import numpy as np
#%%
# 실습 입력 [영역 특징 PPT 27쪽]
# 외곽선 좌표는 binaryGroup.bmp에서 얻는다. 원본에 그릴 때 두 영상의 크기가 같아야 한다.
bin=cv.imread('binaryGroup.bmp',0)
img=cv.imread('group.jpg')

cv.imshow("binary", bin); cv.imshow("Original", img)
cv.waitKey()

# [서술형] 외곽선 필터링과 모양 분석 실습
# 관련 PPT: 영역 특징 3,10~14,24~27쪽
# 특징: 이진 영상의 경계를 얻고 크기 조건으로 고른 뒤 중심·둘레·면적·도형 표현을 계산한다.
# 장점: 작은 잡음과 큰 배경 후보를 제외하고 관심 물체의 기하 특징에 집중할 수 있다.
# 단점: 분할·조건에 따라 물체가 빠질 수 있고 점 개수를 둘레로 오해하면 필터 기준이 달라진다.
# 활용: 현재 binaryGroup.bmp의 여러 물체 중 후보 선택과 도형별 표현 비교.
# 한 문장 답안: 외곽선 필터링은 관심 후보를 줄이는 데 유리하지만 선택 기준과 반환 좌표의 의미를 정확히 맞춰야 한다.
# [계산·코드 연결]
# EXTERNAL은 바깥만, NONE은 모든 경계점을 보존. 반환 순서가 위에서부터/크기 순이라고 보장되지 않는다.
# 실제 둘레는 arcLength, 점 개수는 len. 나머지 모양 기술자의 장단점은 ch4_8.py에 정리했다.
contours, hierarchy=cv.findContours(bin, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

img1=cv.resize(img, (0,0), fx=0.7, fy=0.7);
img2=cv.cvtColor(bin, cv.COLOR_GRAY2BGR)
img3=img1.copy()
img4=img1.copy()
img5=img1.copy()

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
m=cv.moments(lcontours[0])
# [출력 확인: 현재 첫 번째 lcontours, 영상/버전 변경 시 번호·수치는 달라질 수 있음]
# arcLength = 170.6690467596054
# Area: 1005.5
# center of x, y 235.0043096303663 170.04293054864908
# mom_Area 1005.5
# sum of x, y 236296.8333333333 170978.16666666666
# m00=면적, m10=sum(x*질량), m01=sum(y*질량). 중심 계산은 각각 m00으로 나눈 결과이다.
# 외곽선으로 구한 면적이므로 .5 같은 소수도 가능하며 픽셀 개수와 동일한 양이 아니다.
print(cv.arcLength(lcontours[0], True))
print("Area:", cv.contourArea(lcontours[0])); 
print("center of x, y", m['m10']/m['m00'], m['m01']/m['m00'])
print("mom_Area", m['m00']); print("sum of x, y", m['m10'], m['m01'])

# 경계 상자 [PPT 10쪽]: (x, y, 폭, 높이)를 구해 축에 평행한 사각형으로 표현한다.
# 위 모멘트는 필터링한 lcontours[0], 아래 상자는 원래 contours[0]이므로 다른 물체일 수 있다.
# 기존 실행문을 유지하고 차이를 학습 주석으로 표시했다.
r0=cv.boundingRect(contours[0])
cv.rectangle(img4, r0, (0,0,255),2)

cv.imshow("Draw_Contours", img4)
cv.waitKey()

# 실습 연결 [PPT 11~13, 27쪽]
# 남은 외곽선에 minEnclosingCircle, approxPolyDP, convexHull을 적용하는 부분은 아직 빈칸이다.
# ch4_8.py에 각 함수 예제가 있다. 이 파일은 완성 답안이 아니라 기존 실습 틀이다.
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