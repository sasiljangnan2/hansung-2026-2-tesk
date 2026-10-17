# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 10:00:53 2023

@author: BigData
"""

# ==================================================================
# [선행 이론] 영역 기술자·불변성·모멘트·모양 표현의 배경
# 근거: 4장 영역 특징_3 (1) 1~14,24~26쪽
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
# 5. 현재 예제의 구현 범위
# horse의 바깥 경계를 입력으로 모멘트·원형도·다각형·껍질·상자·원을 비교한다.
# 텍스처·Hu 호출까지 구현한 것은 아니다. Hu의 관련 이론은 파일 끝의 서술형 요약에서 구분한다.
# 현재 scikit-image가 없어 실제 horse의 면적·좌표 수치는 미확인으로 두었다.
# ==================================================================


import skimage
import cv2 as cv
import numpy as np
import sys
import math
#%%

# 이진 영역 준비 [영역 특징 PPT 24쪽]
# bool 배열을 uint8의 0/255로 바꾸고 반전해 물체를 흰색으로 만든다.
# findContours는 0이 배경이고 0이 아닌 픽셀이 물체인 단일 채널 영상을 받는다.
# [입력/출력 기준: 영역 특징 PPT 24~26쪽]
# 공식 horse 샘플은 (328,400)의 bool 배열이다. 현재 scikit-image 미설치로 이 파일 전체는 미실행이다.
# 아래 크기/자료형은 공식 샘플과 변환 규칙으로 대조했으며, 면적·좌표의 실제 숫자는 임의로 채우지 않았다.
# 공식 샘플 설명: https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.horse
org=skimage.data.horse()

if org is None:
    sys.exit(' No file')
    
# [표준 샘플의 출력 형식] 첫 줄의 세 print는 각각 (328,400), <class 'numpy.ndarray'>, bool이다.
# 다음 두 print는 1, 0이다. uint8(org)는 False/True를 0/1로 바꾸며 아직 255를 곱하기 전이다.
# uint8(org)*255는 0/255, 255-uint8(org)*255는 이를 반전한 255/0이다.
print(org.shape); print(type(org)); print(np.dtype(org[0,0]))
print(np.max(np.uint8(org))); print(np.min(np.uint8(org)))

cv.imshow("Original", np.uint8(org)*255)
img= 255-np.uint8(org)*255
cv.imshow("Horse", img)
cv.waitKey()
cv.destroyAllWindows()
#%%
# 외곽선으로 물체 표현 [PPT 3, 24쪽]
# RETR_EXTERNAL은 바깥 외곽선만, CHAIN_APPROX_NONE은 경계점을 생략하지 않고 저장한다.
# contours[0]이 가장 크거나 맨 위 물체라고 보장되지는 않는다. 반환 순서를 확인한다.
contours, hierarchy = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

img2=cv.cvtColor(img, cv.COLOR_GRAY2BGR)
cv.drawContours(img2, contours, -1, (255, 0, 255), 2)
cv.imshow("Contours", img2)
cv.waitKey()

# [서술형] 모멘트와 모양 특징 분석
# 관련 PPT: 영역 특징 2~9,14쪽
# 특징: 영역의 질량·중심·분포를 모멘트로 요약하고 면적·둘레·원형도 같은 모양 수치를 구한다.
# 장점: 큰 영역을 적은 수의 값으로 표현하고 중심/크기 정규화로 변환의 영향을 줄일 수 있다.
# 단점: 분할·경계 오차에 영향을 받고 같은 요약값을 가진 다른 모양을 완전히 구별하지 못한다.
# 활용: 물체 위치·크기·둥근 정도 비교, 후속 모양 인식의 특징 준비.
# 한 문장 답안: 모멘트는 영역 분포를 간결하게 표현하지만 분할 품질과 정규화 범위에 따라 해석해야 한다.
# [계산·코드 연결]
# 영상의 m_pq=sum(x**p*y**q*f). 현재 입력은 외곽선이므로 다각형 면적을 계산하며 픽셀 수와 다르다.
# 중심 cx=m10/m00, cy=m01/m00. m00=0이면 나눌 수 없다. mu는 중심 기준으로 이동 영향을 제거한다.
# nu_pq=mu_pq/mu00**(1+(p+q)/2)는 균일 스케일 보정. 원형도=4*pi*A/P**2, 둘레는 arcLength.
# 회전·이동·스케일의 불변성과, 변환에 맞춰 값이 변하는 등변성을 구별한다.
m=cv.moments(contours[0]) #dictionary
# [출력 해석] m은 dict이고 m00,m10,m01,m20,m11,m02,...,mu20,...,nu20,... 같은 이름과 실수값이 나온다.
# OpenCV는 3차까지의 공간/중심/정규화 중심 모멘트를 반환한다. 숫자는 선택 외곽선에 달려 있다.
# 면적 배율은 크기 배율 s의 제곱, 둘레 배율은 s이므로 4*pi*A/P**2는 균일 스케일에 불변이다.
# 연속 원에서는 원형도 1이며 이산 영상은 면적·둘레 추정법 때문에 근사 오차가 있다.
print(m)
area=cv.contourArea(contours[0]) # 면적
cx, cy = m['m10']/m['m00'], m['m01']/m['m00']
perimeter = cv.arcLength(contours[0], True)
roundness = 4.0*np.pi*area/(perimeter*perimeter)
# [출력 형식 문제] 문자열 '/n'은 줄바꿈이 아니다. '/n중점=' 같은 글자가 한 줄에 그대로 찍힌다.
# 줄바꿈 특수문자는 역슬래시와 영문 n을 연결한 표기이다. 아래 원래 실행문은 유지했다.
# 면적/cx/cy/둘레/둥근 정도는 실수로 출력되고 좌표 순서는 (x,y)이다.
print('면적= ', area, '/n중점=(', cx, cy, ')', '/n둘레= ', perimeter, '/n둥근정도= ', roundness)
#%%
img3=cv.cvtColor(img, cv.COLOR_GRAY2BGR)
# [서술형] 다각형 근사 분석
# 관련 PPT: 영역 특징 12쪽
# 특징: 허용 오차 epsilon 안에서 외곽선을 더 적은 꼭짓점의 다각형으로 근사한다.
# 장점: 점을 줄여 모양을 단순화하고 꼭짓점 수와 큰 구조를 다루기 쉽다.
# 단점: epsilon이 크면 중요한 굴곡·세부가 사라지고 작으면 단순화 효과가 줄어든다.
# 활용: 윤곽선 압축, 도형의 대략적인 모양과 꼭짓점 비교.
# 한 문장 답안: 다각형 근사는 오차 허용값으로 경계를 단순화하지만 세부 보존과 점 수 감소의 절충이 있다.
# [계산·코드 연결]
# 여기 epsilon=8, True=닫힌 곡선. 볼록 껍질처럼 반드시 볼록한 결과를 만드는 함수는 아니다.
# ndarray인 근사 좌표와 [ndarray] 리스트는 type/len이 다르다. 아래 출력 주석을 확인한다.
contour_approxP = cv.approxPolyDP(contours[0],8, True)
# [출력] <class 'numpy.ndarray'> 다음 <class 'list'>. 그 뒤 len은 근사 꼭짓점 수 K이다.
# contour_approxP.shape=(K,1,2), len([contour_approxP])는 1이다. 아래 코드는 list 길이는 출력하지 않는다.
print(type(contour_approxP)); print(type([contour_approxP]));
print(len(contour_approxP))
cv.drawContours(img3, [contour_approxP], -1, (0, 255,0),2)

# [서술형] 볼록 껍질 분석
# 관련 PPT: 영역 특징 13쪽
# 특징: 모든 경계점을 감싸는 가장 작은 볼록 집합의 외곽을 구한다.
# 장점: 전체적인 외형을 간단하게 표현하고 오목한 부분의 영향을 제거한다.
# 단점: 손가락 사이처럼 중요한 오목한 모양도 건너뛰어 원래 경계의 세부를 잃는다.
# 활용: 외형의 볼록성 비교, 물체를 감싼 전체 모양의 표현.
# 한 문장 답안: 볼록 껍질은 물체 전체를 감싸는 볼록한 외형이지만 오목한 형상은 표현하지 못한다.
# [계산·코드 연결]
# approxPolyDP의 오차 기반 단순화와 목적이 다르다. reshape는 저장 축만 바꾸고 좌표를 바꾸지 않는다.
hull = cv.convexHull(contours[0])
# [출력 형식] hull.shape=(K,1,2), hull은 K개의 (x,y) 좌표, type은 numpy.ndarray, len은 K이다.
# reshape(1,K,2) 후 점 좌표/총 원소 수는 같지만 len(hull)은 1로 바뀐다. len은 항상 첫 축의 길이이다.
print(hull.shape); print(hull);print(type(hull)) 
print(len(hull))
hull=hull.reshape(1, hull.shape[0], hull.shape[2])
cv.drawContours(img3, hull, -1, (0, 0, 255),2)

# [서술형] 축에 평행한 경계 상자 분석
# 관련 PPT: 영역 특징 10쪽
# 특징: 모든 점을 포함하는 영상 축에 평행한 최소 사각형을 (x,y,w,h)로 반환한다.
# 장점: 위치·크기를 간단히 저장하고 잘라낼 ROI를 만들기 쉽다.
# 단점: 물체의 회전에 따라 상자 크기와 포함 배경이 달라지며 정확한 외곽 모양은 담지 못한다.
# 활용: 물체 표시, 영역 잘라내기, 위치/너비/높이 비교.
# 한 문장 답안: 경계 상자는 위치와 크기의 간단한 표현이지만 방향과 세밀한 모양을 충분히 나타내지 못한다.
# [계산·코드 연결]
# minAreaRect의 회전 가능한 최소 면적 상자와 다르다. w/h는 오른쪽 아래 좌표가 아니다.
rect=cv.boundingRect(contours[0])
# [출력 형식] (x,y,w,h) 튜플 다음 <class 'tuple'>이 나온다. w/h는 오른쪽 아래 좌표가 아니다.
# boundingRect는 축에 평행한 최소 상자이다. 회전할 수 있는 minAreaRect와 구별한다.
print(rect);print(type(rect)) 
cv.rectangle(img3, rect, (255, 0,0), 2)

# [서술형] 최소 둘러싼 원 분석
# 관련 PPT: 영역 특징 11쪽
# 특징: 전체 경계점을 포함하는 가장 작은 원의 중심과 반지름을 구한다.
# 장점: 한 중심·반지름으로 물체를 감싼 범위를 간단히 표현하고 방향 의존을 줄인다.
# 단점: 길쭉하거나 오목한 물체는 빈 배경을 많이 포함하고 멀리 튄 경계점의 영향을 받을 수 있다.
# 활용: 물체 범위 표시, 둥근 대상의 대략적인 크기 비교.
# 한 문장 답안: 최소 둘러싼 원은 간결한 범위 표현이지만 길쭉하거나 오목한 물체의 세부 모양을 잃는다.
# [계산·코드 연결]
# 중심/반지름은 실수이고 면적 중심과 같다는 보장은 없다.
# 현재 np.uint8 좌표 변환은 255 초과 시 순환 가능. 원래 실행문과 출력 수치의 차이를 확인한다.
center, radius = cv.minEnclosingCircle(contours[0])
# [출력 형식] center= (실수 x,실수 y), 다음 줄 Radius= 실수 반지름이다. 여기 '\n'은 실제 줄바꿈이다.
# 최소 둘러싼 원의 중심은 면적 중심(cx,cy)과 같다고 보장되지 않는다.
# 아래 np.uint8(center)는 큰 좌표를 순환시킬 수 있다. 출력한 실수와 그려진 정수 좌표의 차이를 확인한다.
print('center= ', center, '\nRadius= ', radius)
cv.circle(img3, np.uint8(center), np.uint8(radius), (0, 255, 255), 2)


cv.imshow("Horse with Contours", img3)
cv.waitKey()

cv.destroyAllWindows()

# [서술형] Hu 모멘트 — 모양 분석과 연결하는 관련 이론(미구현)
# 관련 PPT: 영역 특징 5~6쪽
# 특징: 정규화 중심 모멘트를 조합한 7개의 모양 기술자이다.
# 장점: 이상적인 영역에서 이동·회전·균일 확대/축소의 영향을 줄여 모양을 비교할 수 있다.
# 단점: 경계 잡음·이산화 오차가 있고 서로 다른 모양을 완전히 구별하지 못한다. 원근/변형 일반 불변은 아니다.
# 활용: 크기·위치·방향이 다른 물체의 전체 모양 비교.
# 한 문장 답안: Hu 모멘트는 기하 변환에 강한 전역 모양 특징이지만 세부 분별력과 분할 오차의 한계가 있다.
# [계산·코드 연결]
# 반사에서는 7번째 값의 부호가 바뀔 수 있다. moments 딕셔너리에 자동 포함되지는 않는다.
