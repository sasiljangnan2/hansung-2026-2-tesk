# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 10:00:53 2023

@author: BigData
"""

#%% 개념 요약 — 정의·장점·단점, 함수 사용법은 해당 코드 옆 참고

#%% [개념] 모멘트와 모양 특징 분석
# 정의: 모멘트는 픽셀 위치와 값을 가중 합산해 영역의 면적·중심·분포 같은 모양 정보를 수치로 나타내는
#   방법이다.
# 장점: 큰 영역을 적은 수의 값으로 표현하고 중심/크기 정규화로 변환의 영향을 줄일 수 있다.
# 단점: 분할·경계 오차에 영향을 받고 같은 요약값을 가진 다른 모양을 완전히 구별하지 못한다.
# PPT: 영역 특징 2~9,14쪽

#%% [개념] 다각형 근사 분석
# 정의: 원래 곡선과의 허용 오차 안에서 점 수를 줄여 경계를 다각형으로 표현하는 방법이다.
# 장점: 점을 줄여 모양을 단순화하고 꼭짓점 수와 큰 구조를 다루기 쉽다.
# 단점: epsilon이 크면 중요한 굴곡·세부가 사라지고 작으면 단순화 효과가 줄어든다.
# PPT: 영역 특징 12쪽

#%% [개념] 볼록 껍질 분석
# 정의: 입력 점들을 모두 포함하는 가장 작은 볼록 다각형으로 바깥 모양을 표현하는 방법이다.
# 장점: 전체적인 외형을 간단하게 표현하고 오목한 부분의 영향을 제거한다.
# 단점: 손가락 사이처럼 중요한 오목한 모양도 건너뛰어 원래 경계의 세부를 잃는다.
# PPT: 영역 특징 13쪽

#%% [개념] 축에 평행한 경계 상자 분석
# 정의: 물체나 점들을 모두 포함하면서 영상의 가로·세로 축과 평행한 최소 사각형을 구하는 방법이다.
# 장점: 위치·크기를 간단히 저장하고 잘라낼 ROI를 만들기 쉽다.
# 단점: 물체의 회전에 따라 상자 크기와 포함 배경이 달라지며 정확한 외곽 모양은 담지 못한다.
# PPT: 영역 특징 10쪽

#%% [개념] 최소 둘러싼 원 분석
# 정의: 물체나 점들을 모두 포함하는 가장 작은 원의 중심과 반지름을 구하는 방법이다.
# 장점: 한 중심·반지름으로 물체를 감싼 범위를 간단히 표현하고 방향 의존을 줄인다.
# 단점: 길쭉하거나 오목한 물체는 빈 배경을 많이 포함하고 멀리 튄 경계점의 영향을 받을 수 있다.
# PPT: 영역 특징 11쪽

#%% [개념] Hu 모멘트
# 정의: 정규화 중심 모멘트를 조합하여 이동·크기·회전 변화의 영향을 줄인 일곱 가지 모양 특징이다.
# 장점: 이상적인 영역에서 이동·회전·균일 확대/축소의 영향을 줄여 모양을 비교할 수 있다.
# 단점: 경계 잡음·이산화 오차가 있고 서로 다른 모양을 완전히 구별하지 못한다. 원근/변형 일반 불변은 아니다.
# PPT: 영역 특징 5~6쪽
# 참고 이론이며 이 파일에서 실행하지 않음.

#region 핵심 이론·함수 참고
# [핵심 이론] 영역 기술자·불변성·모멘트·모양 표현의 배경
# 근거: 4장 영역 특징_3 (1) 1~14,24~26쪽
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
# 5. 현재 예제의 구현 범위
# horse의 바깥 경계를 입력으로 모멘트·원형도·다각형·껍질·상자·원을 비교한다.
# 텍스처·Hu 호출까지 구현한 것은 아니다. Hu의 관련 이론은 파일 끝의 서술형 요약에서 구분한다.


#%% [PPT 전체 확인 보충] 모멘트·그리기 연결
# 관련 PPT: 영역 특징 5~6·12~14쪽. 아래 호출은 참고 주석이며 현재 외곽선 분석 실행은 유지한다.
# [함수 사용: 이론 연결] hu=cv.HuMoments(moments): moments 딕셔너리에서 Hu 불변 모멘트 7개를 (7,1) 배열로 계산한다.
# 주석 예: hu=cv.HuMoments(cv.moments(contour))
# PPT의 Hu 수식과 연결되는 참고 함수이다. 7개 값이 모두 양수이거나 0~1이라고 보장되지 않으며 이산 영상에서는 완벽히 같지 않을 수 있다.
# [함수 사용] cv.polylines(img,[approx],True,(0,255,0),2)
# pts는 int32 점 배열들의 목록, True는 닫힌 경계, color는 BGR. approxPolyDP/convexHull 결과를 선으로 표시하며 img를 직접 바꾼다.
# PPT 12~13쪽의 다각형·볼록 껍질 그리기 예제와 연결된다. 한 개 경계도 [approx]처럼 목록으로 전달한다.

#endregion

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
# [함수 사용] img=skimage.data.horse()
# 말 모양의 표준 불리언 마스크 (328,400)를 반환한다. OpenCV 이진 영상으로 쓰려면 dtype·값 범위를 맞춘다.
org=skimage.data.horse()

if org is None:
    sys.exit(' No file')
    
# [표준 샘플의 출력 형식] 첫 줄의 세 print는 각각 (328,400), <class 'numpy.ndarray'>, bool이다.
# 다음 두 print는 1, 0이다. uint8(org)는 False/True를 0/1로 바꾸며 아직 255를 곱하기 전이다.
# uint8(org)*255는 0/255, 255-uint8(org)*255는 이를 반전한 255/0이다.
# [함수 사용] dt=np.dtype(type_spec)
# type_spec에 해당하는 NumPy 자료형 설명 객체를 반환한다. 실제 배열을 변환하는 호출과 구분한다.
print(org.shape); print(type(org)); print(np.dtype(org[0,0]))
# [함수 사용] value=np.max(a, axis=None)
# axis 생략 시 전체 원소 중 최댓값, 지정 시 해당 축의 최댓값을 반환한다.
# [함수 사용] value=np.min(a, axis=None)
# axis 생략 시 전체 원소 중 최솟값, 지정 시 해당 축의 최솟값을 반환한다.
# [함수 사용] dst=np.uint8(values)
# 값을 8비트 부호 없는 정수로 변환한다. 안전한 범위 표시에는 먼저 clip을 고려하며 좌표값을 8비트로 제한하지 않도록 주의한다.
print(np.max(np.uint8(org))); print(np.min(np.uint8(org)))

# [함수 사용: 입출력·창]
# cv.imshow(winname,img): 이름이 winname인 창에 배열을 표시한다. 저장 함수는 아니다.
# key=cv.waitKey(delay=0): GUI 이벤트·키 처리. 0은 계속 대기, 양수는 ms, 입력 없으면 보통 -1이다.
# cv.destroyAllWindows(): OpenCV 창을 닫는다. 캡처 자원 해제는 cap.release()이다.
cv.imshow("Original", np.uint8(org)*255)
img= 255-np.uint8(org)*255
cv.imshow("Horse", img)
cv.waitKey()
cv.destroyAllWindows()
#%%
# 외곽선으로 물체 표현 [PPT 3, 24쪽]
# RETR_EXTERNAL은 바깥 외곽선만, CHAIN_APPROX_NONE은 경계점을 생략하지 않고 저장한다.
# contours[0]이 가장 크거나 맨 위 물체라고 보장되지는 않는다. 반환 순서를 확인한다.
# [함수 사용] contours,hierarchy=cv.findContours(binary, mode, method)
# binary=단일 채널 이진 영상, mode=검색 방식, method=점 보존/근사 방식. 외곽선 목록과 계층 배열을 반환한다.
# [PPT 인자 보충] 에지 검출 33~34쪽: offset=(x,y)는 ROI 외곽선 좌표에 더할 이동량. hierarchy의 각 항은 [next,prev,first_child,parent], 없는 관계는 -1이다.
contours, hierarchy = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

# [함수 사용] dst=cv.cvtColor(src, code)
# src=입력 영상, code=색 변환 상수. 예: BGR2GRAY·BGR2HSV·HSV2BGR. 변환 영상을 반환한다.
# [PPT 인자 보충] 영상처리_1 16쪽: dstCn=0은 출력 채널 수 자동 결정. 출력 높이·너비와 깊이는 같고 채널 수는 변환에 따라 달라진다.
img2=cv.cvtColor(img, cv.COLOR_GRAY2BGR)
# [함수 사용] cv.drawContours(img, contours, contourIdx, color, thickness=1)
# contours=외곽선 목록, contourIdx=-1이면 전체, color=BGR 색. img에 직접 그리며 thickness<0이면 내부를 채운다.
# [PPT 인자 보충] 에지 검출 35쪽: thickness<0은 내부 채움, lineType은 선 모양, hierarchy/maxLevel은 계층 범위, offset은 그릴 좌표의 이동량이다. img를 직접 수정한다.
cv.drawContours(img2, contours, -1, (255, 0, 255), 2)
cv.imshow("Contours", img2)
cv.waitKey()

# 영상의 m_pq=sum(x**p*y**q*f). 현재 입력은 외곽선이므로 다각형 면적을 계산하며 픽셀 수와 다르다.
# 중심 cx=m10/m00, cy=m01/m00. m00=0이면 나눌 수 없다. mu는 중심 기준으로 이동 영향을 제거한다.
# nu_pq=mu_pq/mu00**(1+(p+q)/2)는 균일 스케일 보정. 원형도=4*pi*A/P**2, 둘레는 arcLength.
# 회전·이동·스케일의 불변성과, 변환에 맞춰 값이 변하는 등변성을 구별한다.
# [함수 사용] m=cv.moments(array, binaryImage=False)
# array=외곽선 또는 단일 채널 영상. 모멘트 딕셔너리를 반환한다. 영상에서 binaryImage=True면 0이 아닌 픽셀을 1로 센다.
m=cv.moments(contours[0]) #dictionary
# [출력 해석] m은 dict이고 m00,m10,m01,m20,m11,m02,...,mu20,...,nu20,... 같은 이름과 실수값이 나온다.
# OpenCV는 3차까지의 공간/중심/정규화 중심 모멘트를 반환한다. 숫자는 선택 외곽선에 달려 있다.
# 면적 배율은 크기 배율 s의 제곱, 둘레 배율은 s이므로 4*pi*A/P**2는 균일 스케일에 불변이다.
# 연속 원에서는 원형도 1이며 이산 영상은 면적·둘레 추정법 때문에 근사 오차가 있다.
print(m)
# [함수 사용] area=cv.contourArea(contour, oriented=False)
# contour=외곽선 점 배열. 둘러싼 다각형 면적을 실수로 반환하며 oriented=True이면 방향에 따른 부호를 유지한다.
area=cv.contourArea(contours[0]) # 면적
cx, cy = m['m10']/m['m00'], m['m01']/m['m00']
# [함수 사용] length=cv.arcLength(curve, closed)
# curve=점 배열, closed=True이면 끝점과 시작점도 연결한다. 점 개수가 아니라 거리 합인 실수 길이를 반환한다.
perimeter = cv.arcLength(contours[0], True)
roundness = 4.0*np.pi*area/(perimeter*perimeter)
# [출력 형식 문제] 문자열 '/n'은 줄바꿈이 아니다. '/n중점=' 같은 글자가 한 줄에 그대로 찍힌다.
# 줄바꿈 특수문자는 역슬래시와 영문 n을 연결한 표기이다. 아래 원래 실행문은 유지했다.
# 면적/cx/cy/둘레/둥근 정도는 실수로 출력되고 좌표 순서는 (x,y)이다.
print('면적= ', area, '/n중점=(', cx, cy, ')', '/n둘레= ', perimeter, '/n둥근정도= ', roundness)
#%%
img3=cv.cvtColor(img, cv.COLOR_GRAY2BGR)
# 여기 epsilon=8, True=닫힌 곡선. 볼록 껍질처럼 반드시 볼록한 결과를 만드는 함수는 아니다.
# ndarray인 근사 좌표와 [ndarray] 리스트는 type/len이 다르다. 아래 출력 주석을 확인한다.
# [함수 사용] approx=cv.approxPolyDP(curve, epsilon, closed)
# epsilon=최대 허용 거리 오차, closed=닫힌 곡선 여부. 근사 다각형의 꼭짓점 배열을 반환한다.
# [PPT 인자 보충] 영역 특징 12쪽: epsilon은 원래 곡선과 근사 곡선의 허용 거리. closed=True는 닫힌 곡선이며 원래 외곽선 배열과 근사 결과는 별도이다.
contour_approxP = cv.approxPolyDP(contours[0],8, True)
# [출력] <class 'numpy.ndarray'> 다음 <class 'list'>. 그 뒤 len은 근사 꼭짓점 수 K이다.
# contour_approxP.shape=(K,1,2), len([contour_approxP])는 1이다. 아래 코드는 list 길이는 출력하지 않는다.
print(type(contour_approxP)); print(type([contour_approxP]));
print(len(contour_approxP))
cv.drawContours(img3, [contour_approxP], -1, (0, 255,0),2)

# approxPolyDP의 오차 기반 단순화와 목적이 다르다. reshape는 저장 축만 바꾸고 좌표를 바꾸지 않는다.
# [함수 사용] hull=cv.convexHull(points, clockwise=False, returnPoints=True)
# points=점 배열. 기본은 볼록 껍질 점 좌표를 반환하고 returnPoints=False이면 원래 점의 인덱스를 반환한다.
# [PPT 인자 보충] 영역 특징 13쪽: returnPoints=False면 입력 점의 인덱스를 반환한다. clockwise는 좌표계의 방향 기준이며 영상의 y축은 화면 아래로 증가함을 구분한다.
hull = cv.convexHull(contours[0])
# [출력 형식] hull.shape=(K,1,2), hull은 K개의 (x,y) 좌표, type은 numpy.ndarray, len은 K이다.
# reshape(1,K,2) 후 점 좌표/총 원소 수는 같지만 len(hull)은 1로 바뀐다. len은 항상 첫 축의 길이이다.
print(hull.shape); print(hull);print(type(hull)) 
print(len(hull))
# [함수 사용] a.reshape(newshape)
# 원소 수를 유지하며 모양만 바꾼다. -1은 자동 크기 추정이며 전치와 다르다. 복사 없이 메모리를 공유할 수 있다.
hull=hull.reshape(1, hull.shape[0], hull.shape[2])
cv.drawContours(img3, hull, -1, (0, 0, 255),2)

# minAreaRect의 회전 가능한 최소 면적 상자와 다르다. w/h는 오른쪽 아래 좌표가 아니다.
# [함수 사용] x,y,w,h=cv.boundingRect(points)
# 점 배열 또는 0이 아닌 픽셀을 감싸는 축에 평행한 사각형을 반환한다. w·h는 끝 좌표가 아니라 너비·높이이다.
rect=cv.boundingRect(contours[0])
# [출력 형식] (x,y,w,h) 튜플 다음 <class 'tuple'>이 나온다. w/h는 오른쪽 아래 좌표가 아니다.
# boundingRect는 축에 평행한 최소 상자이다. 회전할 수 있는 minAreaRect와 구별한다.
print(rect);print(type(rect)) 
# [함수 사용] cv.rectangle(img, pt1, pt2, color, thickness=1)
# pt1·pt2=(x,y) 꼭짓점, color=BGR 색. img에 그리며 -1은 채우기이다. (x,y,w,h) 사각형 인자 형식도 있다.
# [PPT 인자 보충] OpenCV 기본 30쪽: lineType은 선 연결·안티앨리어싱, shift는 좌표의 소수 비트 수. thickness<0이면 채우며 img를 직접 수정한다.
cv.rectangle(img3, rect, (255, 0,0), 2)

# 중심/반지름은 실수이고 면적 중심과 같다는 보장은 없다.
# 현재 np.uint8 좌표 변환은 255 초과 시 순환 가능. 원래 실행문과 출력 수치의 차이를 확인한다.
# [함수 사용] center,radius=cv.minEnclosingCircle(points)
# points=외곽선 점 배열. 모든 점을 포함하는 최소 원의 중심 (실수 x,y)과 실수 반지름을 반환한다.
center, radius = cv.minEnclosingCircle(contours[0])
# [출력 형식] center= (실수 x,실수 y), 다음 줄 Radius= 실수 반지름이다. 여기 '\n'은 실제 줄바꿈이다.
# 최소 둘러싼 원의 중심은 면적 중심(cx,cy)과 같다고 보장되지 않는다.
# 아래 np.uint8(center)는 큰 좌표를 순환시킬 수 있다. 출력한 실수와 그려진 정수 좌표의 차이를 확인한다.
print('center= ', center, '\nRadius= ', radius)
# [함수 사용] cv.circle(img, center, radius, color, thickness=1)
# center=(x,y), radius=반지름, color=BGR 색. img에 직접 그리며 thickness=-1이면 원 내부를 채운다.
# [PPT 인자 보충] OpenCV 기본 37쪽: lineType=LINE_4/LINE_8/LINE_AA는 선 연결·안티앨리어싱 방식. shift는 좌표의 소수 비트 수이며 thickness<0이면 채운다.
cv.circle(img3, np.uint8(center), np.uint8(radius), (0, 255, 255), 2)


cv.imshow("Horse with Contours", img3)
cv.waitKey()

cv.destroyAllWindows()

# 반사에서는 7번째 값의 부호가 바뀔 수 있다. moments 딕셔너리에 자동 포함되지는 않는다.
