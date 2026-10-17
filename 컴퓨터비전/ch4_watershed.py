# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 10:49:25 2023

@author: BigData
"""

# ==================================================================
# [선행 이론] 마커 기반 워터셰드가 필요한 이유와 조건
# 근거: 4장 영역 분할_2 3,7~13쪽
# 읽는 순서: 선행 이론 → 관련 코드 앞의 서술형 요약 → 계산·출력 주석.
# PPT의 배경 이론과 현재 코드의 구현 범위를 구분하여 공부한다.
#
# 1. 지형·유역·분수령이라는 비유
# 빗물이 같은 곳으로 모이는 분지를 유역으로, 서로 다른 유역을 가르는 능선을 분수령으로 본다.
# 영상의 변화가 큰 경계가 지형의 높은 곳에 대응한다고 생각한다.
# 낮은 곳에서 물을 채울 때 서로 다른 분지의 물이 만나려는 곳에 댐을 세우면 분할 경계가 된다.
# 작은 잡음이 지역 최소점을 많이 만들면 너무 잘게 나뉘는 과분할(over-segmentation)이 생길 수 있다.
#
# 2. 마커가 과분할을 제어하는 원리
# 어디가 확실한 전경/배경인지 시작 번호를 주고 나머지 소속을 경계 계산으로 정한다.
# 전경의 내부만 남기도록 침식하면 확실한 fg, 전경을 팽창한 바깥을 반전하면 확실한 bg를 준비할 수 있다.
# 두 확정 영역 사이 띠는 미확정 0이다. 배경 0이라는 이진 영상 약속과 워터셰드 마커의 0은 뜻이 다르다.
# 초기 전경 255/배경 128은 이 코드가 선택한 두 양수 영역 번호이지 확률이나 함수의 고정 전용값이 아니다.
# 같은 번호를 여러 물체에 주면 물체마다 다른 고유 레이블을 부여한 것과 다르다.
#
# 3. 계산 자료형과 최종 표시
# OpenCV 입력은 uint8 BGR 영상, 마커는 같은 높이·너비의 int32 단일 채널이다.
# 결과 경계는 -1이어서 unsigned 마커로 처리하면 안 된다. 초기 미확정 0과 최종 경계 -1도 구분한다.
# clip→uint8은 표시를 위한 변환이며 경계 -1을 0으로 바꾼다. 변환 뒤 -1을 찾으면 경계를 잃는다.
# 영상 바깥 테두리도 경계 -1로 설정될 수 있다. 화면 테두리를 모두 물체 경계로 해석하지 않는다.
#
# 4. 한계와 전처리의 역할
# 마커가 잘못 확정되면 경계 계산만으로 모두 복구하지 못한다. 약한 경계와 그림자도 결과에 영향을 준다.
# 평활화는 작은 잡음을 줄일 수 있지만 실제 경계도 약화시킨다. 색 마스크·마커·최종 경계를 따로 관찰한다.
# 침식/팽창 반복은 확정 영역의 크기와 미확정 띠를 바꾸며, 클수록 정확하다는 관계는 아니다.
# ==================================================================


import cv2 as cv
import numpy as np

#%% 영상 읽기 및 흑백 변환
img=cv.imread("group.jpg")
gray=cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# Otsu 반전 이진화로 물체와 배경 구분
# [이론/값 문제: 영역 분할_2 PPT 7~12쪽]
# 어두운 물체를 흰 전경으로 삼으므로 BINARY_INV이다. Otsu가 현재 group.jpg에서 t=90.0을 선택한다.
# gray<=90이면 전경 255, gray>90이면 배경 0이 된다. 반전 여부와 부등호를 함께 본다.
# binary.shape=(319,480), dtype=uint8이다. threshold는 (임계값,결과영상) 두 값을 반환한다.
t, binary=cv.threshold(gray, 0,255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
print(t)  # 출력: 90.0 (현재 group.jpg의 Otsu 임계값)

cv.imshow("Color", img); cv.imshow("Gray", gray); cv.imshow("Binary", binary)
cv.waitKey(); cv.destroyAllWindows()
#%% 침식으로 확실한 전경 만들기
# [서술형] 마커 기반 워터셰드 분할
# 관련 PPT: 영역 분할_2 7~13쪽
# 특징: 지형의 분지가 만나는 곳에 경계를 세우며, 전경/배경 마커로 시작 영역을 제어한다.
# 장점: 서로 접한 후보의 경계를 정하고, 마커로 불필요한 지역 최소점의 과분할을 줄일 수 있다.
# 단점: 잡음·약한 경계·잘못된 확정 마커에 민감하다. 물체 의미나 모든 그림자를 자동으로 구별하지 않는다.
# 활용: 이진 후보에서 확실한 내부/배경을 알 수 있는 물체 윤곽 분리.
# 한 문장 답안: 워터셰드는 마커에서 영역을 확장해 경계를 정하지만 결과 품질은 초기 마커와 영상 경계에 좌우된다.
# [계산·코드 연결]
# fg=침식한 전경, bg=팽창한 물체 바깥. 두 확정 영역 사이 0 띠가 미확정 영역이다.
# 침식 반복 증가→fg 축소, 팽창 반복 증가→확실한 bg 시작점이 멀어짐. 작은 물체는 사라질 수 있다.
# 현재 group 예제는 침식 8/팽창 6, PPT는 6/6. 반복 횟수가 크다고 정확도가 보장되지 않는다.
kernel=cv.getStructuringElement(cv.MORPH_RECT, (3,3))
fg=cv.erode(binary,kernel, iterations= 8)
cv.imshow("Foreground", fg); cv.waitKey()

# 팽창 후 반전하여 확실한 배경 만들기
bg=cv.dilate(binary, kernel, iterations=6)
cv.imshow("bg", bg); 
cv.waitKey()
t, bg=cv.threshold(bg, 1, 128, cv.THRESH_BINARY_INV)
cv.imshow("Background", bg); cv.waitKey()

# 전경과 배경을 합쳐 초기 마커 생성
# [마커 계산 문제] fg는 침식한 전경 0/255, bg는 팽창 바깥 128/안쪽 0이다.
# fg와 확실한 bg가 겹치지 않으므로 합은 0/128/255이다. 단순한 밝기 합이 아니라 영역 번호 준비이다.
# 같은 양수 번호는 같은 영역으로 취급한다. 모든 물체에 255 하나를 쓰는 예제는 물체별 고유 번호를 주지 않는다.
# int32로 바꿔야 계산 후 음수 경계 -1을 저장할 수 있다. 입력 영상은 같은 높이·너비의 uint8 3채널이다.
markers=fg+bg
cv.imshow("Markers", markers)
cv.waitKey()

#%% 워터셰드 실행 (경계는 -1)
# [계산·자료형] 초기 번호는 전경 255, 배경 128, 미확정 0이다. 확률이나 단순 명암이 아니다.
# 입력은 uint8 BGR, 마커는 같은 높이·너비의 int32 단일 채널이다. 계산 뒤 경계는 -1이다.
# 같은 양수 번호는 같은 영역이다. 모든 물체에 255를 준 예제는 물체마다 다른 고유 번호를 주지 않는다.
markers=np.int32(markers)
markers=cv.watershed(img, markers)
# [출력 여부 문제] 여기서 max=255, min=-1인 계산 결과를 확인할 수 있다.
# 아래 np.max/np.min은 print가 없는 표현식이므로 .py 전체 실행에서는 콘솔에 출력되지 않는다.
# Interactive 셀은 마지막 표현식을 보여 줄 수 있어 .py 실행과 표시 결과가 다르다.
np.max(markers)
np.min(markers)
#%% 흰 바탕에 검은 윤곽 표시
dst=np.ones(img.shape, np.uint8)*255
dst[markers==-1] = 0
cv.imshow("Watershed", dst)
# 원본 영상에 파란 윤곽 표시
img[markers==-1]=[255,0,0]
cv.imshow("Color&Boundry", img)
cv.waitKey()

# 영역 분할 결과 표시
# [표시 변환 문제] clip은 -1을 0으로 바꾼다. 여기 이후 markers는 uint8, 값 0/128/255이다.
# 이 변환 뒤 markers==-1은 모두 False가 되므로 경계 표시는 반드시 변환 전에 해야 한다.
markers=np.uint8(np.clip(markers, 0, 255))
cv.imshow("Segmentation", markers)
cv.waitKey() 

cv.destroyAllWindows()
