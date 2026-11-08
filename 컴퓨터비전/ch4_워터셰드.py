# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 10:49:25 2023

@author: BigData
"""

#%% 개념 요약 — 정의·장점·단점, 함수 사용법은 해당 코드 옆 참고

#%% [개념] 마커 기반 워터셰드 분할
# 정의: 확정된 영역을 마커로 주고 영역을 확장하며 만나는 경계에서 영상을 분할하는 방법이다.
# 장점: 서로 접한 후보의 경계를 정하고, 마커로 불필요한 지역 최소점의 과분할을 줄일 수 있다.
# 단점: 잡음·약한 경계·잘못된 확정 마커에 민감하다. 물체 의미나 모든 그림자를 자동으로 구별하지 않는다.
# PPT: 영역 분할_2 7~13쪽

#region 핵심 이론·함수 참고
# [핵심 이론] 마커 기반 워터셰드가 필요한 이유와 조건
# 근거: 4장 영역 분할_2 3,7~13쪽
# 1. 지형·유역·분수령이라는 비유
# 빗물이 같은 곳으로 모이는 분지를 유역으로, 서로 다른 유역을 가르는 능선을 분수령으로 본다.
# 2. 마커가 과분할을 제어하는 원리
# 어디가 확실한 전경/배경인지 시작 번호를 주고 나머지 소속을 경계 계산으로 정한다.
# 두 확정 영역 사이 띠는 미확정 0이다. 배경 0이라는 이진 영상 약속과 워터셰드 마커의 0은 뜻이 다르다.
# 3. 계산 자료형과 최종 표시
# OpenCV 입력은 uint8 BGR 영상, 마커는 같은 높이·너비의 int32 단일 채널이다.
# 결과 경계는 -1이어서 unsigned 마커로 처리하면 안 된다. 초기 미확정 0과 최종 경계 -1도 구분한다.
# 4. 한계와 전처리의 역할
# 마커가 잘못 확정되면 경계 계산만으로 모두 복구하지 못한다. 약한 경계와 그림자도 결과에 영향을 준다.
# 침식/팽창 반복은 확정 영역의 크기와 미확정 띠를 바꾸며, 클수록 정확하다는 관계는 아니다.


#endregion

import cv2 as cv
from cv_assets import read_image
import numpy as np

#%% 영상 읽기 및 흑백 변환
# [함수 사용: 입출력·창]
# img=cv.imread(path, flags=cv.IMREAD_COLOR): 파일→배열. 기본 BGR, 읽기 실패 시 None인지 확인한다.
# cv.imshow(winname,img): 이름이 winname인 창에 배열을 표시한다. 저장 함수는 아니다.
# cv.destroyAllWindows(): OpenCV 창을 닫는다. 캡처 자원 해제는 cap.release()이다.
# key=cv.waitKey(delay=0): GUI 이벤트·키 처리. 0은 계속 대기, 양수는 ms, 입력 없으면 보통 -1이다.
img = read_image("group.jpg")
# [함수 사용] dst=cv.cvtColor(src, code)
# src=입력 영상, code=색 변환 상수. 예: BGR2GRAY·BGR2HSV·HSV2BGR. 변환 영상을 반환한다.
# [PPT 인자 보충] 영상처리_1 16쪽: dstCn=0은 출력 채널 수 자동 결정. 출력 높이·너비와 깊이는 같고 채널 수는 변환에 따라 달라진다.
gray=cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# Otsu 반전 이진화로 물체와 배경 구분
# [이론/값 문제: 영역 분할_2 PPT 7~12쪽]
# 어두운 물체를 흰 전경으로 삼으므로 BINARY_INV이다. Otsu가 현재 group.jpg에서 t=90.0을 선택한다.
# gray<=90이면 전경 255, gray>90이면 배경 0이 된다. 반전 여부와 부등호를 함께 본다.
# binary.shape=(319,480), dtype=uint8이다. threshold는 (임계값,결과영상) 두 값을 반환한다.
# [함수 사용] t,dst=cv.threshold(src, thresh, maxval, type)
# thresh=기준값, maxval=이진화 출력값, type=방법. t는 사용한 임계값, dst는 결과이며 OTSU는 기준을 자동 계산한다.
# [PPT 인자 보충] 영상처리_1 29쪽: 비교는 src>thresh이다(같으면 이하 쪽). THRESH_OTSU는 자동 T를 반환하며 초기 thresh는 무시한다.
t, binary=cv.threshold(gray, 0,255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
print(t)  # 출력: 90.0 (현재 group.jpg의 Otsu 임계값)

cv.imshow("Color", img); cv.imshow("Gray", gray); cv.imshow("Binary", binary)
cv.waitKey(); cv.destroyAllWindows()
#%% 침식으로 확실한 전경 만들기
# fg=침식한 전경, bg=팽창한 물체 바깥. 두 확정 영역 사이 0 띠가 미확정 영역이다.
# 침식 반복 증가→fg 축소, 팽창 반복 증가→확실한 bg 시작점이 멀어짐. 작은 물체는 사라질 수 있다.
# 현재 group 예제는 침식 8/팽창 6, PPT는 6/6. 반복 횟수가 크다고 정확도가 보장되지 않는다.
# [함수 사용] kernel=cv.getStructuringElement(shape, ksize)
# shape=RECT·ELLIPSE·CROSS 등의 모양, ksize=(너비,높이). 침식·팽창에 줄 구조 요소를 반환한다.
# [PPT 인자 보충] 영역 분할 9쪽: shape=RECT/CROSS/ELLIPSE, ksize=(너비,높이). anchor=(-1,-1)은 중심이며 CROSS는 anchor 위치에 따라 십자 모양도 달라진다.
kernel=cv.getStructuringElement(cv.MORPH_RECT, (3,3))
# [함수 사용] dst=cv.erode(src, kernel, iterations=1)
# src=입력, kernel=구조 요소, iterations=침식 반복 횟수. 같은 크기의 결과를 반환한다.
# [PPT 인자 보충] 영상처리_2 7쪽: anchor=(-1,-1)은 커널 중심. borderType/borderValue는 영상 바깥 처리이며 iterations는 같은 연산의 반복 수이다.
fg=cv.erode(binary,kernel, iterations= 8)
cv.imshow("Foreground", fg); cv.waitKey()

# 팽창 후 반전하여 확실한 배경 만들기
# [함수 사용] dst=cv.dilate(src, kernel, iterations=1)
# src=입력, kernel=구조 요소, iterations=팽창 반복 횟수. 같은 크기의 결과를 반환한다.
# [PPT 인자 보충] 영상처리_2 7쪽: anchor=(-1,-1)은 커널 중심. borderType/borderValue는 영상 바깥 처리이며 iterations는 같은 연산의 반복 수이다.
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
# [함수 사용] dst=np.int32(values)
# 32비트 부호 있는 정수로 변환한다. 워터셰드 마커의 영역 번호와 경계 -1을 담는 자료형이다.
markers=np.int32(markers)
# [함수 사용] markers=cv.watershed(img, markers)
# img=uint8 3채널, markers=같은 높이·너비의 int32 단일 채널. 양수=영역·0=미확정이며 markers를 갱신하고 경계를 -1로 둔다.
# [PPT 인자 보충] 영역 분할 10쪽: image는 8비트 3채널, markers는 같은 높이·너비의 int32. 양수=알려진 영역, 0=미확정, 결과 -1=분할 경계이며 markers 자체도 바뀐다.
markers=cv.watershed(img, markers)
# [출력 여부 문제] 여기서 max=255, min=-1인 계산 결과를 확인할 수 있다.
# 아래 np.max/np.min은 print가 없는 표현식이므로 .py 전체 실행에서는 콘솔에 출력되지 않는다.
# Interactive 셀은 마지막 표현식을 보여 줄 수 있어 .py 실행과 표시 결과가 다르다.
# [함수 사용] value=np.max(a, axis=None)
# axis 생략 시 전체 원소 중 최댓값, 지정 시 해당 축의 최댓값을 반환한다.
np.max(markers)
# [함수 사용] value=np.min(a, axis=None)
# axis 생략 시 전체 원소 중 최솟값, 지정 시 해당 축의 최솟값을 반환한다.
np.min(markers)
#%% 흰 바탕에 검은 윤곽 표시
# [함수 사용] a=np.ones(shape, dtype=float)
# shape=배열 크기. 1로 채운 ndarray를 만든다. *255를 하면 흰 배경을 준비할 수 있다.
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
# [함수 사용] dst=np.uint8(values)
# 값을 8비트 부호 없는 정수로 변환한다. 안전한 범위 표시에는 먼저 clip을 고려하며 좌표값을 8비트로 제한하지 않도록 주의한다.
# [함수 사용] dst=np.clip(a, a_min, a_max)
# a_min 미만을 하한, a_max 초과를 상한으로 잘라 반환한다. dtype 변환은 별도이며 자동으로 uint8이 되지는 않는다.
markers=np.uint8(np.clip(markers, 0, 255))
cv.imshow("Segmentation", markers)
cv.waitKey() 

cv.destroyAllWindows()
