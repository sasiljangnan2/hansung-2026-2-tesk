
# ==================================================================
# [선행 이론] 손 분할을 색 후보·마커·최종 경계로 나누어 이해하기
# 근거: 영상처리_1 11~22쪽 / 영상처리_2 25~27쪽 / 영역 분할_2 7~14쪽
# 읽는 순서: 선행 이론 → 관련 코드 앞의 서술형 요약 → 계산·출력 주석.
# PPT의 배경 이론과 현재 코드의 구현 범위를 구분하여 공부한다.
#
# 1. 색 공간을 바꾸는 이유
# 같은 색도 조명이 달라지면 BGR 숫자가 달라진다. 표현을 바꾸면 필요한 성분을 따로 다루기 쉽다.
# 색 공간 변환은 새로운 물체 정보를 만드는 것이 아니라 같은 색을 다른 좌표로 나타내는 것이다.
# RGB/BGR: 세 빛 성분을 직접 표현한다. 화면 표시와 저장에 편리하지만 밝기가 여러 채널에 섞인다.
# HSV: 색상 H, 채도 S, 밝기 V로 나눈다. 특정 색을 고르거나 밝기만 바꾸기에 편리하다.
# Lab: 밝기 L과 녹색↔빨강 a, 파랑↔노랑 b의 색 성분으로 나눈다.
# Lab의 이론 좌표와 OpenCV uint8 저장값의 범위·오프셋은 같지 않으므로 수치를 그대로 섞지 않는다.
#
# 2. HSV의 세 성분과 주의점
# H(hue)는 색상 원의 위치, S(saturation)는 무채색에서 벗어난 정도, V(value)는 밝기 성분이다.
# uint8 일반 BGR2HSV에서 H는 0~179, S/V는 0~255이다. 다른 자료형·변환 옵션은 범위를 확인한다.
# 색상 원은 양 끝이 이어진다. 붉은색은 낮은 H와 높은 H의 두 구간으로 잡을 수 있다.
# S가 작으면 거의 회색이라 H의 의미가 약하다. 매우 어두우면 색/채도 정보를 안정적으로 믿기 어렵다.
# H의 최대 허용값이 179라고 모든 영상의 np.max(H)가 반드시 179인 것은 아니다.
#
# 3. 현재 손 전처리의 가정
# 피부는 붉은 계열 후보에 자주 속한다는 가정으로 HSV 두 구간을 OR하고 Lab a 기준을 AND한다.
# 밝은 피부는 채도가 낮을 수 있고 어두운 손/그림자는 색이 겹칠 수 있으므로 범위만으로 완전 분리가 어렵다.
# 조건을 더 엄격히 하면 그림자뿐 아니라 손바닥도 빠질 수 있다. 한 사진에 맞춘 숫자를 보편 규칙으로 외우지 않는다.
# 메디안은 이웃 중앙값으로 작은 점 잡음을 완화한다. 가우시안 가중 평균과는 다른 비선형 필터이다.
# 3x3 이진 값 9개 중 다섯 번째를 고르므로 출력도 0/255를 유지하지만 작은 손 형상은 바뀔 수 있다.
#
# 4. 손 후보와 경계 계산은 별도의 단계
# 피부 마스크를 그대로 윤곽선으로 쓰는 대신, 침식/팽창으로 확실한 전경과 배경을 만든다.
# 마커가 0인 미확정 띠에서 워터셰드가 경계를 결정한다. 잘못된 확정 마커까지 모두 고치지는 못한다.
# 침식 반복을 늘리면 손 내부 마커가 줄고 팽창 반복을 늘리면 확실한 배경 시작점이 멀어진다.
# 초기 255/128/0은 영역 번호/미확정이고 결과 -1은 최종 경계이다. 계산은 int32, 표시 후에는 uint8이다.
# 현재는 메디안 필터를 마스크와 BGR 입력에 적용한다. 닫힘·구멍 채우기·수동 칠하기를 추가한 코드는 아니다.
# ==================================================================

# [1장 연결 이론] 분할·환경 변화·고전 방법의 한계 [1장 소개 PPT 3,10,17~18쪽]
# 이 예제의 목표는 손/배경 픽셀의 영역과 경계를 나누는 분할이다. 손의 행동을 분류하는 코드는 아니다.
# 피부의 조명이 달라지면 같은 손의 색도 바뀌고, 배경 그림자가 손과 비슷한 색으로 관측될 수 있다.
# 관측 색만으로 원래 물체·조명을 유일하게 정하기 어려운 것은 비전의 역문제와 연결된다.
# 수동 색 범위·마커·경계 계산을 설계하므로 주로 고전 비전의 흐름이며 학습된 신경망은 사용하지 않는다.
# 장점: 단계별 결과와 조건을 직접 확인할 수 있다. 한계: 다른 사진·조명에서는 조건을 다시 맞출 수 있다.
# 조건을 좁히면 그림자뿐 아니라 실제 손도 빠질 수 있어 누락·오검출 사이의 절충이 생긴다.
# 필터는 잡음을 완화하며 그림자의 의미를 이해해 삭제하지 않는다. 잘못 확정한 마커도 결과에 영향을 준다.
# 한 장에서 잘 된다는 것과 다양한 환경에 일반화된다는 것을 구별한다. 현재 실행 조건은 그대로 유지했다.

import cv2 as cv
import numpy as np

#%% 영상 읽기 및 HSV 피부색 검출
# [함수 사용: 입출력·창]
# img=cv.imread(path, flags=cv.IMREAD_COLOR): 파일→배열. 기본 BGR, 읽기 실패 시 None인지 확인한다.
# cv.imshow(winname,img): 이름이 winname인 창에 배열을 표시한다. 저장 함수는 아니다.
# cv.destroyAllWindows(): OpenCV 창을 닫는다. 캡처 자원 해제는 cap.release()이다.
# key=cv.waitKey(delay=0): GUI 이벤트·키 처리. 0은 계속 대기, 양수는 ms, 입력 없으면 보통 -1이다.
img=cv.imread("hand_sample2.jpg")
# [함수 사용] dst=cv.cvtColor(src, code)
# src=입력 영상, code=색 변환 상수. 예: BGR2GRAY·BGR2HSV·HSV2BGR. 변환 영상을 반환한다.
# [PPT 인자 보충] 영상처리_1 16쪽: dstCn=0은 출력 채널 수 자동 결정. 출력 높이·너비와 깊이는 같고 채널 수는 변환에 따라 달라진다.
gray=cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# [서술형] HSV·Lab을 조합한 피부색 후보 분석
# 관련 PPT: 영상처리_1 11~22쪽 / 영역 분할_2 14쪽
# 특징: HSV의 붉은색 두 구간을 합친 뒤 Lab a 성분 조건으로 후보를 추가 제한한다.
# 장점: 밝기만으로 고를 때보다 피부의 색 성분을 이용하고 여러 색 조건을 조합할 수 있다.
# 단점: 조건이 엄격하면 밝은 피부가 빠지고 느슨하면 그림자/비슷한 배경도 남는다. 사진 의존적이다.
# 활용: 손 워터셰드에 줄 초기 마스크 후보 준비. 피부색만으로 최종 경계를 확정하지 않는다.
# 한 문장 답안: 피부색 후보는 색 정보를 활용하지만 조명과 배경색이 겹치면 누락·오검출의 절충이 필요하다.
# [계산·코드 연결]
# H는 원형이므로 0 부근과 179 부근의 두 구간을 OR한다. Lab 조건은 AND로 함께 만족시킨다.
# 현재 H=0~25/160~179, S>=40, V 전체, Lab a>=134. 이전에 정한 실행 조건은 유지했다.
hsv=cv.cvtColor(img, cv.COLOR_BGR2HSV)

# 밝은 피부도 포함하고, 아래 Lab 조건으로 배경 그림자를 추가 제외
# [함수 사용] mask=cv.inRange(src, lowerb, upperb)
# 각 채널의 하한·상한을 모두 만족하면 255, 아니면 0인 uint8 마스크를 반환한다. 경계값도 포함한다.
mask1=cv.inRange(hsv, (0,40,0), (25,255,255))
mask2=cv.inRange(hsv, (160,40,0), (179,255,255))

# [논리식 문제: 영상처리_1 PPT 19~22쪽]
# mask1 OR mask2는 붉은색 원의 양 끝 중 어느 쪽에 속해도 후보로 남긴다.
# 뒤의 AND skin은 HSV와 Lab 조건을 모두 만족한 후보만 남긴다.
# 현재 mask1/mask2/binary/skin.shape=(367,426), dtype=uint8, 값은 0 또는 255이다.
# 피부 후보는 전경의 초기 추정이며 어두운 손과 비슷한 색의 그림자를 완벽히 분류하는 정답 마스크는 아니다.
binary=mask1 | mask2

# 이 사진에서 붉은 성분이 약한 그림자 제외 (a 채널 기준 134)
lab=cv.cvtColor(img, cv.COLOR_BGR2LAB)
skin=cv.inRange(lab[:,:,1], 134, 255)
# [함수 사용] dst=cv.bitwise_and(src1, src2, mask=mask)
# 원소별 비트 AND 결과를 반환한다. 선택 인자 mask는 0이 아닌 위치만 계산·기록하게 하는 uint8 마스크이다.
binary=cv.bitwise_and(binary, skin)
# [함수 사용] kernel=cv.getStructuringElement(shape, ksize)
# shape=RECT·ELLIPSE·CROSS 등의 모양, ksize=(너비,높이). 침식·팽창에 줄 구조 요소를 반환한다.
# [PPT 인자 보충] 영역 분할 9쪽: shape=RECT/CROSS/ELLIPSE, ksize=(너비,높이). anchor=(-1,-1)은 중심이며 CROSS는 anchor 위치에 따라 십자 모양도 달라진다.
kernel=cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))

# [서술형] 메디안 필터 분석
# 관련 PPT: 영상처리_2 25~27쪽
# 특징: 이웃 값을 정렬한 중앙값을 선택하는 비선형 필터이다.
# 장점: 소금·후추 같은 점 잡음에 유리하고 평균 대신 기존 값을 골라 이진 마스크의 0/255를 유지한다.
# 단점: 작은 돌기·가는 손가락·유용한 세부도 사라질 수 있으며 넓은 그림자를 선택적으로 없애지 못한다.
# 활용: 손 마스크의 점 잡음과 작은 거친 부분 완화, 워터셰드 입력의 작은 잡음 감소.
# 한 문장 답안: 메디안은 중앙값으로 충격성 잡음을 줄이지만 작은 형상도 변형할 수 있다.
# [계산·코드 연결]
# 3x3의 9개 중 다섯 번째 값이다. 0이 5개,255가 4개면 결과 0이다.
# 가우시안은 가중 평균인 선형 필터여서 중간 밝기가 나올 수 있다. 두 필터의 원리를 구분한다.
# [함수 사용] dst=cv.medianBlur(src, ksize)
# ksize=1보다 큰 홀수 커널 한 변 길이(예: 3). 주변 중앙값으로 바꾼 같은 크기의 영상을 반환한다.
binary=cv.medianBlur(binary, 3)

cv.imshow("Color", img); cv.imshow("Gray", gray); cv.imshow("Binary", binary)
cv.waitKey(); cv.destroyAllWindows()

#%% 침식·팽창으로 마커 준비
kernel=cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))
# [서술형] 손 워터셰드 연결
# 관련 PPT: 영역 분할_2 9~14쪽
# 특징: 확실한 손 내부와 배경 사이의 미확정 영역에서 영상 경계를 결정한다.
# 장점: 색 마스크의 바깥선을 그대로 그리지 않고 경계 계산으로 다듬을 수 있다.
# 단점: 잘못된 확정 마커까지 모두 복구하지 못하므로 전처리 누락·그림자 오표시가 최종 경계에 영향을 준다.
# 활용: 현재 손 후보 마스크에서 외곽 경계 결정. 일반 워터셰드의 원리·장단점은 ch4_워터셰드.py 참고.
# 한 문장 답안: 손 워터셰드는 색 후보를 마커로 활용하므로 경계 개선과 초기 색 검출의 한계를 함께 가진다.
# 반복 증가 시 fg는 더 작아지고, bg 팽창 범위는 더 커져 두 확정 영역 사이 미확정 띠가 넓어진다.
# [함수 사용] dst=cv.erode(src, kernel, iterations=1)
# src=입력, kernel=구조 요소, iterations=침식 반복 횟수. 같은 크기의 결과를 반환한다.
# [PPT 인자 보충] 영상처리_2 7쪽: anchor=(-1,-1)은 커널 중심. borderType/borderValue는 영상 바깥 처리이며 iterations는 같은 연산의 반복 수이다.
fg=cv.erode(binary,kernel, iterations=5)
cv.imshow("Foreground", fg); cv.waitKey()

# 손 경계 주변에 좁은 미확정 영역을 두고 워터셰드로 경계 결정
# [함수 사용] dst=cv.dilate(src, kernel, iterations=1)
# src=입력, kernel=구조 요소, iterations=팽창 반복 횟수. 같은 크기의 결과를 반환한다.
# [PPT 인자 보충] 영상처리_2 7쪽: anchor=(-1,-1)은 커널 중심. borderType/borderValue는 영상 바깥 처리이며 iterations는 같은 연산의 반복 수이다.
bg=cv.dilate(binary, kernel, iterations=2)
cv.imshow("bg", bg);
# 팽창한 손 바깥을 배경 마커(128)로 바꾼 뒤 전경과 합침
# [함수 사용] t,dst=cv.threshold(src, thresh, maxval, type)
# thresh=기준값, maxval=이진화 출력값, type=방법. t는 사용한 임계값, dst는 결과이며 OTSU는 기준을 자동 계산한다.
# [PPT 인자 보충] 영상처리_1 29쪽: 비교는 src>thresh이다(같으면 이하 쪽). THRESH_OTSU는 자동 T를 반환하며 초기 thresh는 무시한다.
t, bg=cv.threshold(bg, 1, 128, cv.THRESH_BINARY_INV)
# [워터셰드 값/배열 문제: 영역 분할_2 PPT 9~14쪽]
# 현재 fg는 0/255, 반전한 bg는 0/128, markers는 0/128/255의 (367,426) 배열이다.
# 침식 5를 늘리면 확실한 손 내부가 작아지고, 팽창 2를 늘리면 확실한 배경의 시작점이 멀어진다.
# 두 확정 영역 사이의 0 띠에서 경계를 결정한다. 잘못된 확정 마커를 워터셰드가 모두 고쳐 주지는 않는다.
markers=fg+bg
cv.imshow("Markers", markers)
cv.waitKey()

#%% 워터셰드 실행
# 경계 계산과 표시 [영역 분할_2 PPT 10~14쪽]
# 전경 255/배경 128/미확정 0을 int32로 넘기고 결과 -1인 픽셀을 경계로 그린다.
# 현재 전처리는 medianBlur(중앙값)이며 GaussianBlur(가중 평균)와 다른 필터이다.
# [함수 사용] dst=np.int32(values)
# 32비트 부호 있는 정수로 변환한다. 워터셰드 마커의 영역 번호와 경계 -1을 담는 자료형이다.
markers=np.int32(markers)
# 메디안 필터로 잡음을 줄인 뒤 워터셰드 실행
blur=cv.medianBlur(img, 3)
# [함수 사용] markers=cv.watershed(img, markers)
# img=uint8 3채널, markers=같은 높이·너비의 int32 단일 채널. 양수=영역·0=미확정이며 markers를 갱신하고 경계를 -1로 둔다.
# [PPT 인자 보충] 영역 분할 10쪽: image는 8비트 3채널, markers는 같은 높이·너비의 int32. 양수=알려진 영역, 0=미확정, 결과 -1=분할 경계이며 markers 자체도 바뀐다.
markers=cv.watershed(blur, markers)
# [출력 여부 문제] 이 위치의 markers는 int32, max=255, min=-1이다.
# np.max/np.min 단독 줄은 .py에서 출력되지 않는다. 아래 표시용 uint8 변환 후에는 min=0이 된다.
# -1은 최종 경계이며 초기 미확정 0과 다르다. BGR [255,0,0]은 파란색, 흰 dst의 경계값 0은 검정이다.
# [함수 사용] value=np.max(a, axis=None)
# axis 생략 시 전체 원소 중 최댓값, 지정 시 해당 축의 최댓값을 반환한다.
np.max(markers)
# [함수 사용] value=np.min(a, axis=None)
# axis 생략 시 전체 원소 중 최솟값, 지정 시 해당 축의 최솟값을 반환한다.
np.min(markers)


#%% 경계 및 영역 분할 결과 표시
# [함수 사용] a=np.ones(shape, dtype=float)
# shape=배열 크기. 1로 채운 ndarray를 만든다. *255를 하면 흰 배경을 준비할 수 있다.
dst=np.ones(img.shape, np.uint8)*255
dst[markers==-1] = 0
cv.imshow("Watershed", dst)
img[markers==-1]=[255,0,0]
cv.imshow("Color&Boundry", img)
cv.waitKey()

# [함수 사용] dst=np.uint8(values)
# 값을 8비트 부호 없는 정수로 변환한다. 안전한 범위 표시에는 먼저 clip을 고려하며 좌표값을 8비트로 제한하지 않도록 주의한다.
# [함수 사용] dst=np.clip(a, a_min, a_max)
# a_min 미만을 하한, a_max 초과를 상한으로 잘라 반환한다. dtype 변환은 별도이며 자동으로 uint8이 되지는 않는다.
markers=np.uint8(np.clip(markers, 0, 255))
cv.imshow("Segmentation", markers)
cv.waitKey()

cv.destroyAllWindows()
