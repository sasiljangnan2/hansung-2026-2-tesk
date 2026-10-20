# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 14:04:24 2023

@author: BigData
"""

# ==================================================================
# [선행 이론] 구조 요소로 영역의 모양을 다루는 원리
# 근거: 3장 영상처리_2 2~10쪽
# 읽는 순서: 선행 이론 → 관련 코드 앞의 서술형 요약 → 계산·출력 주석.
# PPT의 배경 이론과 현재 코드의 구현 범위를 구분하여 공부한다.
#
# 1. 모폴로지와 구조 요소
# 모폴로지는 영역을 픽셀 집합으로 보고 구조 요소(structuring element)를 옮기며 모양을 처리한다.
# 원영상 A와 작은 집합 B를 생각하며, B의 중심(anchor)을 현재 처리 위치에 둔다.
# 구조 요소의 1인 위치만 비교에 참여한다. 사각형·십자·타원·마름모는 같은 크기여도 결과가 다를 수 있다.
# 이진 전경이 흰색일 때 팽창은 B가 전경에 닿는 위치로 넓히고, 침식은 B 전체가 들어가는 곳만 남긴다.
# 명암 모폴로지에서는 이웃 최댓값/최솟값으로도 해석한다. 이웃 평균인 가우시안과 원리가 다르다.
#
# 2. 네 연산을 구분하기
# 팽창 A⊕B: 작은 홈을 메우고 끊긴 부분을 잇지만 가까운 다른 물체가 붙을 수 있다.
# 침식 A⊖B: 돌기·작은 전경을 줄이지만 가는 선이나 손가락이 사라질 수 있다.
# 열림 A∘B=(A⊖B)⊕B: 침식 후 팽창. 작은 전경 잡음을 없애는 데 사용한다.
# 닫힘 A•B=(A⊕B)⊖B: 팽창 후 침식. 작은 빈틈·구멍을 줄이는 데 사용한다.
# 열림/닫힘은 첫 연산의 크기 변화를 일부 되돌리는 것이지 원래 모양·면적을 정확히 복원하는 연산이 아니다.
# 열림으로 없어진 가는 전경은 팽창해도 원래대로 돌아오지 않을 수 있다.
# 닫힘 역시 구조 요소보다 큰 임의의 구멍을 무조건 채우는 알고리즘은 아니다.
#
# 3. 반복·구조·마커의 관계
# 같은 3x3이라도 반복을 많이 하면 더 넓은 범위를 처리한다. 경계뿐 아니라 작은 유효 형상도 달라진다.
# 워터셰드의 침식은 확실한 내부를, 팽창 후 바깥 영역은 확실한 배경을 얻기 위한 용도이다.
# 화면에서 전경을 검정으로 표현했다면 위의 흰 전경 기준 해석을 그대로 적용하지 않도록 확인한다.
# 현재 예제는 알파가 모양을 담은 PNG의 알파를 이진화한 뒤 5x5 마름모로 팽창·침식·닫힘을 비교한다.
# ==================================================================



#%% [PPT 함수 보충] 모폴로지 조합·그래프 표시
# 관련 PPT: 영상처리_2 2~10쪽. 현재 실행의 닫힘은 dilate→erode이며 아래는 같은 조합의 참고 표기이다.
# [함수 사용: 이론 연결] dst=cv.morphologyEx(src,op,kernel,iterations=1)
# op=MORPH_OPEN은 침식→팽창, MORPH_CLOSE는 팽창→침식. 예: cv.morphologyEx(b,cv.MORPH_CLOSE,se).
# PPT에는 두 연산의 정의와 erode/dilate 조합이 나오며 morphologyEx는 그 조합을 한 번에 표현하는 참고 함수이다(현재 새 실행 없음).
# iterations=2이면 기본 침식/팽창을 각각 두 번 수행하는 뜻이지 열림/닫힘 전체를 단순히 두 번 반복하는 뜻이 아니다.
#
# [가장자리·중심] kernel=None은 기본 3×3 직사각형. 0이 아닌 커널 원소가 이웃에 참여한다.
# anchor=(-1,-1)은 중심이다. 영상 경계에는 바깥 픽셀 규칙(borderType/borderValue)이 필요하며 임의로 0을 넣는 경우와 기본 처리는 다를 수 있다.

import cv2 as cv
from cv_assets import read_image
import matplotlib.pyplot as plt
import numpy as np
import sys

#%%
# 알파 채널을 이용한 이진 영상 [영상처리_1 PPT 15쪽 / 영상처리_2 PPT 8~10쪽]
# IMREAD_UNCHANGED로 투명도까지 보존해야 네 번째 채널 [:,:,3]을 읽을 수 있다.
# 일반 컬러 읽기는 BGR 3채널이므로 같은 인덱스를 사용할 수 없다.
# [함수 사용: 입출력·창]
# img=cv.imread(path, flags=cv.IMREAD_COLOR): 파일→배열. 기본 BGR, 읽기 실패 시 None인지 확인한다.
# cv.imshow(winname,img): 이름이 winname인 창에 배열을 표시한다. 저장 함수는 아니다.
# key=cv.waitKey(delay=0): GUI 이벤트·키 처리. 0은 계속 대기, 양수는 ms, 입력 없으면 보통 -1이다.
# cv.destroyAllWindows(): OpenCV 창을 닫는다. 캡처 자원 해제는 cap.release()이다.
img = read_image("JohnHancocksSignature.png", cv.IMREAD_UNCHANGED)

imgc = read_image("JohnHancocksSignature.png")

if img is None:
    sys.exit("No File exists.")
    
# [출력 확인: 현재 서명 PNG]
# img는 BGRA 네 채널, imgc는 BGR 세 채널이다. 알파=0은 투명, 255는 불투명이다.
# 두 배열을 출력하면 가장자리의 0과 ...가 보인다. 생략된 내부까지 전부 0이라는 뜻은 아니다.
# 이 파일의 BGR은 전부 0(검정)이며 서명 모양은 알파 채널에 있으므로 max(imgc)=0이다.
print(type(img))  # 출력: <class 'numpy.ndarray'>

print(img.shape)  # 출력: (525, 1920, 4)
print(imgc.shape)  # 출력: (525, 1920, 3)
print(img)
print(imgc)
# [함수 사용] value=np.max(a, axis=None)
# axis 생략 시 전체 원소 중 최댓값, 지정 시 해당 축의 최댓값을 반환한다.
print(np.max(imgc))  # 출력: 0 (현재 PNG의 BGR)
print(np.max(img[:,:,3]))  # 출력: 255 (알파 최댓값)
# [함수 사용] value=np.min(a, axis=None)
# axis 생략 시 전체 원소 중 최솟값, 지정 시 해당 축의 최솟값을 반환한다.
print(np.min(img[:,:,3]))  # 출력: 0 (알파 최솟값)

cv.imshow("unchanged", img[:,:,3]) #graylevel image
cv.waitKey()
cv.imshow("color", imgc)
cv.waitKey()
cv.destroyAllWindows()
#%%
# [서술형] 이진 모폴로지 분석
# 관련 PPT: 영상처리_2 2~10쪽
# 특징: 구조 요소의 모양을 이용해 흰 전경을 침식·팽창하거나 두 연산을 조합한다.
# 장점: 작은 잡음·연결·틈을 공간적 모양 기준으로 정리할 수 있다.
# 단점: 구조 요소와 반복 횟수에 따라 가는 형상이 사라지거나 다른 물체가 붙는다.
# 활용: 마스크 정리, 확실한 전경/배경 마커 만들기, 작은 구조의 크기 분석.
# 한 문장 답안: 모폴로지는 구조 요소로 영역의 모양을 처리하며 유용한 세부도 함께 변할 수 있다.
# [계산·코드 연결]
# 팽창: 전경 확대·작은 간격 연결 / 침식: 전경 축소·작은 돌기와 연결 제거.
# 이진 전경 기준 팽창은 이웃 최댓값, 침식은 이웃 최솟값으로 이해한다.
# 현재는 BGRA의 알파를 이진화한다. 검은 BGR 자체를 이진화한 결과와 구분한다.
# [함수 사용] t,dst=cv.threshold(src, thresh, maxval, type)
# thresh=기준값, maxval=이진화 출력값, type=방법. t는 사용한 임계값, dst는 결과이며 OTSU는 기준을 자동 계산한다.
# [PPT 인자 보충] 영상처리_1 29쪽: 비교는 src>thresh이다(같으면 이하 쪽). THRESH_OTSU는 자동 T를 반환하며 초기 thresh는 무시한다.
t, binimg=cv.threshold(img[:,:,3], 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
cv.imshow("Binary", binimg)
cv.waitKey()
cv.destroyAllWindows()
#%%
# [함수 사용] plt.imshow(X,cmap=None,vmin=None,vmax=None)
# 2D 명암에는 cmap="gray", 컬러는 RGB를 사용한다. vmin/vmax는 명암 표시 범위이며 OpenCV BGR은 변환 후 표시한다.
# [함수 사용] plt.xticks(ticks,labels=None) / plt.yticks(ticks,labels=None)
# 가로/세로 눈금 위치와 표시 문자열을 설정한다. []는 눈금을 숨길 뿐 영상 픽셀을 삭제하지 않는다. [영상처리_2 8쪽]
plt.imshow(binimg, cmap='gray') ;  plt.xticks([]) ; plt.yticks([])
# [함수 사용] plt.show()
# 준비된 그래프 창을 표시한다. 원본 배열을 저장하거나 픽셀 값을 바꾸는 함수가 아니다.
plt.show()
# [배열 크기 예측] 높이 525에서 525//2=262이다.
# [262:525,0:263]이므로 b.shape=(263,263). 원래 너비 1920 전체를 자른 것은 아니다.
# 침식/팽창/닫힘은 같은 크기 배열을 반환하므로 아래 결과도 모두 (263,263)이다.
b=binimg[binimg.shape[0]//2: binimg.shape[0],0: binimg.shape[0]//2+1]
plt.imshow(b, cmap='gray') ; plt.xticks([]) ; plt.yticks([])
plt.show()
#%%
# 구조 요소와 모폴로지 [영상처리_2 PPT 2~10쪽]
# 흰색(전경)을 기준으로 팽창은 넓히고 침식은 줄인다. 여기서는 5x5 마름모를 사용한다.
# 아래 closing=팽창 후 침식은 작은 틈을 잇는 연산이다. opening=침식 후 팽창과 순서가 다르다.
# [함수 사용] dst=np.uint8(values)
# 값을 8비트 부호 없는 정수로 변환한다. 안전한 범위 표시에는 먼저 clip을 고려하며 좌표값을 8비트로 제한하지 않도록 주의한다.
se=np.uint8([[0,0,1,0,0],[0,1,1,1,0],
             [1,1,1,1,1], [0,1,1,1,0],[0,0,1,0,0]])
print(se.shape)  # 출력: (5, 5), 구조 요소의 높이/너비
# [함수 사용] dst=cv.dilate(src, kernel, iterations=1)
# src=입력, kernel=구조 요소, iterations=팽창 반복 횟수. 같은 크기의 결과를 반환한다.
# [PPT 인자 보충] 영상처리_2 7쪽: anchor=(-1,-1)은 커널 중심. borderType/borderValue는 영상 바깥 처리이며 iterations는 같은 연산의 반복 수이다.
b_dilation = cv.dilate(b, se, iterations=1)
plt.imshow(b_dilation, cmap='gray'), plt.xticks([]) ; plt.yticks([])
plt.show()

# [함수 사용] dst=cv.erode(src, kernel, iterations=1)
# src=입력, kernel=구조 요소, iterations=침식 반복 횟수. 같은 크기의 결과를 반환한다.
# [PPT 인자 보충] 영상처리_2 7쪽: anchor=(-1,-1)은 커널 중심. borderType/borderValue는 영상 바깥 처리이며 iterations는 같은 연산의 반복 수이다.
b_eroision = cv.erode(b, se, iterations=1)
plt.imshow(b_eroision, cmap='gray'), plt.xticks([]) ; plt.yticks([])
plt.show()

# [서술형] 열림과 닫힘 비교
# 관련 PPT: 영상처리_2 4~10쪽
# 특징: 열림은 침식→팽창, 닫힘은 팽창→침식이다. 적용 순서가 다르다.
# 장점: 열림은 작은 전경 잡음 제거, 닫힘은 작은 틈·구멍과 끊긴 부분의 완화에 유리하다.
# 단점: 열림은 가는 전경을 없앨 수 있고 닫힘은 가까운 물체를 붙일 수 있다.
# 활용: 이진 마스크를 후처리할 때 잡음을 없앨지 작은 틈을 메울지에 따라 선택한다.
# 한 문장 답안: 열림은 작은 전경을 제거하고 닫힘은 작은 빈틈을 줄이며, 둘의 순서는 서로 반대이다.
# [계산·코드 연결]
# 아래는 닫힘만 실행한다. 임의 크기의 내부 구멍을 모두 채우는 별도 알고리즘은 아니다.
# iterations가 커지면 처리 범위도 커진다. b와 각 결과의 크기는 (263,263)으로 같다.
b_closing = cv.erode(cv.dilate(b, se, iterations=1), se, iterations=1)
plt.imshow(b_closing, cmap='gray'), plt.xticks([]) ; plt.yticks([])
plt.show()



