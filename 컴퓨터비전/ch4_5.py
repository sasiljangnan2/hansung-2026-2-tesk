# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 13:58:38 2023

@author: BigData
"""

# ==================================================================
# [선행 이론] 슈퍼 화소·군집화·SLIC의 배경
# 근거: 4장 영역 분할_2 1~3,15~19쪽
# 읽는 순서: 선행 이론 → 관련 코드 앞의 서술형 요약 → 계산·출력 주석.
# PPT의 배경 이론과 현재 코드의 구현 범위를 구분하여 공부한다.
#
# 1. 영역 분할과 의미 분할
# 영역 분할은 밝기·색·텍스처가 비슷한 픽셀을 묶거나 다른 영역의 경계를 정하는 작업이다.
# 의미 분할은 픽셀에 사람·도로 같은 의미 범주까지 연결한다. 단순 색 군집의 번호가 물체 이름은 아니다.
# 이 강의의 고전적 방법은 주로 픽셀 특성으로 분할한 뒤 해석/인식하는 접근이다.
# 단순 배경에서는 임계값이 유리할 수 있고 복잡한 영상에서는 이웃·그래프·사용자 초기 정보 등을 고려한다.
#
# 2. 왜 픽셀을 작은 영역으로 먼저 묶는가?
# 슈퍼 화소(superpixel)는 화소보다 크고 물체보다 작은 비슷한 특성의 자잘한 영역이다.
# 수많은 화소를 작은 영역들로 줄이면 후속 영역 특징·그래프 노드 계산을 간단히 만들 수 있다.
# 슈퍼 화소 하나가 완전한 물체라는 뜻은 아니다. 하나의 물체가 여러 슈퍼 화소로 나뉘는 것이 자연스럽다.
#
# 3. SLIC의 반복 과정
# 색과 위치를 함께 비교해 픽셀을 근처 군집에 할당하고, 군집 중심을 갱신하는 과정을 반복한다.
# 색만 묶으면 떨어진 곳이 같은 군집이 될 수 있어 공간적 근접성도 함께 사용한다.
# PPT는 (R,G,B,x,y)의 5차원 발상으로 설명하며 실제 skimage 기본 RGB 입력은 Lab 변환을 사용한다.
# n_segments는 목표 개수, compactness는 색 거리와 위치 거리의 균형을 조절한다.
# compactness를 키우면 보통 더 규칙적인 모양이 되지만 실제 색 경계를 덜 따를 수도 있다.
# 반복 수렴·연결성 처리 때문에 최종 영역 개수가 목표 숫자와 정확히 같다고 보장되지 않는다.
#
# 4. 레이블 지도와 경계 표시
# SLIC 출력은 픽셀별 영역 번호인 2차원 정수 지도이다. 3채널 RGB 결과가 아니다.
# mark_boundaries는 번호가 달라지는 경계를 원래 색 영상에 덧그린 표시용 RGB 영상이다.
# 그 번호가 크다고 밝은 물체이거나, 같은 번호가 매 실행 같은 물체를 뜻하는 것은 아니다.
# 현재 scikit-image가 없어 구체적 레이블 출력은 미확인이고 표준 샘플 크기/자료형만 대조했다.
# ==================================================================


import skimage
import numpy as np
import cv2 as cv
#%%
# [표준 샘플의 출력: 영역 분할_2 PPT 15~19쪽]
# coffee는 RGB uint8이고 shape=(400,600,3)이다. OpenCV의 BGR과 채널 순서가 다르다.
# 아래 값 중 샘플 크기는 scikit-image 공식 data 문서로 대조했다.
# 현재 가상환경에는 scikit-image가 없어 이 파일의 실제 SLIC 레이블 수는 실행 확인하지 못했다.
img = skimage.data.coffee()
print(img.shape)  # 표준 coffee 입력 크기: (400, 600, 3), 전체 실행은 미확인
cv.imshow("Coffee", cv.cvtColor(img, cv.COLOR_RGB2BGR))
cv.waitKey()

# [서술형] SLIC 슈퍼 화소 분할
# 관련 PPT: 영역 분할_2 15~19쪽
# 특징: 색과 공간 위치를 함께 사용해 인접한 픽셀을 작은 영역으로 군집화한다.
# 장점: 화소 단위보다 작은 영역 단위로 후속 그래프·특징 계산을 줄이고 경계에 맞춘 분할을 준비한다.
# 단점: 슈퍼 화소는 물체 자체가 아니고 개수·compactness·색 분포에 따라 실제 경계를 넘을 수 있다.
# 활용: 영역 인접 그래프 RAG의 노드 준비, 정규화 절단의 전처리.
# 한 문장 답안: SLIC는 색·위치로 영상을 작은 영역으로 줄이지만 물체 의미와 완전한 경계 일치를 보장하지 않는다.
# [계산·코드 연결]
# n_segments=600은 목표 개수. compactness가 커지면 공간 비중이 커져 더 규칙적인 경향이다.
# PPT는 RGBxy로 발상을 설명하고, skimage의 RGB 기본 입력은 Lab 변환을 사용한다.
# slic1은 (400,600) 정수 레이블. min/max는 밝기가 아니라 영역 번호이고 max=600을 보장하지 않는다.
# start_label 기본값 1인 설정에서는 1부터 시작. 현재 scikit-image 미설치로 실제 분할 번호는 미확인.
slic1=skimage.segmentation.slic(img, compactness=20, n_segments=600) #result of clustering
print(np.min(slic1)); print(np.max(slic1))

# [표시·출력] 레이블 slic1은 2차원이고 mark_boundaries는 색 경계가 덧그려진 RGB 실수 영상이다.
# sp_img1.shape=(400,600,3), 값은 0~1. 이 print는 *255와 uint8 변환 전이다.
# 경계 기본색 (1,1,0)은 노랑이라 max=1.0이다. *255 후 RGB2BGR로 바꾸어 OpenCV에 표시한다.
# sp_img2의 min/max 출력도 변환 전이다. 밝기 범위와 영역 번호 범위를 혼동하지 않는다.
sp_img1=skimage.segmentation.mark_boundaries(img, slic1) #mode='outer'
print(np.min(sp_img1)); print(np.max(sp_img1)); print(sp_img1.shape)
sp_img1=np.uint8(sp_img1*255)

slic2=skimage.segmentation.slic(img, compactness=40, n_segments=600)

sp_img2=skimage.segmentation.mark_boundaries(img, slic2)
print(np.min(sp_img2)); print(np.max(sp_img2))
sp_img2=np.uint8(sp_img2*255)


cv.imshow("Super Pixels, Com=20", cv.cvtColor(sp_img1, cv.COLOR_RGB2BGR))
cv.imshow("Super Pixels, Com=40", cv.cvtColor(sp_img2, cv.COLOR_RGB2BGR))
cv.waitKey()

cv.destroyAllWindows()
