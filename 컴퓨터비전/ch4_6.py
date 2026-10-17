# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 15:51:43 2023

@author: BigData
"""

# ==================================================================
# [선행 이론] 유사도 그래프·절단·정규화의 배경
# 근거: 4장 영역 분할_2 20~27쪽
# 읽는 순서: 선행 이론 → 관련 코드 앞의 서술형 요약 → 계산·출력 주석.
# PPT의 배경 이론과 현재 코드의 구현 범위를 구분하여 공부한다.
#
# 1. 영역 분할과 의미 분할
# 영역 분할은 밝기·색·텍스처가 비슷한 픽셀을 묶거나 다른 영역의 경계를 정하는 작업이다.
# 의미 분할은 픽셀에 사람·도로 같은 의미 범주까지 연결한다. 단순 색 군집의 번호가 물체 이름은 아니다.
# 이 강의의 고전적 방법은 주로 픽셀 특성으로 분할한 뒤 해석/인식하는 접근이다.
# 단순 배경에서는 임계값이 유리할 수 있고 복잡한 영상에서는 이웃·그래프·사용자 초기 정보 등을 고려한다.
#
# 2. 영상을 그래프로 나타내기
# G=(V,E)에서 V는 화소 또는 슈퍼 화소 노드, E는 이웃 노드 사이 간선이다.
# 간선 가중치는 서로 비슷한 정도를 나타낸다. 유사도가 크면 자르기 어렵게 하여 같은 집단에 남긴다.
# 평균 색의 distance와 similarity는 크기의 의미가 반대일 수 있다. 함수의 mode를 확인한다.
# 현재는 SLIC를 노드로 한 영역 인접 그래프(RAG)를 평균 색 similarity로 만든다.
#
# 3. 단순 cut의 문제
# cut(A,B)=sum(w_pq,p in A,q in B)는 두 영역 사이에서 끊는 가중치의 합이다.
# 작은 덩어리를 떼면 간선 수가 적어 값도 쉽게 작아지므로 직관적 물체 분할보다 유리해지는 편향이 있다.
# PPT는 더 타당해 보이는 분할 cut=12보다 작은 덩어리를 떼는 cut=7이 선택될 수 있음을 보여 준다.
# 절단값이 작다는 기준과 원하는 의미의 물체라는 기준은 같은 것이 아니다.
#
# 4. normalized cut이 보정하는 것
# Ncut=cut(A,B)/assoc(A,V)+cut(A,B)/assoc(B,V). assoc는 각 집단과 전체 그래프의 연결 가중치 합이다.
# 영역의 연결 규모를 함께 고려해 작은 부분만 떼는 편향을 줄인다. 단순 픽셀 수로만 나누는 식은 아니다.
# 정확한 조합 최적화는 어려워 실제 함수는 스펙트럴 완화를 이용한 근사 분할을 수행한다.
# 정규화했다고 항상 의미상 옳은 물체가 나오는 것은 아니다. 초기 SLIC와 유사도 정의의 영향을 받는다.
# 최대 유량/최소 S-T 절단과는 목적 함수·문제 설정이 다르다. 두 방법을 같은 알고리즘으로 외우지 않는다.
# 현재 ncut==120은 결과 번호 하나를 선택하는 예이며 120이 고정된 물체 이름인 것은 아니다.
# ==================================================================


import skimage
import numpy as np
import cv2 as cv
import time
#%%
img = skimage.data.coffee()
cv.imshow("Coffee", cv.cvtColor(img, cv.COLOR_RGB2BGR))
cv.waitKey()

start=time.time()
slic=skimage.segmentation.slic(img, compactness=20, n_segments=600, start_label=1)
# [출력 형식] coffee 입력=(400,600,3), SLIC 레이블 배열=(400,600)이다.
# 현재 scikit-image 미설치로 전체 미실행. 크기는 공식 샘플 기준이며 구체적인 분할 번호는 단정하지 않는다.
print(slic.shape)

# [서술형] 그래프 기반 정규화 절단(Ncut) 분석
# 관련 PPT: 영역 분할_2 20~27쪽
# 특징: 픽셀/슈퍼 화소의 유사도 그래프를 만들고 영역 사이 절단 비용을 연결 규모로 정규화한다.
# 장점: 국소 밝기만 보지 않고 그래프의 연결 관계를 고려하며 작은 덩어리만 떼는 편향을 줄인다.
# 단점: 그래프·고유벡터 계산 비용이 들고 유사도 설정과 초기 영역의 영향을 받는다. 의미 분할은 아니다.
# 활용: SLIC 영역을 더 큰 색 영역으로 묶는 분할, 국소 분할의 한계 보완.
# 한 문장 답안: 정규화 절단은 연결 규모를 고려한 그래프 분할로 작은 영역 편향을 줄이지만 계산과 유사도 설정이 중요하다.
# [계산·코드 연결]
# cut(A,B)=영역 사이 가중치 합, Ncut=cut/assoc(A,V)+cut/assoc(B,V). assoc는 전체 연결 합이다.
# 여기서는 SLIC 영역의 평균 색으로 RAG를 만들고 similarity 모드에서 비슷한 색에 큰 가중치를 준다.
# 정확한 조합 최적화 대신 스펙트럴 근사를 사용한다. 최대 유량/최소 S-T 절단과 목적 함수가 다르다.
g=skimage.graph.rag_mean_color(img, slic, mode='similarity')
ncut=skimage.graph.cut_normalized(slic, g)
# [출력 해석] 첫 값은 (400,600,3), '분할 소요 시간' 뒤 실수는 실행마다 달라지는 초 단위 시간이다.
# ncut.shape=(400,600), ncut 출력은 픽셀별 영역 번호 배열이다. 번호가 크다고 더 밝은 물체는 아니다.
# np.unique(ncut)는 존재하는 번호를 오름차순으로 한 번씩만 출력한다. 영역 수는 len(np.unique(ncut)).
# 번호에 빈칸이 있을 수 있어 max(ncut)+1을 영역 수라고 단정하지 않는다. 레이블 120의 존재도 보장되지 않는다.
print(img.shape, '분할 소요 시간', time.time()-start)
print(np.min(ncut)); print(np.max(ncut))
print(ncut);print(ncut.shape);
print(np.unique(ncut))

marking=skimage.segmentation.mark_boundaries(img, ncut)
ncut_img=np.uint8(marking*255)

cv.imshow("Normalized Cut", cv.cvtColor(ncut_img, cv.COLOR_RGB2BGR))
cv.waitKey()

cv.destroyAllWindows()

#%%

# 선택 레이블만 추출 [영역 분할_2 PPT 24~27쪽 응용]
# 레이블 120은 현재 결과의 영역 번호일 뿐 고정된 물체 이름이 아니다. 실행마다 존재·의미를 확인한다.
# 검은 영상에서 mask가 참인 픽셀에 원본 색을 복사한다.
dst=np.zeros_like(img)
mask=(ncut==120)
dst[mask]=img[mask]
cv.imshow("Cut", cv.cvtColor(dst, cv.COLOR_RGB2BGR))
cv.waitKey()

cv.destroyAllWindows()
