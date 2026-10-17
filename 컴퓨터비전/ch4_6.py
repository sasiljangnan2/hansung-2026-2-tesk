# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 15:51:43 2023

@author: BigData
"""

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
print(slic.shape)

# 영역 인접 그래프와 정규화 절단 [영역 분할_2 PPT 20~27쪽]
# SLIC 영역을 노드로, 인접 영역의 평균 색 유사도를 간선 가중치로 만든다.
# cut_normalized는 절단 비용을 연결 규모로 정규화하여 작은 영역만 쉽게 잘리는 편향을 줄인다.
g=skimage.graph.rag_mean_color(img, slic, mode='similarity')
ncut=skimage.graph.cut_normalized(slic, g)
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
