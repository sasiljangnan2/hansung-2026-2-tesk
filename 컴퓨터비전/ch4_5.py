# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 13:58:38 2023

@author: BigData
"""

import skimage
import numpy as np
import cv2 as cv
#%%
img = skimage.data.coffee()
print(img.shape)
cv.imshow("Coffee", cv.cvtColor(img, cv.COLOR_RGB2BGR))
cv.waitKey()

# SLIC 슈퍼 화소 [영역 분할_2 PPT 15~19쪽]
# 색과 위치가 가까운 픽셀을 작은 영역으로 묶는다. n_segments=600은 목표 개수이지 고정 보장은 아니다.
# compactness가 크면 위치의 비중이 커져 더 규칙적인 모양이 된다. RGB 입력의 기본 설정은 Lab 변환을 사용한다.
slic1=skimage.segmentation.slic(img, compactness=20, n_segments=600) #result of clustering
print(np.min(slic1)); print(np.max(slic1))

# 레이블과 결과 표시 [영역 분할_2 PPT 16~19쪽]
# slic1은 영역 번호가 들어 있는 2차원 정수 배열이다. mark_boundaries의 0~1 실수 결과는
# 255를 곱해 uint8로 바꾸고, skimage의 RGB를 OpenCV 표시용 BGR로 변환한다.
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
