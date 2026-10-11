# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 10:49:25 2023

@author: BigData
"""

import cv2 as cv
import numpy as np

#%% 영상 읽기 및 흑백 변환
img=cv.imread("group.jpg")
gray=cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# Otsu 반전 이진화로 물체와 배경 구분
t, binary=cv.threshold(gray, 0,255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
print(t)

cv.imshow("Color", img); cv.imshow("Gray", gray); cv.imshow("Binary", binary)
cv.waitKey(); cv.destroyAllWindows()
#%% 침식으로 확실한 전경 만들기
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
markers=fg+bg
cv.imshow("Markers", markers)
cv.waitKey()

#%% 워터셰드 실행 (경계는 -1)
markers=np.int32(markers)
markers=cv.watershed(img, markers)
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
markers=np.uint8(np.clip(markers, 0, 255))
cv.imshow("Segmentation", markers)
cv.waitKey() 

cv.destroyAllWindows()
