# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 20:54:33 2023

@author: BigData
"""

import cv2 as cv
import sys
#%%
img=cv.imread('soccer.jpg')\

if img is None:
    sys.exit("No File exists.")
#%%    
#t, bin = cv.threshold(img[:,:,2], 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
# Otsu 전역 이진화 [영상처리_1 PPT 24~30쪽]
# R 채널에서 두 집단의 가중 클래스 내 분산이 최소가 되는 임계값 t를 자동 선택한다.
# THRESH_BINARY는 입력값>t인 픽셀을 255로 만든다. OTSU 사용 시 전달한 0은 초기 임계값일 뿐이다.
t, bin = cv.threshold(img[:,:,2], 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

print("threshold", t)
cv.imshow('R', img[:,:,2])
cv.imshow('R binary', bin)

cv.waitKey()
cv.destroyAllWindows()
