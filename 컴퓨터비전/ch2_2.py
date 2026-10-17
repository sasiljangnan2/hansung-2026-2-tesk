# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 15:54:21 2023

@author: BigData
"""

import cv2 as cv
import sys

#%%
# 영상의 배열 표현 [영상처리_1 PPT 5~10쪽]
# 배열은 [행, 열, 채널]=[y,x,BGR] 순서이다. shape는 높이·너비·채널 수를 뜻한다.
# 그리기 함수에 넣는 좌표 (x,y)와 배열 인덱스 [y,x]를 구분한다.
img=cv.imread('soccer.jpg')

if img is None:
    sys.exit("No File exists.")

print(type(img))
print(img.shape)
type(img)
#pixel elements
print(img[0,0,0], img[0,0,1], img[0,0,2])   
print(img[0,1,0], img[0,1,1], img[0,1,2])   
print(img[0:10,0:10,0], img[0:10,0:10,1], img[0:10,0:10,2])   

cv.imshow('Img Display', img)
cv.imshow("Partial", img[0:10,0:10,:])

cv.waitKey()
cv.destroyAllWindows()
