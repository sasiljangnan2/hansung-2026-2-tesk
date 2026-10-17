# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 12:16:58 2023

@author: BigData
"""

import cv2 as cv
import matplotlib.pyplot as plt
import sys
#%%

img = cv.imread("mistyroad.jpg")

if img is None:
    sys.exit("No File exists.")
    
cv.imshow('Original', img)
cv.waitKey()

print(img.shape)
#%%
gray=cv.cvtColor(img, cv.COLOR_BGR2GRAY)
print(gray.shape)

plt.imshow(gray, cmap='gray'), plt.xticks([]), plt.yticks([])
#%%
# 전역 텍스처의 기초 [영역 특징 PPT 15~17쪽]
# 밝기별 빈도수는 전체 영상의 분포이다. 정규화하면 평균/분산/균일성/엔트로피의 입력이 된다.
# 공간 배치는 잃기 때문에 같은 히스토그램으로 다른 무늬가 가능하다. GLCM/LBP는 이웃을 본다.
# 기존 범위 [0,255]는 상한 255를 제외한다. 0~255 전체 범위는 [0,256]이다.
h=cv.calcHist([gray], [0], None, [256], [0, 255])
plt.plot(h, color='r', linewidth=1) 
plt.show()
#%%
# 아래는 대비를 조절하는 히스토그램 평활화이다. 텍스처 기술자 자체를 계산하는 코드는 아니다.
# 누적 분포로 대비 조절 [영상처리_2 PPT 15~20쪽]
# equalizeHist는 단일 채널 uint8 히스토그램의 누적 분포를 이용해 밝기를 재배치한다.
# 모든 밝기 빈도가 정확히 같아지는 것은 아니다. 컬러는 밝기 성분만 처리하면 색 변형을 줄일 수 있다.
equal=cv.equalizeHist(gray)
plt.imshow(equal, cmap='gray'), plt.xticks([]), plt.yticks([])
cv.imshow('Equal', equal)
cv.waitKey()
cv.destroyAllWindows()
#%%
h=cv.calcHist([equal], [0], None, [256], [0, 255])
plt.plot(h, color='r', linewidth=1) 
plt.show()
