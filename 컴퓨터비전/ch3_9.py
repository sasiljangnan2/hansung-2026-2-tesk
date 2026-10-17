# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 16:38:26 2023

@author: BigData
"""

import cv2 as cv
import numpy as np
import sys
import time

#%%

# 반복문과 배열 연산의 성능 비교 [영상처리_2 PPT 43~45쪽]
# 같은 BGR 가중합 0.114B+0.587G+0.299R을 픽셀별 반복문과 배열 연산으로 비교한다.
# NumPy/OpenCV는 내부의 최적화된 연산을 사용한다. uint8 절삭과 OpenCV의 반올림 차이는 가능하다.
def my_cvtGray1(bar_img):
    g = np.zeros([bar_img.shape[0], bar_img.shape[1]])
    for r in range(bar_img.shape[0]):
        for c in range(bar_img.shape[1]):
            g[r,c]= 0.114*bar_img[r,c,0]+0.587*bar_img[r,c,1]+0.299*bar_img[r,c,2]
    return np.uint8(g)

def my_cvtGray2(bar_img):
    g = np.zeros([bar_img.shape[0], bar_img.shape[1]])
    g = 0.114*bar_img[:,:,0]+0.587*bar_img[:,:,1]+0.299*bar_img[:,:,2]
    return np.uint8(g)
        
#%%
img = cv.imread("girl_laughing.jpg")
if img is None:
    sys.exit("No File exists.")

# 처리 시간 측정 [영상처리_2 PPT 44~45쪽]
# 영상 읽기를 제외하고 각 변환의 시간을 잰다. 한 번의 측정값은 실행 환경의 영향을 받으므로
# 모든 크기의 영상에서 같은 배수로 빠르다는 뜻은 아니다.
start=time.time()
my_cvtGray1(img)
print('My time1: ', time.time()-start)

start=time.time()
my_cvtGray2(img)
print('My time2: ', time.time()-start)

start=time.time()
cv.cvtColor(img, cv.COLOR_BGR2GRAY)
print('MOpenCV time1: ', time.time()-start)
