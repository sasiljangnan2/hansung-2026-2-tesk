# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 11:17:44 2023

@author: BigData
"""

import cv2 as cv
import numpy as np
import sys

#%%
img = cv.imread("soccer.jpg")

if img is None:
    sys.exit("No File exists.")

img=cv.resize(img, dsize=(0,0), fx=0.25, fy=0.25)

# 감마 점 연산 [영상처리_2 PPT 11~14쪽]
# 0~1로 정규화한 밝기에 거듭제곱을 적용하고 0~255로 복원한다. 이 코드의 정의에서는
# gamma<1이면 중간 밝기가 밝아지고, gamma>1이면 어두워진다. 0과 255는 유지된다.
def gamma(f, gamma=1.0):
    f1=f/255.0
    return np.uint8(255*(f1**gamma))

gc = np.hstack((gamma(img, 0.5),gamma(img, 0.75), gamma(img, 1.0), 
                gamma(img, 2.0), gamma(img, 3.0)))    
cv.imshow("gamma", gc)

cv.waitKey()
cv.destroyAllWindows()