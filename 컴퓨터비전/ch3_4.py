# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 14:04:24 2023

@author: BigData
"""

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import sys

#%%
# 알파 채널을 이용한 이진 영상 [영상처리_1 PPT 15쪽 / 영상처리_2 PPT 8~10쪽]
# IMREAD_UNCHANGED로 투명도까지 보존해야 네 번째 채널 [:,:,3]을 읽을 수 있다.
# 일반 컬러 읽기는 BGR 3채널이므로 같은 인덱스를 사용할 수 없다.
img = cv.imread("JohnHancocksSignature.png", cv.IMREAD_UNCHANGED)

imgc = cv.imread("JohnHancocksSignature.png")

if img is None:
    sys.exit("No File exists.")
    
print(type(img))

print(img.shape)
print(imgc.shape)
print(img)
print(imgc)
print(np.max(imgc))
print(np.max(img[:,:,3]))
print(np.min(img[:,:,3]))

cv.imshow("unchanged", img[:,:,3]) #graylevel image
cv.waitKey()
cv.imshow("color", imgc)
cv.waitKey()
cv.destroyAllWindows()
#%%
t, binimg=cv.threshold(img[:,:,3], 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
cv.imshow("Binary", binimg)
cv.waitKey()
cv.destroyAllWindows()
#%%
plt.imshow(binimg, cmap='gray') ;  plt.xticks([]) ; plt.yticks([])
plt.show()
b=binimg[binimg.shape[0]//2: binimg.shape[0],0: binimg.shape[0]//2+1]
plt.imshow(b, cmap='gray') ; plt.xticks([]) ; plt.yticks([])
plt.show()
#%%
# 구조 요소와 모폴로지 [영상처리_2 PPT 2~10쪽]
# 흰색(전경)을 기준으로 팽창은 넓히고 침식은 줄인다. 여기서는 5x5 마름모를 사용한다.
# 아래 closing=팽창 후 침식은 작은 틈을 잇는 연산이다. opening=침식 후 팽창과 순서가 다르다.
se=np.uint8([[0,0,1,0,0],[0,1,1,1,0],
             [1,1,1,1,1], [0,1,1,1,0],[0,0,1,0,0]])
print(se.shape)
b_dilation = cv.dilate(b, se, iterations=1)
plt.imshow(b_dilation, cmap='gray'), plt.xticks([]) ; plt.yticks([])
plt.show()

b_eroision = cv.erode(b, se, iterations=1)
plt.imshow(b_eroision, cmap='gray'), plt.xticks([]) ; plt.yticks([])
plt.show()

b_closing = cv.erode(cv.dilate(b, se, iterations=1), se, iterations=1)
plt.imshow(b_closing, cmap='gray'), plt.xticks([]) ; plt.yticks([])
plt.show()



