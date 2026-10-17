# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 17:03:54 2023

@author: BigData
"""

import numpy as np
import cv2 as cv
#%%
img=cv.imread('soccer.jpg')
img_show=np.copy(img) # 깊은 복사

# GrabCut의 초기 레이블 [영역 분할_2 PPT 41~46쪽]
# 레이블은 확실한 배경=0, 전경=1, 추정 배경=2, 추정 전경=3이다. 일반 0/255 마스크와 다르다.
# 여기서는 전체를 추정 배경으로 시작하고 마우스로 확실한 전경·배경을 지정한다.
mask=np.zeros((img.shape[0], img.shape[1]), np.uint8)
mask[:,:]=cv.GC_PR_BGD

BrushSiz=9
LColor, RColor = (255, 0, 0), (0, 0, 255)

# 사용자 입력 단계 [영역 분할_2 PPT 43~44쪽]
# 왼쪽 버튼은 전경, 오른쪽 버튼은 배경이다. 화면의 색 표시와 실제 mask의 레이블은 별개이다.
# 양쪽 정보를 충분히 칠한 다음 q를 눌러 분할 계산으로 넘어간다.
def painting(event, x, y, flags, param):
    if event==cv.EVENT_LBUTTONDOWN:
        cv.circle(img_show, (x,y), BrushSiz, LColor, -1)
        cv.circle(mask, (x,y), BrushSiz, cv.GC_FGD, -1)
    elif event==cv.EVENT_RBUTTONDOWN:
        cv.circle(img_show, (x,y), BrushSiz, RColor, -1)
        cv.circle(mask, (x,y), BrushSiz, cv.GC_BGD, -1)
    elif event==cv.EVENT_MOUSEMOVE and flags==cv.EVENT_FLAG_LBUTTON:
        cv.circle(img_show, (x,y), BrushSiz, LColor, -1)
        cv.circle(mask, (x,y), BrushSiz, cv.GC_FGD, -1)
    elif event==cv.EVENT_MOUSEMOVE and flags==cv.EVENT_FLAG_RBUTTON:
        cv.circle(img_show, (x,y), BrushSiz, RColor, -1)
        cv.circle(mask, (x,y), BrushSiz, cv.GC_BGD, -1)
        
    cv.imshow("Painting", img_show)

    
cv.namedWindow("Painting")
cv.setMouseCallback("Painting", painting)


while(True):
    if cv.waitKey(1)==ord('q'):
        break
    
background=np.zeros((1,65), np.float64)
foreground=np.zeros((1,65), np.float64)   

# 색 모델과 그래프 절단 반복 [영역 분할_2 PPT 36~46쪽]
# GC_INIT_WITH_MASK는 그려 둔 mask로 초기화한다. 두 (1,65) float64 배열은 전경·배경 모델 저장용이다.
# 0/2를 배경으로, 1/3을 전경으로 합친 뒤 채널 축을 추가하여 BGR 세 채널에 같은 마스크를 적용한다.
cv.grabCut(img, mask, None, background, foreground, 5, cv.GC_INIT_WITH_MASK)
mask2=np.where((mask==cv.GC_BGD)|(mask==cv.GC_PR_BGD), 0, 1).astype('uint8')
grab=img*mask2[:,:, np.newaxis] # mask brodcasting: 

cv.imshow("Grab Cut Image", grab)
cv.waitKey()

cv.destroyAllWindows()
