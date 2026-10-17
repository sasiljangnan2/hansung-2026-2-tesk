# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 20:04:22 2023

@author: BigData
"""

import cv2 as cv
import sys
import numpy as np

#%%
cap=cv.VideoCapture('slow_traffic_small.mp4')
if not cap.isOpened():
    sys.exit("연결 실패")
    
frames=[]

while True:
    ret, frame=cap.read()
    
    if not ret:
        print("프레임 획득 실패")
        break
    
    cv.imshow('Video Display', frame)
    
    key=cv.waitKey(1)
    if key==ord('c'):
        frames.append(frame)
    elif key==ord('q'):
        break

cap.release()
cv.destroyAllWindows()

if len(frames)> 0:
    imgs=frames[0]
    for i in range(1, min(3, len(frames))):
        imgs=np.hstack((imgs, frames[i]))
        
    cv.imshow('collected images', imgs)
    
    cv.waitKey()
    cv.destroyAllWindows()
        