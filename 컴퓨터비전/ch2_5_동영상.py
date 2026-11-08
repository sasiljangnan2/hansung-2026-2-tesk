# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 20:04:22 2023

@author: BigData
"""

import cv2 as cv
from cv_assets import open_video
import sys
import numpy as np

#%%
# [함수 사용] cap=cv.VideoCapture(source)
# source=카메라 번호 또는 동영상 경로. cap.isOpened()로 열기 확인, ret,frame=cap.read()로 읽고 cap.release()로 해제한다.
# [PPT 인자 보충] OpenCV 기본 23쪽: apiPreference=cv.CAP_ANY는 백엔드 자동 선택. cap.isOpened()는 열기 성공 여부, ret,frame=cap.read()는 다음 프레임, cap.release()는 자원 해제이다.
cap=open_video('slow_traffic_small.mp4')
if not cap.isOpened():
    sys.exit("연결 실패")
    
frames=[]

while True:
    ret, frame=cap.read()
    
    if not ret:
        print("프레임 획득 실패")
        break
    
    # [함수 사용: 입출력·창]
    # cv.imshow(winname,img): 이름이 winname인 창에 배열을 표시한다. 저장 함수는 아니다.
    # key=cv.waitKey(delay=0): GUI 이벤트·키 처리. 0은 계속 대기, 양수는 ms, 입력 없으면 보통 -1이다.
    # cv.destroyAllWindows(): OpenCV 창을 닫는다. 캡처 자원 해제는 cap.release()이다.
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
        # [함수 사용] dst=np.hstack((a,b))
        # 영상은 너비 축으로 연결하므로 높이·채널 등 다른 축이 같아야 한다. 새 연결 배열을 반환한다.
        imgs=np.hstack((imgs, frames[i]))
        
    cv.imshow('collected images', imgs)
    
    cv.waitKey()
    cv.destroyAllWindows()
        