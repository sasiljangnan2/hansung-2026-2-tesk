# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 20:24:46 2023

@author: BigData
"""

import cv2 as cv
import sys

#%%
# [함수 사용: 입출력·창]
# img=cv.imread(path, flags=cv.IMREAD_COLOR): 파일→배열. 기본 BGR, 읽기 실패 시 None인지 확인한다.
# cv.imshow(winname,img): 이름이 winname인 창에 배열을 표시한다. 저장 함수는 아니다.
# cv.namedWindow(winname): 마우스 콜백 등을 등록할 이름 있는 창을 만든다.
# key=cv.waitKey(delay=0): GUI 이벤트·키 처리. 0은 계속 대기, 양수는 ms, 입력 없으면 보통 -1이다.
# cv.destroyAllWindows(): OpenCV 창을 닫는다. 캡처 자원 해제는 cap.release()이다.
img=cv.imread('girl_laughing.jpg') 
  
if img is None:
    sys.exit('파일을 찾을 수 없습니다.')

def draw(event, x, y, flags, param):
    if event==cv.EVENT_LBUTTONDOWN:
        # [함수 사용] cv.rectangle(img, pt1, pt2, color, thickness=1)
        # pt1·pt2=(x,y) 꼭짓점, color=BGR 색. img에 그리며 -1은 채우기이다. (x,y,w,h) 사각형 인자 형식도 있다.
        # [PPT 인자 보충] OpenCV 기본 30쪽: lineType은 선 연결·안티앨리어싱, shift는 좌표의 소수 비트 수. thickness<0이면 채우며 img를 직접 수정한다.
        cv.rectangle(img, (x,y), (x+200, y+200), (0,0,255),2)
    elif event==cv.EVENT_RBUTTONDOWN:
        cv.rectangle(img, (x,y), (x+100, y+100), (255,0,0),2)

    cv.imshow('Drawing',img)


cv.namedWindow('Drawing')
cv.imshow('Drawing', img)

# [함수 사용] cv.setMouseCallback(winname, onMouse, param=None)
# winname=창 이름, onMouse=콜백 함수 자체. 이벤트가 나면 onMouse(event,x,y,flags,param)를 호출하도록 등록한다.
cv.setMouseCallback('Drawing', draw)

while(True):
    if cv.waitKey(1)==ord('q'):
        cv.destroyAllWindows()
        break