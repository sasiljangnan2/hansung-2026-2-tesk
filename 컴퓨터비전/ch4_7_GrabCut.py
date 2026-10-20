# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 17:03:54 2023

@author: BigData
"""

#%% 개념 요약 — 정의·장점·단점, 함수 사용법은 해당 코드 옆 참고

#%% [개념] GrabCut 대화식 물체 분할
# 정의: 사용자가 지정한 사각형이나 전경·배경 정보를 이용하고, 색 모델과 그래프 절단을 반복하여 물체를
#   분리하는 방법이다.
# 장점: 사용자가 선택한 관심 물체를 색 분포와 이웃 관계를 함께 이용해 분리할 수 있다.
# 단점: 초기 지정과 전경·배경 색 겹침에 영향을 받으며 가는 부분이나 유사한 배경에서 오류가 남는다.
# PPT: 영역 분할_2 36~46쪽

#region 핵심 이론·함수 참고
# [핵심 이론] 그래프 절단·사용자 정보·GrabCut의 배경
# 근거: 4장 영역 분할_2 28,36~46쪽
# 1. 영역 분할과 의미 분할
# 영역 분할은 밝기·색·텍스처가 비슷한 픽셀을 묶거나 다른 영역의 경계를 정하는 작업이다.
# 의미 분할은 픽셀에 사람·도로 같은 의미 범주까지 연결한다. 단순 색 군집의 번호가 물체 이름은 아니다.
# 2. 최대 유량과 최소 절단
# 노드 사이 간선에 용량을 둔 네트워크에서 S에서 T로 보낼 수 있는 최대 흐름을 구한다.
# 병목 때문에 각 간선이 동시에 최대 용량만큼 흐르는 것은 아니다.
# 3. 전경 S·배경 T와 간선 비용
# 비슷한 이웃 사이 가중치를 크게 주면 두 픽셀을 서로 다른 쪽으로 자르기 어렵다.
# 미확정 픽셀은 색 모델과 이웃 관계를 함께 고려한다. 큰 숫자=밝은 픽셀이라는 뜻은 아니다.
# 4. GrabCut은 초기 정보와 색 모델을 반복 사용한다
# 전경/배경 색 모델을 추정하고 그래프 절단을 한 뒤 모델과 분할을 갱신하는 발상이다.
# 사각형 초기화는 밖을 확실한 배경으로, 안은 후보로 시작한다. 안의 모든 픽셀을 확정 전경으로 두는 것은 아니다.


#endregion

import numpy as np
import cv2 as cv
#%%
# [함수 사용: 입출력·창]
# img=cv.imread(path, flags=cv.IMREAD_COLOR): 파일→배열. 기본 BGR, 읽기 실패 시 None인지 확인한다.
# cv.imshow(winname,img): 이름이 winname인 창에 배열을 표시한다. 저장 함수는 아니다.
# cv.namedWindow(winname): 마우스 콜백 등을 등록할 이름 있는 창을 만든다.
# key=cv.waitKey(delay=0): GUI 이벤트·키 처리. 0은 계속 대기, 양수는 ms, 입력 없으면 보통 -1이다.
# cv.destroyAllWindows(): OpenCV 창을 닫는다. 캡처 자원 해제는 cap.release()이다.
img=cv.imread('soccer.jpg')
# [함수 사용] dst=np.copy(src)
# src의 배열 데이터를 복사해 독립적인 수정용 배열을 만든다. 영상에서는 src.copy()도 사용한다.
img_show=np.copy(img) # 깊은 복사

# 이 코드는 마스크 초기화 방식. 전체를 추정 배경 2로 놓고 왼쪽=전경 1, 오른쪽=배경 0을 칠한다.
# S/T 간 최소 절단 용량은 최대 유량과 같다. 이웃 유사도와 전경/배경 배정 비용을 함께 고려한다.
# 확실한 지정은 큰 용량으로 유지하며 미확정 픽셀을 계산한다. Ncut과 같은 목적 함수는 아니다.
# [함수 사용] a=np.zeros(shape, dtype=float)
# shape=배열 크기, dtype=자료형. 0으로 채운 배열을 만든다. 영상 마스크에는 보통 dtype=np.uint8을 지정한다.
mask=np.zeros((img.shape[0], img.shape[1]), np.uint8)
# [값/크기 문제] 현재 mask.shape=(948,1434), dtype=uint8이고 이 줄 이후 모든 값은 2이다.
# 0=확실한 배경,1=확실한 전경,2=추정 배경,3=추정 전경. 일반 0/255 이진 마스크와 다르다.
# 원본 img와 깊은 복사 img_show를 분리했기 때문에 붓 색이 원래 분할 입력 픽셀에 섞이지 않는다.
mask[:,:]=cv.GC_PR_BGD

BrushSiz=9
LColor, RColor = (255, 0, 0), (0, 0, 255)

# 사용자 입력 단계 [영역 분할_2 PPT 43~44쪽]
# 왼쪽 버튼은 전경, 오른쪽 버튼은 배경이다. 화면의 색 표시와 실제 mask의 레이블은 별개이다.
# 양쪽 정보를 충분히 칠한 다음 q를 눌러 분할 계산으로 넘어간다.
def painting(event, x, y, flags, param):
    if event==cv.EVENT_LBUTTONDOWN:
        # [함수 사용] cv.circle(img, center, radius, color, thickness=1)
        # center=(x,y), radius=반지름, color=BGR 색. img에 직접 그리며 thickness=-1이면 원 내부를 채운다.
        # [PPT 인자 보충] OpenCV 기본 37쪽: lineType=LINE_4/LINE_8/LINE_AA는 선 연결·안티앨리어싱 방식. shift는 좌표의 소수 비트 수이며 thickness<0이면 채운다.
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
# [함수 사용] cv.setMouseCallback(winname, onMouse, param=None)
# winname=창 이름, onMouse=콜백 함수 자체. 이벤트가 나면 onMouse(event,x,y,flags,param)를 호출하도록 등록한다.
cv.setMouseCallback("Painting", painting)


while(True):
    if cv.waitKey(1)==ord('q'):
        break
    
background=np.zeros((1,65), np.float64)
foreground=np.zeros((1,65), np.float64)   

# 색 모델과 그래프 절단 반복 [영역 분할_2 PPT 36~46쪽]
# GC_INIT_WITH_MASK는 그려 둔 mask로 초기화한다. 두 (1,65) float64 배열은 전경·배경 모델 저장용이다.
# 0/2를 배경으로, 1/3을 전경으로 합친 뒤 채널 축을 추가하여 BGR 세 채널에 같은 마스크를 적용한다.
# [함수 사용] cv.grabCut(img, mask, rect, bgdModel, fgdModel, iterCount, mode)
# mask=0/1/2/3 레이블, rect=(x,y,w,h), 두 모델=(1,65) float64, iterCount=반복 수, mode=초기화 방식. mask·모델을 갱신하며 세 값도 반환한다.
# [PPT 인자 보충] 영역 분할 46쪽: mode=GC_INIT_WITH_RECT는 사각형 초기화, GC_INIT_WITH_MASK는 마스크 초기화, GC_EVAL은 이어서 반복. mask와 두 모델 배열을 갱신한다.
cv.grabCut(img, mask, None, background, foreground, 5, cv.GC_INIT_WITH_MASK)
# [조건/브로드캐스팅 문제] np.where(조건,참값,거짓값)이므로 0/2->0, 1/3->1이다.
# 예: mask 행 [0,1,2,3]은 mask2 행 [0,1,0,1]로 바뀐다.
# mask2.shape=(948,1434), mask2[:,:,np.newaxis].shape=(948,1434,1)이다.
# BGR 세 채널에 동일한 0/1이 곱해져 전경은 보존하고 배경은 [0,0,0]이 된다.
# [함수 사용] dst=np.where(condition, x, y)
# condition이 True인 위치는 x, False인 위치는 y를 선택한 배열을 반환한다.
mask2=np.where((mask==cv.GC_BGD)|(mask==cv.GC_PR_BGD), 0, 1).astype('uint8')
grab=img*mask2[:,:, np.newaxis] # mask brodcasting: 

cv.imshow("Grab Cut Image", grab)
cv.waitKey()

cv.destroyAllWindows()
