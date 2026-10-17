# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 17:03:54 2023

@author: BigData
"""

# ==================================================================
# [선행 이론] 그래프 절단·사용자 정보·GrabCut의 배경
# 근거: 4장 영역 분할_2 28,36~46쪽
# 읽는 순서: 선행 이론 → 관련 코드 앞의 서술형 요약 → 계산·출력 주석.
# PPT의 배경 이론과 현재 코드의 구현 범위를 구분하여 공부한다.
#
# 1. 영역 분할과 의미 분할
# 영역 분할은 밝기·색·텍스처가 비슷한 픽셀을 묶거나 다른 영역의 경계를 정하는 작업이다.
# 의미 분할은 픽셀에 사람·도로 같은 의미 범주까지 연결한다. 단순 색 군집의 번호가 물체 이름은 아니다.
# 이 강의의 고전적 방법은 주로 픽셀 특성으로 분할한 뒤 해석/인식하는 접근이다.
# 단순 배경에서는 임계값이 유리할 수 있고 복잡한 영상에서는 이웃·그래프·사용자 초기 정보 등을 고려한다.
#
# 2. 최대 유량과 최소 절단
# 노드 사이 간선에 용량을 둔 네트워크에서 S에서 T로 보낼 수 있는 최대 흐름을 구한다.
# 병목 때문에 각 간선이 동시에 최대 용량만큼 흐르는 것은 아니다.
# 최대 유량은 S/T를 나누는 최소 절단 용량과 같다. PPT 예에서는 그 값이 6이다.
# 영상에서는 이웃의 유사도와 전경/배경 단자 연결 비용으로 어느 쪽에 픽셀을 남길지 정한다.
#
# 3. 전경 S·배경 T와 간선 비용
# 비슷한 이웃 사이 가중치를 크게 주면 두 픽셀을 서로 다른 쪽으로 자르기 어렵다.
# 확실한 전경은 S에 큰 용량/T에 0, 확실한 배경은 반대로 연결해 사용자의 지정을 유지한다.
# 미확정 픽셀은 색 모델과 이웃 관계를 함께 고려한다. 큰 숫자=밝은 픽셀이라는 뜻은 아니다.
# 픽셀을 T 배경 쪽에 남기면 S-픽셀 간선이 잘리므로 그 용량은 배경으로 배정하는 비용에 대응한다.
# 단자 이름만 보고 이 비용을 반대로 외우지 않는다. 정규화 절단과는 다른 목적 함수이다.
#
# 4. GrabCut은 초기 정보와 색 모델을 반복 사용한다
# 전경/배경 색 모델을 추정하고 그래프 절단을 한 뒤 모델과 분할을 갱신하는 발상이다.
# 사각형 초기화는 밖을 확실한 배경으로, 안은 후보로 시작한다. 안의 모든 픽셀을 확정 전경으로 두는 것은 아니다.
# 마스크 초기화는 사용자가 확실한 전경·배경과 추정 영역을 직접 지정한다. 현재 코드는 이 방식이다.
# 레이블 0/1/2/3은 확실한 배경/전경, 추정 배경/전경이다. 일반 이진 0/255와 다른 약속이다.
# 최종 표시에는 0/2를 0, 1/3을 1로 합쳐 BGR에 곱한다. 모델 저장 배열과 마스크 배열도 구분한다.
# 초기 지정·색 겹침·가는 형상에 따라 오류가 남을 수 있다. 사용자 입력을 주지 않은 실행은 올바른 검증이 아니다.
# ==================================================================


import numpy as np
import cv2 as cv
#%%
img=cv.imread('soccer.jpg')
img_show=np.copy(img) # 깊은 복사

# [서술형] GrabCut 대화식 물체 분할
# 관련 PPT: 영역 분할_2 36~46쪽
# 특징: 사용자의 전경/배경 초기 정보에서 색 모델 추정과 그래프 절단을 반복한다.
# 장점: 사용자가 선택한 관심 물체를 색 분포와 이웃 관계를 함께 이용해 분리할 수 있다.
# 단점: 초기 지정과 전경·배경 색 겹침에 영향을 받으며 가는 부분이나 유사한 배경에서 오류가 남는다.
# 활용: 전경/배경을 대략 알려 줄 수 있는 물체 추출과 배경 제거.
# 한 문장 답안: GrabCut은 적은 초기 정보로 물체를 분리하지만 사용자 지정과 색 모델의 품질에 영향을 받는다.
# [계산·코드 연결]
# 이 코드는 마스크 초기화 방식. 전체를 추정 배경 2로 놓고 왼쪽=전경 1, 오른쪽=배경 0을 칠한다.
# S/T 간 최소 절단 용량은 최대 유량과 같다. 이웃 유사도와 전경/배경 배정 비용을 함께 고려한다.
# 확실한 지정은 큰 용량으로 유지하며 미확정 픽셀을 계산한다. Ncut과 같은 목적 함수는 아니다.
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
# [조건/브로드캐스팅 문제] np.where(조건,참값,거짓값)이므로 0/2->0, 1/3->1이다.
# 예: mask 행 [0,1,2,3]은 mask2 행 [0,1,0,1]로 바뀐다.
# mask2.shape=(948,1434), mask2[:,:,np.newaxis].shape=(948,1434,1)이다.
# BGR 세 채널에 동일한 0/1이 곱해져 전경은 보존하고 배경은 [0,0,0]이 된다.
mask2=np.where((mask==cv.GC_BGD)|(mask==cv.GC_PR_BGD), 0, 1).astype('uint8')
grab=img*mask2[:,:, np.newaxis] # mask brodcasting: 

cv.imshow("Grab Cut Image", grab)
cv.waitKey()

cv.destroyAllWindows()
