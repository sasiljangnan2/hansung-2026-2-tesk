# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 12:16:58 2023

@author: BigData
"""

#%% 개념 요약 — 정의·장점·단점, 함수 사용법은 해당 코드 옆 참고

#%% [개념] 히스토그램을 이용한 전역 분석
# 정의: 각 밝기 값 또는 밝기 구간에 속하는 픽셀 수를 세어 영상 전체의 밝기 분포를 나타내는 방법이다.
# 장점: 표현이 간단하고 전체 밝기·대비·분포를 빠르게 비교할 수 있다.
# 단점: 공간 배치를 잃어 서로 다른 무늬가 같은 히스토그램을 가질 수 있다.
# PPT: 영상처리_1 25~27쪽 / 영역 특징 15~17쪽

#%% [개념] 히스토그램 평활화
# 정의: 누적 밝기 분포를 이용해 밝기 값을 다시 배치하여 영상의 대비를 조절하는 방법이다.
# 장점: 좁은 밝기 범위에 몰린 영상의 대비를 임계값 수동 조절 없이 넓힐 수 있다.
# 단점: 잡음도 강조될 수 있고 색 채널별 적용은 색을 왜곡할 수 있다. 항상 보기 좋아지는 것은 아니다.
# PPT: 영상처리_2 15~21쪽

#%% [개념] GLCM(동시 발생 행렬) 분석
# 정의: 정한 거리·방향에 있는 두 픽셀의 밝기 조합이 나타나는 빈도를 행렬로 표현하는 텍스처 분석 방법이다.
# 장점: 전역 히스토그램이 놓치는 이웃 관계와 텍스처의 대비·동질성을 반영한다.
# 단점: 밝기 양자화·거리·방향 설정에 영향을 받고 밝기 단계가 많으면 행렬도 커진다.
# PPT: 영역 특징 18~19쪽
# 참고 이론이며 이 파일에서 실행하지 않음.

#%% [개념] LBP(지역 이진 패턴) 분석
# 정의: 중심 픽셀과 주변 픽셀의 대소관계를 이진 코드로 표현하는 지역 텍스처 기술자이다.
# 장점: 계산이 단순하고 공통의 단조 증가 밝기 변환에서는 이웃 대소관계가 유지된다.
# 단점: 평탄한 곳의 작은 잡음에도 비트가 바뀌고 고정된 이웃 범위는 스케일 변화에 약하다.
# PPT: 영역 특징 20~21,23쪽
# 참고 이론이며 이 파일에서 실행하지 않음.

#%% [개념] LTP(지역 삼진 패턴) 분석
# 정의: 중심과 이웃의 밝기 차이를 허용 구간에 따라 -1·0·1로 표현하는 지역 텍스처 기술자이다.
# 장점: 허용 구간을 두어 중심 부근의 작은 밝기 잡음에 LBP보다 안정적으로 대응한다.
# 단점: t 선택이 필요하고 작지만 유용한 차이도 무시할 수 있다. 고정 범위의 스케일 한계도 있다.
# PPT: 영역 특징 22~23쪽
# 참고 이론이며 이 파일에서 실행하지 않음.

#region 핵심 이론·함수 참고
# [핵심 이론] 히스토그램·텍스처 분석의 전체 배경
# 근거: 3장 영상처리_1 25~27쪽 / 영상처리_2 15~21쪽 / 영역 특징 15~23쪽
# 1. 히스토그램과 위치 정보
# h(k)는 밝기 k의 빈도, p(k)=h(k)/전체 픽셀 수는 확률 분포이다.
# 평균=sum(k*p(k)), r차 중심 모멘트 mu_r=sum((k-평균)**r*p(k)). mu_2는 분산이다.
# 강의의 skew=mu_3는 표준화 왜도 mu_3/sigma**3와 구분한다.
# 2. 평활화는 특징 추출이 아니라 대비 변경
# CDF(k)=sum(p(i),i<=k)는 해당 밝기 이하의 누적 비율이다.
# 같은 입력 밝기는 같은 출력으로 이동하므로 모든 밝기 빈도가 정확히 같아지는 것은 아니다.
# 3. 텍스처와 전역/지역 관계 기술자
# 텍스처는 일정한 패턴의 반복이나 통계적 규칙을 가진 무늬이다. 구조적 방법과 통계적 방법으로 볼 수 있다.
# 4. 동시 발생 행렬 GLCM의 발상
# 거리·방향을 먼저 정하고 이웃 밝기 쌍이 (i,j)인 횟수를 O(i,j)에 센다.
# P(i,j)=O(i,j)/전체 쌍 수로 정규화한 뒤 여러 특징을 구한다.
# 에너지=sum(P**2), 엔트로피=-sum(P*log2(P)), 대비=sum((i-j)**2*P).
# 강의의 동질성=sum(P/(1+abs(i-j))). 분모를 제곱 차이로 쓰는 다른 정의와 섞지 않는다.
# 5. LBP와 LTP의 발상·조건·한계
# LBP는 중심과 8이웃의 대소관계를 8비트로 만들고 각 픽셀 코드의 히스토그램을 사용한다.
# 두 이진 패턴 히스토그램으로 나누는 강의 표현은 256+256=512차원이다. t는 따로 정해야 한다.


#%% [PPT 전체 확인 보충] 컬러 평활화·텍스처 계산 예시
# 관련 PPT: 영상처리_2 15~21쪽 / 영역 특징 16·20~23쪽. 아래는 읽는 주석 예시이며 현재 gray 평활화 실행은 유지한다.
# [컬러 평활화 사용 순서] 밝기 성분만 바꾸어 채널별 평활화의 색 변화를 줄인다.
# 주석 예: ycc=cv.cvtColor(img,cv.COLOR_BGR2YCrCb)  # OpenCV 순서 Y,Cr,Cb; PPT 그림의 Y,Cb,Cr와 구분.
# 주석 예: y,cr,cb=cv.split(ycc); y=cv.equalizeHist(y)
# 주석 예: result=cv.cvtColor(cv.merge([y,cr,cb]),cv.COLOR_YCrCb2BGR)
# 비교 예: b,g,r=cv.split(img); separate=cv.merge([cv.equalizeHist(b),cv.equalizeHist(g),cv.equalizeHist(r)])
# equalizeHist는 uint8 단일 채널을 받는다. 컬러 3채널을 한 번에 넘기는 함수가 아니다.
#
# [전역 텍스처] 명암 분산을 (L-1)²로 정규화해 v라고 하면 smooth=1-1/(1+v). v=0이면 0, 커질수록 1에 가까워진다.
# 에지 강도 히스토그램은 강도 구간별 빈도, 방향 히스토그램은 각도 구간별 빈도이다. 강의는 방향을 8단계로 양자화한다.
# 두 히스토그램 모두 전역 통계이므로 같은 분포를 가지는 다른 공간 무늬는 구별하지 못할 수 있다.
# [PPT 계산 예] LBP 20쪽의 정해진 순서로 읽은 00110010₂=32+16+2=50. 비트 시작점·읽는 방향이 바뀌면 코드도 달라진다.
# LTP 22쪽은 중심 150, t=5: 145~155는 0, >155는 1, <145는 -1이다(양 끝 포함).
# 해당 그림의 양 패턴 00010010₂=18, 음 패턴 11001001₂=201. 두 256차원 히스토그램을 연결하면 512차원이다.
# 지역별 히스토그램 연결은 얼굴 부위의 위치 정보를 일부 유지한다. 넓은 원형 이웃·보간은 크기 변화 대응을 위한 확장 발상이다.

#endregion

import cv2 as cv
import matplotlib.pyplot as plt
import sys
#%%

# [함수 사용: 입출력·창]
# img=cv.imread(path, flags=cv.IMREAD_COLOR): 파일→배열. 기본 BGR, 읽기 실패 시 None인지 확인한다.
# cv.imshow(winname,img): 이름이 winname인 창에 배열을 표시한다. 저장 함수는 아니다.
# key=cv.waitKey(delay=0): GUI 이벤트·키 처리. 0은 계속 대기, 양수는 ms, 입력 없으면 보통 -1이다.
# cv.destroyAllWindows(): OpenCV 창을 닫는다. 캡처 자원 해제는 cap.release()이다.
img = cv.imread("mistyroad.jpg")

if img is None:
    sys.exit("No File exists.")
    
cv.imshow('Original', img)
cv.waitKey()

# [출력 형식: 입력 영상이 있어야 실행됨]
# 컬러가 높이 H,너비 W이면 아래는 (H,W,3), gray.shape는 (H,W)이다.
# 현재 프로젝트에는 mistyroad.jpg가 없어 구체적인 H/W를 임의로 적지 않았다.
# 파일이 없으면 위 sys.exit에서 종료하므로 이 print 줄까지 도달하지 않는다.
print(img.shape)
#%%
# [함수 사용] dst=cv.cvtColor(src, code)
# src=입력 영상, code=색 변환 상수. 예: BGR2GRAY·BGR2HSV·HSV2BGR. 변환 영상을 반환한다.
# [PPT 인자 보충] 영상처리_1 16쪽: dstCn=0은 출력 채널 수 자동 결정. 출력 높이·너비와 깊이는 같고 채널 수는 변환에 따라 달라진다.
gray=cv.cvtColor(img, cv.COLOR_BGR2GRAY)
print(gray.shape)

# [함수 사용] plt.imshow(X,cmap=None,vmin=None,vmax=None)
# 2D 명암에는 cmap="gray", 컬러는 RGB를 사용한다. vmin/vmax는 명암 표시 범위이며 OpenCV BGR은 변환 후 표시한다.
# [함수 사용] plt.xticks(ticks,labels=None) / plt.yticks(ticks,labels=None)
# 가로/세로 눈금 위치와 표시 문자열을 설정한다. []는 눈금을 숨길 뿐 영상 픽셀을 삭제하지 않는다. [영상처리_2 8쪽]
plt.imshow(gray, cmap='gray'), plt.xticks([]), plt.yticks([])
#%%
# h.shape=(256,1), dtype=float32. h[k]는 해당 밝기 구간의 개수이다.
# p(k)=h(k)/전체수, 평균=sum(k*p(k)), 분산=sum((k-평균)**2*p(k)).
# 엔트로피=-sum(p*log2(p)), 균일성=sum(p*p). p=0인 엔트로피 항은 0으로 처리한다.
# 현재 범위 [0,255]는 255를 제외한다. [0,256]이고 마스크가 없으면 sum(h)=H*W이다.
# [함수 사용] hist=cv.calcHist(images, channels, mask, histSize, ranges)
# images=[영상], channels=[채널 번호], mask=None이면 전체. histSize=bin 수, ranges=상한 제외 범위(8비트 전체는 [0,256]).
# [PPT 인자 보충] 영상처리_1 25~26쪽: images는 [img], channels는 [0]처럼 목록. histSize는 bin 수, ranges의 상한은 제외한다. hist=기존 출력, accumulate=True면 누적한다.
h=cv.calcHist([gray], [0], None, [256], [0, 255])
# [함수 사용] plt.plot(x,y,fmt,color=...,linewidth=...) 또는 plt.plot(y)
# 좌표·곡선을 그린다. x 생략 시 0부터의 인덱스. fmt="--r"는 빨간 파선, lw는 linewidth의 별칭이다.
plt.plot(h, color='r', linewidth=1) 
# [함수 사용] plt.show()
# 준비된 그래프 창을 표시한다. 원본 배열을 저장하거나 픽셀 값을 바꾸는 함수가 아니다.
plt.show()
#%%
# CDF(k)=sum(p(i),i<=k), 강의 식은 round(255*CDF(k)). OpenCV의 최소 누적값 처리와는 구별한다.
# 같은 입력 밝기를 여러 값으로 나눌 수 없어 출력 빈도가 전부 같아질 필요는 없다.
# equal.shape는 gray.shape, dtype=uint8. 이 코드는 텍스처 기술자를 직접 계산하지 않는다.
# [함수 사용] dst=cv.equalizeHist(src)
# src=uint8 단일 채널 영상. 히스토그램을 평활화한 같은 크기의 명암 영상을 반환한다.
equal=cv.equalizeHist(gray)
plt.imshow(equal, cmap='gray'), plt.xticks([]), plt.yticks([])
cv.imshow('Equal', equal)
cv.waitKey()
cv.destroyAllWindows()
#%%
h=cv.calcHist([equal], [0], None, [256], [0, 255])
plt.plot(h, color='r', linewidth=1) 
plt.show()


# [연결 이론: 아래 3개는 현재 파일에 구현되어 있지 않음]
# 정규화 P(i,j)에서 대비=sum((i-j)**2*P), 에너지=sum(P**2). 방향을 명시해야 비교가 가능하다.
#
# 8비트 코드는 0~255. 포화·양자화로 대소관계가 바뀔 수 있고 기본 LBP 자체가 회전 불변인 것도 아니다.
#
# p=150,t=5이면 145~155는 0, 145 미만은 -1, 155 초과는 1이다.
# 두 이진 패턴의 히스토그램으로 나누면 강의의 기본 표현은 256+256=512차원이다.
