
# ==================================================================
# [선행 이론] 색 공간·HSV 채널 분석의 배경
# 근거: 3장 영상처리_1 8~19쪽
# 읽는 순서: 선행 이론 → 관련 코드 앞의 서술형 요약 → 계산·출력 주석.
# PPT의 배경 이론과 현재 코드의 구현 범위를 구분하여 공부한다.
#
# 1. 색 공간을 바꾸는 이유
# 같은 색도 조명이 달라지면 BGR 숫자가 달라진다. 표현을 바꾸면 필요한 성분을 따로 다루기 쉽다.
# 색 공간 변환은 새로운 물체 정보를 만드는 것이 아니라 같은 색을 다른 좌표로 나타내는 것이다.
# RGB/BGR: 세 빛 성분을 직접 표현한다. 화면 표시와 저장에 편리하지만 밝기가 여러 채널에 섞인다.
# HSV: 색상 H, 채도 S, 밝기 V로 나눈다. 특정 색을 고르거나 밝기만 바꾸기에 편리하다.
# Lab: 밝기 L과 녹색↔빨강 a, 파랑↔노랑 b의 색 성분으로 나눈다.
# Lab의 이론 좌표와 OpenCV uint8 저장값의 범위·오프셋은 같지 않으므로 수치를 그대로 섞지 않는다.
#
# 2. HSV의 세 성분과 주의점
# H(hue)는 색상 원의 위치, S(saturation)는 무채색에서 벗어난 정도, V(value)는 밝기 성분이다.
# uint8 일반 BGR2HSV에서 H는 0~179, S/V는 0~255이다. 다른 자료형·변환 옵션은 범위를 확인한다.
# 색상 원은 양 끝이 이어진다. 붉은색은 낮은 H와 높은 H의 두 구간으로 잡을 수 있다.
# S가 작으면 거의 회색이라 H의 의미가 약하다. 매우 어두우면 색/채도 정보를 안정적으로 믿기 어렵다.
# H의 최대 허용값이 179라고 모든 영상의 np.max(H)가 반드시 179인 것은 아니다.
#
# 3. 채널 분리와 재결합
# split은 3채널 영상을 같은 높이·너비의 세 장으로 나누고 merge는 다시 묶는다.
# V를 바꾸고 H/S를 유지하면 색상·채도를 유지한 밝기 성분 수정 결과를 비교할 수 있다.
# 현재 V를 모두 255로 만든다. 상수 밝기 영상에 equalizeHist를 적용해도 대비가 새로 생기지 않는다.
# HSV 배열을 BGR이라고 생각하고 표시하면 색이 다르다. 최종 표시는 HSV2BGR 변환을 사용한다.
# 이 파일은 색 성분의 관찰·수정 예제이며, 피부색 분할이나 물체 이름 인식까지 하지는 않는다.
# ==================================================================


#%% [PPT 전체 확인 보충] HSV 해석·복사 인자
# 관련 PPT: 영상처리_1 18~22쪽.
# 정규화 RGB에서 V=max(R,G,B), S=(max-min)/max이며 max=0이면 S=0이다.
# (0.5,0,0)과 (1,0,0)은 둘 다 S=1이나 V는 .5와 1로 다르다. 채도가 높아도 어두운 피부색 판단은 불안정할 수 있다.
# 밝기가 매우 작을 때 S, 채도가 작을 때 H의 신뢰도가 낮다. uint8 OpenCV HSV는 H=0~179, S/V=0~255이다.
# cv.copyTo(src,mask,dst)는 mask의 255를 복사하는 것이 아니라 mask!=0인 위치의 src를 복사한다.
# 예: dst를 흰색으로 미리 만든 뒤 copyTo를 하면 선택되지 않은 배경은 계속 흰색이다.
# cv.bitwise_and(src,src,mask=mask)처럼 mask를 키워드로 쓴다. 세 번째 위치 인자는 mask가 아니라 dst이다.

import cv2 as cv
from cv_assets import read_image
import sys
import numpy as np

# [함수 사용: 입출력·창]
# img=cv.imread(path, flags=cv.IMREAD_COLOR): 파일→배열. 기본 BGR, 읽기 실패 시 None인지 확인한다.
# cv.imshow(winname,img): 이름이 winname인 창에 배열을 표시한다. 저장 함수는 아니다.
# key=cv.waitKey(delay=0): GUI 이벤트·키 처리. 0은 계속 대기, 양수는 ms, 입력 없으면 보통 -1이다.
# cv.destroyAllWindows(): OpenCV 창을 닫는다. 캡처 자원 해제는 cap.release()이다.
img=read_image('soccer.jpg')
if img is None:
    sys.exit("No File exists.")

cv.imshow("Original", img); cv.waitKey() # 원본 
#%%
# 색 공간 변환과 채널 분리 [영상처리_1 PPT 11~19쪽]
# BGR을 HSV로 바꾸면 H=색상, S=채도, V=밝기 성분을 따로 다룰 수 있다.
# uint8 HSV에서 H는 0~179, S/V는 0~255이다. 영상의 실제 최댓값이 항상 상한인 것은 아니다.
# [함수 사용] dst=cv.cvtColor(src, code)
# src=입력 영상, code=색 변환 상수. 예: BGR2GRAY·BGR2HSV·HSV2BGR. 변환 영상을 반환한다.
# [PPT 인자 보충] 영상처리_1 16쪽: dstCn=0은 출력 채널 수 자동 결정. 출력 높이·너비와 깊이는 같고 채널 수는 변환에 따라 달라진다.
hsv=cv.cvtColor(img, cv.COLOR_BGR2HSV)
cv.imshow('Hsv', hsv) ; cv.waitKey() # hsv 는 hue, saturation, value 의 약자로 색상, 채도, 명도를 의미한다. 따라서 hsv 영상은 3채널 영상이 된다.
#%%
# [서술형] HSV 색 공간 분석
# 관련 PPT: 영상처리_1 11~19쪽
# 특징: 색상 H, 채도 S, 밝기 V로 나누어 색과 밝기를 따로 다룬다.
# 장점: BGR 세 채널을 동시에 조절하는 것보다 특정 색 범위를 지정하기 쉽다.
# 단점: 어둡거나 채도가 낮은 픽셀에서는 H가 불안정하며, 색이 같은 배경도 통과한다.
# 활용: 파란 풍선·피부색 후보 검출, 밝기 성분을 바꾼 영상 비교.
# 한 문장 답안: HSV는 색상 기반 분리에 편리하지만 조명과 저채도 영역의 영향을 받는다.
# [계산·코드 연결]
# uint8에서 H=0~179, S/V=0~255. 실제 영상의 최댓값이 항상 범위 상한은 아니다.
# split 후 h/s/v는 각각 (948,1434). V는 원래 RGB 성분의 최댓값에 대응한다.
# HSV 배열을 BGR처럼 바로 표시하면 원래 색이 아니다. 복원은 HSV2BGR로 한다.
# [함수 사용] channels=cv.split(src)
# src의 채널을 같은 높이·너비의 배열들로 분리한다. HSV는 h,s,v=cv.split(hsv)로 받는다.
h, s, v = cv.split(hsv)
type(h)
cv.imshow('Hue', h) ; cv.waitKey() # uint8 H의 범위는 0~179이며, 실제 최댓값은 영상에 따라 달라진다.
# [함수 사용] np.set_printoptions(precision=4, suppress=True)
# 배열 출력 자릿수·작은 수의 지수 표기 등을 설정한다. 원래 배열의 계산값이나 dtype을 바꾸지는 않는다.
#np.set_printoptions(threshold=np.inf, linewidth=np.inf)
#print(h)
# [함수 사용] value=np.max(a, axis=None)
# axis 생략 시 전체 원소 중 최댓값, 지정 시 해당 축의 최댓값을 반환한다.
print(np.max(h))  # 출력: 179 (현재 영상의 최댓값)

cv.imshow('Sat', s) ; cv.waitKey() # S는 0~255 범위이다. 범위의 상한과 영상의 실제 최댓값을 구분한다.
cv.imshow('Value', v) ; cv.waitKey() # V는 0~255 범위의 밝기 성분이다.
print(v.ndim)  # 출력: 2 (단일 채널 배열의 축 수)
print(v.shape)  # 출력: (948, 1434)
# [배열 출력 확인] 현재 v의 맨 윗줄은 [162 163 162 ... 115 121 120]이다.
# ...는 큰 배열의 중간 원소를 생략했다는 뜻이고, 실제 배열에 점 세 개가 들어 있는 것은 아니다.
# 이 print는 v를 255로 덮어쓰기 전이다. 아래 수정 후 v를 출력하면 모든 원소가 255이다.
print(v)
# 채널 수정 후 복원 [영상처리_1 PPT 18쪽]
# V를 모두 255로 바꾸고 원래 H/S와 합친다. 현재 크기 (948,1434)는 soccer.jpg 전용이다.
# 상수 255 영상에 equalizeHist를 적용해도 대비는 생기지 않는다.
# [함수 사용] a=np.zeros(shape, dtype=float)
# shape=배열 크기, dtype=자료형. 0으로 채운 배열을 만든다. 영상 마스크에는 보통 dtype=np.uint8을 지정한다.
v=np.zeros((948, 1434), np.uint8)+255
# [함수 사용] dst=cv.equalizeHist(src)
# src=uint8 단일 채널 영상. 히스토그램을 평활화한 같은 크기의 명암 영상을 반환한다.
v=cv.equalizeHist(v)
# [함수 사용] dst=cv.merge(mv)
# mv=합칠 채널 배열 목록(예: [h,s,v]). 높이·너비와 dtype이 같아야 하며, 결과 채널 수는 입력 채널 수의 합이다.
# [PPT 인자 보충] 영상처리_1 18쪽: mv는 [h,s,v] 같은 배열 목록. dst는 선택 출력 배열이며 Python에서는 반환값을 받는 형태가 간단하다.
dst=cv.merge([h,s,v])
new=cv.cvtColor(dst,cv.COLOR_HSV2BGR)
cv.imshow('New', new) # equalizeHist() 함수는 히스토그램 평활화 기능을 수행하는 함수이다. 히스토그램 평활화는 영상의 명암비를 향상시키는 방법 중 하나로, 영상의 밝기 분포를 균일하게 만들어서 어두운 영역과 밝은 영역의 대비를 높이는 기법이다.
cv.waitKey()
cv.destroyAllWindows()