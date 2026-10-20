
#%% 개념 요약 — 정의·장점·단점, 함수 사용법은 해당 코드 옆 참고

#%% [개념] 색 임계값을 이용한 물체 후보 분리
# 정의: 정해 둔 색 범위에 속하는 픽셀을 골라 물체 후보 마스크를 만드는 방법이다.
# 장점: 계산과 구현이 단순하고, 원하는 색이 배경과 다르면 후보를 빠르게 좁힐 수 있다.
# 단점: 범위를 좁히면 물체가 누락되고 넓히면 배경이 포함된다. 조명·그림자·비슷한 색에 민감하다.
# PPT: 영상처리_1 19~23쪽

#region 핵심 이론·함수 참고
# [핵심 이론] 색 후보·이진 마스크·결과 영상의 차이
# 근거: 3장 영상처리_1 19~23쪽
# 1. 색 분할은 무엇을 결정하는가?
# 각 픽셀이 정한 색 범위 안인지 판단하여 후보/비후보를 나눈다. 물체의 이름을 인식하는 단계는 아니다.
# 2. inRange와 논리 연산
# inRange는 각 채널이 하한<=값<=상한을 모두 만족하면 255, 아니면 0인 단일 채널 마스크를 만든다.
# 3. 흰 배경에 원본을 복사하는 원리
# src는 원본 BGR, mask는 선택 지도, dst는 결과를 기록할 BGR 영상이다.
# dst를 세 채널 모두 255로 채우면 흰색이다. mask!=0인 위치에만 src의 픽셀을 복사한다.
# bitwise_and에 마스크를 쓰는 검은 배경 결과와 흰 dst에 copyTo한 결과는 다르다.

#endregion

import cv2 as cv
from cv_assets import read_image
import sys
import numpy as np
#%%
# [함수 사용: 입출력·창]
# img=cv.imread(path, flags=cv.IMREAD_COLOR): 파일→배열. 기본 BGR, 읽기 실패 시 None인지 확인한다.
# cv.imshow(winname,img): 이름이 winname인 창에 배열을 표시한다. 저장 함수는 아니다.
# key=cv.waitKey(delay=0): GUI 이벤트·키 처리. 0은 계속 대기, 양수는 ms, 입력 없으면 보통 -1이다.
# cv.destroyAllWindows(): OpenCV 창을 닫는다. 캡처 자원 해제는 cap.release()이다.
img=read_image('BALLOON.bmp') ; type(img)
if img is None:
    sys.exit("No File exists.")
cv.imshow("Original", img) ; cv.waitKey()
#%%
# 조건은 90<=H<=130 AND 60<=S<=255 AND 50<=V<=255이다. 상·하한도 포함한다.
# 현재 img/hsv=(239,296,3), mask=(239,296) uint8. 채널별 마스크 세 장을 반환하는 것은 아니다.
# full_like(img,255)는 원본 크기의 흰 BGR 영상이며 copyTo의 복사 대상이다.
# [함수 사용] dst=cv.cvtColor(src, code)
# src=입력 영상, code=색 변환 상수. 예: BGR2GRAY·BGR2HSV·HSV2BGR. 변환 영상을 반환한다.
# [PPT 인자 보충] 영상처리_1 16쪽: dstCn=0은 출력 채널 수 자동 결정. 출력 높이·너비와 깊이는 같고 채널 수는 변환에 따라 달라진다.
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
# [함수 사용] mask=cv.inRange(src, lowerb, upperb)
# 각 채널의 하한·상한을 모두 만족하면 255, 아니면 0인 uint8 마스크를 반환한다. 경계값도 포함한다.
mask = cv.inRange(hsv, (10, 60, 50), (25, 255, 255))
# [함수 사용] a=np.full_like(src, fill_value)
# src와 같은 shape·dtype으로 fill_value를 채운 배열을 반환한다. np.full_like(img,255)는 흰 배경 준비이다.
dst = np.full_like(img, 255) 
#%%
# 흰 배경에 검출 영역 복사 [영상처리_1 PPT 22~23쪽]
# dst는 원본과 같은 크기·자료형의 흰 영상이다. mask가 0이 아닌 곳만 원본을 복사한다.
# 따라서 마스크 밖은 255인 흰색으로 남는다. 마스크 자체와 컬러 결과는 서로 다른 영상이다.
# [결과 예측] mask 값이 255인 픽셀은 원래 BGR, 0인 픽셀은 [255,255,255]가 된다.
# copyTo는 전달한 dst 자체에도 기록한다. 호출 후 dst도 검출 결과이므로 더 이상 모두 흰 배열이 아니다.
# bitwise_and(img,img,mask=mask)는 초기 배경이 검은 결과이므로 흰 dst에 복사한 결과와 다르다.
# [함수 사용] dst=cv.copyTo(src, mask, dst)
# mask가 0이 아닌 위치의 src를 dst에 복사한다. 기존 배경을 남기려면 같은 크기·dtype의 dst를 미리 채워 둔다.
# [PPT 인자 보충] 영상처리_1 22~23쪽: cv.copyTo(src,mask,dst)는 mask!=0 위치만 기존 dst에 복사한다. dst 생략으로 새 배열을 만들면 나머지는 0이다.
detected = cv.copyTo(img, mask, dst)  
# [함수 사용] dst=cv.bitwise_and(src1, src2, mask=mask)
# 원소별 비트 AND 결과를 반환한다. 선택 인자 mask는 0이 아닌 위치만 계산·기록하게 하는 uint8 마스크이다.
#detected=cv.bitwise_and(img, img, mask)
cv.imshow("Result", detected); cv.waitKey()
cv.destroyAllWindows()
