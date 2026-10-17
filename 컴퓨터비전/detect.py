import cv2 as cv
import sys
import numpy as np
#%%
img=cv.imread('BALLOON.bmp') ; type(img)
if img is None:
    sys.exit("No File exists.")
cv.imshow("Original", img) ; cv.waitKey()
#%%
# HSV 범위로 파란색 분리 [영상처리_1 PPT 19~23쪽]
# inRange는 H/S/V가 각 하한~상한 안에 모두 들어오면 255, 아니면 0인 uint8 마스크를 만든다.
# H로 파란색을 고르고 S/V 하한으로 무채색·매우 어두운 픽셀을 제외한다.
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
mask = cv.inRange(hsv, (90, 60, 50), (130, 255, 255))
dst = np.full_like(img, 255) 
#%%
# 흰 배경에 검출 영역 복사 [영상처리_1 PPT 22~23쪽]
# dst는 원본과 같은 크기·자료형의 흰 영상이다. mask가 0이 아닌 곳만 원본을 복사한다.
# 따라서 마스크 밖은 255인 흰색으로 남는다. 마스크 자체와 컬러 결과는 서로 다른 영상이다.
detected = cv.copyTo(img, mask, dst)  
#detected=cv.bitwise_and(img, img, mask)
cv.imshow("Result", detected); cv.waitKey()
cv.destroyAllWindows()