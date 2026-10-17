import cv2 as cv
import sys
import numpy as np

img=cv.imread('soccer.jpg')
if img is None:
    sys.exit("No File exists.")

cv.imshow("Original", img); cv.waitKey() # 원본 
#%%
# 색 공간 변환과 채널 분리 [영상처리_1 PPT 11~19쪽]
# BGR을 HSV로 바꾸면 H=색상, S=채도, V=밝기 성분을 따로 다룰 수 있다.
# uint8 HSV에서 H는 0~179, S/V는 0~255이다. 영상의 실제 최댓값이 항상 상한인 것은 아니다.
hsv=cv.cvtColor(img, cv.COLOR_BGR2HSV)
cv.imshow('Hsv', hsv) ; cv.waitKey() # hsv 는 hue, saturation, value 의 약자로 색상, 채도, 명도를 의미한다. 따라서 hsv 영상은 3채널 영상이 된다.
#%%
h, s, v = cv.split(hsv)
type(h)
cv.imshow('Hue', h) ; cv.waitKey() # hue 는 색상 정보이므로 0~179 범위의 값만 가지게 된다. 따라서 np.max(h) 를 출력하면 179가 나온다.
#np.set_printoptions(threshold=np.inf, linewidth=np.inf)
#print(h)
print(np.max(h))

cv.imshow('Sat', s) ; cv.waitKey() # saturation 은 채도 정보이므로 0~255 범위의 값만 가지게 된다. 따라서 np.max(s) 를 출력하면 255가 나온다.
cv.imshow('Value', v) ; cv.waitKey() # value 는 명도 정보이므로 0~255 범위의 값만 가지게 된다. 따라서 np.max(v) 를 출력하면 255가 나온다.
print(v.ndim)
print(v.shape)
print(v)
# 채널 수정 후 복원 [영상처리_1 PPT 18쪽]
# V를 모두 255로 바꾸고 원래 H/S와 합친다. 현재 크기 (948,1434)는 soccer.jpg 전용이다.
# 상수 255 영상에 equalizeHist를 적용해도 대비는 생기지 않는다.
v=np.zeros((948, 1434), np.uint8)+255
v=cv.equalizeHist(v)
dst=cv.merge([h,s,v])
new=cv.cvtColor(dst,cv.COLOR_HSV2BGR)
cv.imshow('New', new) # equalizeHist() 함수는 히스토그램 평활화 기능을 수행하는 함수이다. 히스토그램 평활화는 영상의 명암비를 향상시키는 방법 중 하나로, 영상의 밝기 분포를 균일하게 만들어서 어두운 영역과 밝은 영역의 대비를 높이는 기법이다.
cv.waitKey()
cv.destroyAllWindows()