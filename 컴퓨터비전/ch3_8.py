import cv2 as cv
import sys

#%% 영상의 일부를 잘라 보간법 비교 [영상처리_2 PPT 38~42쪽, 프로그램 3-8]
# PPT의 rose.png 대신 프로젝트에 있는 soccer.jpg를 사용한다(run.py에서 실행).
# 배열은 [y,x]이며 끝 인덱스는 포함하지 않는다. copy는 원본에 사각형을 그릴 때 패치를 보호한다.
img=cv.imread('soccer.jpg')
if img is None:
    sys.exit('No File exists.')

patch=img[250:350,170:270,:].copy()
img=cv.rectangle(img,(170,250),(270,350),(255,0,0),3)

#%% 같은 패치를 5배 확대
# 최근접은 한 픽셀, 양선형은 주변 4개, 양3차는 주변 16개 픽셀을 이용한다.
# 최근접은 계단 모양이 뚜렷하고, 보간은 부드럽지만 원래 없던 세부 정보를 복원하지는 못한다.
patch1=cv.resize(patch,(0,0),fx=5,fy=5,interpolation=cv.INTER_NEAREST)
patch2=cv.resize(patch,(0,0),fx=5,fy=5,interpolation=cv.INTER_LINEAR)
patch3=cv.resize(patch,(0,0),fx=5,fy=5,interpolation=cv.INTER_CUBIC)

cv.imshow('Original',img)
cv.imshow('Nearest',patch1)
cv.imshow('Linear',patch2)
cv.imshow('Cubic',patch3)
cv.waitKey()
cv.destroyAllWindows()
