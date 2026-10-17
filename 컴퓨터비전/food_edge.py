import cv2 as cv
import numpy as np
import sys

#%%
img=cv.imread('food.jpg')
if img is None:
    sys.exit('No File exists.')

gray=cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# 한 픽셀의 미분·강도·방향 [에지 검출_1 PPT 9~20, 29쪽]
# CV_32F로 부호를 유지한다. atan2(dy,dx)는 사분면을 반영한 그레디언트 방향이고
# 실제 경계의 접선 방향은 이에 수직이다(180도 주기로 비교). 배열의 [30,40]은 (y,x)이다.
grad_x=cv.Sobel(gray, cv.CV_32F, 1, 0, ksize=3)
grad_y=cv.Sobel(gray, cv.CV_32F, 0, 1, ksize=3)

sobel_x=cv.convertScaleAbs(grad_x)
sobel_y=cv.convertScaleAbs(grad_y)

edge_strength=np.sqrt(grad_x**2+grad_y**2)
gradient_direction=np.degrees(np.arctan2(grad_y, grad_x))

# (y,x)=(30,40)의 미분값, 에지 강도, 그레디언트 방향
print('dy:', grad_y[30,40])
print('dx:', grad_x[30,40])
print('edge strength:', edge_strength[30,40])
print('gradient direction (degrees):', gradient_direction[30,40])

# 출력용 에지 강도 맵을 0~255로 변환
max_strength=np.max(edge_strength)
edge_strength=np.uint8(edge_strength/max_strength*255)

#%%
# 3x3 가우시안 필터, sigmaX=sigmaY=1
# 잡음 완화 후 Canny [영상처리_2 PPT 25~27쪽 / 에지 검출_1 PPT 24, 29쪽]
# 3x3, sigmaX=sigmaY=1로 먼저 부드럽게 한다. 잡음 에지는 줄지만 약한 경계도 함께 약해질 수 있다.
# 그 뒤의 Canny 두 임계값은 약한 에지의 연결 유지 여부를 조절한다.
blur=cv.GaussianBlur(gray, (3,3), sigmaX=1, sigmaY=1)

# ch4_2.py와 같은 두 가지 임계값 사용
canny1=cv.Canny(blur, 50, 150)
canny2=cv.Canny(blur, 100, 200)

#%%
cv.imshow('Original', gray)
cv.waitKey()

cv.imshow('sobel_x', sobel_x)
cv.waitKey()

cv.imshow('sobel_y', sobel_y)
cv.waitKey()

cv.imshow('sobel_strength', edge_strength)
cv.waitKey()

cv.imshow('Gaussian', blur)
cv.waitKey()

cv.imshow('Canny1', canny1)
cv.waitKey()

cv.imshow('Canny2', canny2)
cv.waitKey()
cv.destroyAllWindows()
