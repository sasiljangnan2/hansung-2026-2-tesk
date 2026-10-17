import cv2 as cv
import numpy as np

#%% 영상 읽기 및 HSV 피부색 검출
img=cv.imread("hand_sample2.jpg")
gray=cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# HSV의 색상과 채도로 피부색 검출
# 피부색을 이용한 초기 후보 [영상처리_1 PPT 11~22쪽 / 영역 분할_2 PPT 14쪽]
# H는 원형이어서 붉은 계열을 양 끝의 두 범위로 나누고 OR로 합친다. Lab a 조건은 AND로 추가한다.
# 어두운 손과 배경 그림자의 색이 비슷하면 이 조건만으로 완전한 분리는 보장되지 않는다.
hsv=cv.cvtColor(img, cv.COLOR_BGR2HSV)

# 밝은 피부도 포함하고, 아래 Lab 조건으로 배경 그림자를 추가 제외
mask1=cv.inRange(hsv, (0,40,0), (25,255,255))
mask2=cv.inRange(hsv, (160,40,0), (179,255,255))

binary=mask1 | mask2

# 이 사진에서 붉은 성분이 약한 그림자 제외 (a 채널 기준 134)
lab=cv.cvtColor(img, cv.COLOR_BGR2LAB)
skin=cv.inRange(lab[:,:,1], 134, 255)
binary=cv.bitwise_and(binary, skin)
kernel=cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))

# 작은 메디안 필터로 마스크의 점 잡음과 톱니 모양 완화
binary=cv.medianBlur(binary, 3)

cv.imshow("Color", img); cv.imshow("Gray", gray); cv.imshow("Binary", binary)
cv.waitKey(); cv.destroyAllWindows()

#%% 침식·팽창으로 마커 준비
kernel=cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))
# 침식·팽창으로 확실한 영역 설정 [영역 분할_2 PPT 9~12쪽]
# fg 반복을 늘리면 손 내부 마커가 줄고, bg 팽창 반복을 늘리면 배경 마커가 손에서 더 멀어진다.
# 두 경계 사이의 0 영역에서 워터셰드가 경계를 결정한다. 잘못된 확정 마커까지 자동 수정하는 것은 아니다.
fg=cv.erode(binary,kernel, iterations=5)
cv.imshow("Foreground", fg); cv.waitKey()

# 손 경계 주변에 좁은 미확정 영역을 두고 워터셰드로 경계 결정
bg=cv.dilate(binary, kernel, iterations=2)
cv.imshow("bg", bg);
# 팽창한 손 바깥을 배경 마커(128)로 바꾼 뒤 전경과 합침
t, bg=cv.threshold(bg, 1, 128, cv.THRESH_BINARY_INV)
markers=fg+bg
cv.imshow("Markers", markers)
cv.waitKey()

#%% 워터셰드 실행
# 경계 계산과 표시 [영역 분할_2 PPT 10~14쪽]
# 전경 255/배경 128/미확정 0을 int32로 넘기고 결과 -1인 픽셀을 경계로 그린다.
# 현재 전처리는 medianBlur(중앙값)이며 GaussianBlur(가중 평균)와 다른 필터이다.
markers=np.int32(markers)
# 메디안 필터로 잡음을 줄인 뒤 워터셰드 실행
blur=cv.medianBlur(img, 3)
markers=cv.watershed(blur, markers)
np.max(markers)
np.min(markers)


#%% 경계 및 영역 분할 결과 표시
dst=np.ones(img.shape, np.uint8)*255
dst[markers==-1] = 0
cv.imshow("Watershed", dst)
img[markers==-1]=[255,0,0]
cv.imshow("Color&Boundry", img)
cv.waitKey()

markers=np.uint8(np.clip(markers, 0, 255))
cv.imshow("Segmentation", markers)
cv.waitKey()

cv.destroyAllWindows()
