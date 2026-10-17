
# ==================================================================
# [선행 이론] 에지 측정의 원리와 과제 출력의 의미
# 근거: 4장 에지 검출_1 4~29쪽
# 읽는 순서: 선행 이론 → 관련 코드 앞의 서술형 요약 → 계산·출력 주석.
# PPT의 배경 이론과 현재 코드의 구현 범위를 구분하여 공부한다.
#
# 1. 에지와 영역은 같은 영상을 다른 관점으로 본다
# 에지는 명암·색·텍스처가 급격히 달라지는 곳, 영역은 특성이 비슷한 픽셀을 묶은 집합이다.
# 물체 내부보다 경계에서 변화가 크다는 가정으로 에지를 찾지만 그림자·질감도 큰 변화를 만든다.
# 따라서 에지 검출 결과가 의미상 물체의 외곽선 정답과 같다고 보장되지 않는다.
# 사람은 물체 모델·문맥을 이용하지만 여기의 고전적 연산은 픽셀 특성만으로 처리한다.
#
# 2. 디지털 미분과 경계 모델
# 1차 차분은 f(x+1)-f(x), 중심 차분은 (f(x+1)-f(x-1))/2로 생각할 수 있다.
# 2차 차분은 f(x-1)-2*f(x)+f(x+1). 1차 응답의 큰 값과 2차 응답의 영교차를 구분한다.
# 이상적인 계단 경계와 실제 여러 픽셀에 걸친 램프 경계는 응답의 폭·형태가 다르다.
# 미분은 잡음 변화도 크게 만들므로 평활화와 함께 고려한다.
#
# 3. 크기와 방향
# 그레디언트=(dx,dy), 유클리드 강도=sqrt(dx**2+dy**2), 방향=atan2(dy,dx)이다.
# 그레디언트는 밝기가 증가하는 법선 방향이고 경계 접선은 이에 수직이다.
# dx=3,dy=4면 크기는 5, L1 방식 abs(dx)+abs(dy)는 7로 서로 다르다.
# 영상은 y가 아래로 증가한다. 각도의 축과 단위를 확인하고 180도 주기의 경계 방향과 구별한다.
# 음수 미분을 unsigned로 잘라 버리면 방향 정보를 잃으므로 표시용 절댓값과 계산용 미분을 분리한다.
#
# 4. 이 예제에서 서로 다른 두 흐름
# 원본 gray에서 Sobel을 계산하여 지정 위치의 dx/dy/강도/그레디언트 방향을 출력한다.
# 그 뒤 별도로 gray를 Gaussian 3×3, sigmaX=sigmaY=1로 평활화하고 Canny 두 임계값 쌍을 비교한다.
# 뒤의 흐림을 바꿔도 앞에서 이미 계산한 미분 출력이 자동으로 바뀌는 것은 아니다.
# Canny는 NMS로 얇게 하고 두 기준과 연결 관계로 약한 에지를 판정한다. 낮은 기준만 넘으면 모두 남는 것은 아니다.
# 과제의 '에지 방향'을 접선으로 답할 때는 현재 출력한 그레디언트 방향에 수직이라는 관계를 설명한다.
# 출력용 0~255 강도 정규화와 원래 sqrt 강도 수치를 구분한다. 아래 print는 정규화 전이다.
# ==================================================================

import cv2 as cv
import numpy as np
import sys

#%%
img=cv.imread('food.jpg')
if img is None:
    sys.exit('No File exists.')

gray=cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# [서술형] 음식 영상의 미분·에지 분석
# 관련 PPT: 에지 검출_1 9~20,24,29쪽
# 특징: Sobel로 국소 밝기 변화의 강도·방향을 구하고 Gaussian 후 Canny 결과를 비교한다.
# 장점: 변화가 큰 경계를 수치와 영상으로 함께 확인하고 전처리·임계값의 영향을 비교할 수 있다.
# 단점: 잡음·그림자·질감도 반응하고 가우시안이 약한 진짜 경계까지 약화시킬 수 있다.
# 활용: 한 픽셀의 dx/dy/강도 계산 문제, Sobel과 Canny 및 흐림 전후 비교.
# 한 문장 답안: 미분 기반 에지 분석은 밝기 변화에 유리하지만 경계의 물체 의미를 자동으로 판정하지 않는다.
# [계산·코드 연결]
# CV_32F는 부호 보존. atan2(dy,dx)는 그레디언트 방향이고 실제 에지 접선은 수직이다.
# 배열 [30,40]은 (y,x). 현재 출력값과 계산 예는 아래 print 묶음에서 확인한다.
grad_x=cv.Sobel(gray, cv.CV_32F, 1, 0, ksize=3)
grad_y=cv.Sobel(gray, cv.CV_32F, 0, 1, ksize=3)

sobel_x=cv.convertScaleAbs(grad_x)
sobel_y=cv.convertScaleAbs(grad_y)

edge_strength=np.sqrt(grad_x**2+grad_y**2)
gradient_direction=np.degrees(np.arctan2(grad_y, grad_x))

# (y,x)=(30,40)의 미분값, 에지 강도, 그레디언트 방향
# [출력 확인: 현재 food.jpg, (y,x)=(30,40)]
# 부호까지 사용하면 atan2(-5,1)=-78.69006도, 강도=sqrt(1+25)=5.0990195이다.
# 실제 에지 접선은 그레디언트에 90도를 더한 11.30994도(180도 주기)이다.
# convertScaleAbs 후에는 dy의 - 부호가 없어지므로 방향 계산에 그 표시 영상을 사용하면 안 된다.
# 아래 print는 원래 강도를 출력한다. 뒤에 0~255로 정규화한 화면값을 출력하는 것이 아니다.
print('dy:', grad_y[30,40])  # 출력: dy: -5.0
print('dx:', grad_x[30,40])  # 출력: dx: 1.0
print('edge strength:', edge_strength[30,40])  # 출력: edge strength: 5.0990195
print('gradient direction (degrees):', gradient_direction[30,40])  # 출력: gradient direction (degrees): -78.69006

# 출력용 에지 강도 맵을 0~255로 변환
max_strength=np.max(edge_strength)
edge_strength=np.uint8(edge_strength/max_strength*255)

#%%
# 3x3 가우시안 필터, sigmaX=sigmaY=1
# 잡음 완화 후 Canny [영상처리_2 PPT 25~27쪽 / 에지 검출_1 PPT 24, 29쪽]
# 3x3, sigmaX=sigmaY=1로 먼저 부드럽게 한다. 잡음 에지는 줄지만 약한 경계도 함께 약해질 수 있다.
# 그 뒤의 Canny 두 임계값은 약한 에지의 연결 유지 여부를 조절한다.
# [이론/과정 구분: 에지 검출_1 PPT 24,29쪽]
# 위 Sobel/print는 원본 gray에서 계산했고, 아래 Canny는 Gaussian 결과에서 계산한다.
# 따라서 가우시안 설정을 바꿔도 위에 이미 출력한 dy/dx가 자동으로 다시 계산되지는 않는다.
# 흐림은 잡음의 고주파 응답을 줄이지만 약한 진짜 경계도 약해질 수 있고 그림자만 따로 지우지는 않는다.
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
