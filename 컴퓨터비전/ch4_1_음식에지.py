
#%% 개념 요약 — 정의·장점·단점, 함수 사용법은 해당 코드 옆 참고

#%% [개념] 음식 영상의 미분·에지 분석
# 정의: 영상의 가로·세로 미분에서 경계 강도와 방향을 계산하고, Sobel·Canny 결과를 비교하는 실습이다.
# 장점: 변화가 큰 경계를 수치와 영상으로 함께 확인하고 전처리·임계값의 영향을 비교할 수 있다.
# 단점: 잡음·그림자·질감도 반응하고 가우시안이 약한 진짜 경계까지 약화시킬 수 있다.
# PPT: 에지 검출_1 9~20,24,29쪽

#region 핵심 이론·함수 참고
# [핵심 이론] 에지 측정의 원리와 과제 출력의 의미
# 근거: 4장 에지 검출_1 4~29쪽
# 1. 에지와 영역은 같은 영상을 다른 관점으로 본다
# 에지는 명암·색·텍스처가 급격히 달라지는 곳, 영역은 특성이 비슷한 픽셀을 묶은 집합이다.
# 2. 디지털 미분과 경계 모델
# 1차 차분은 f(x+1)-f(x), 중심 차분은 (f(x+1)-f(x-1))/2로 생각할 수 있다.
# 2차 차분은 f(x-1)-2*f(x)+f(x+1). 1차 응답의 큰 값과 2차 응답의 영교차를 구분한다.
# 3. 크기와 방향
# 그레디언트=(dx,dy), 유클리드 강도=sqrt(dx**2+dy**2), 방향=atan2(dy,dx)이다.
# dx=3,dy=4면 크기는 5, L1 방식 abs(dx)+abs(dy)는 7로 서로 다르다.
# 4. 이 예제에서 서로 다른 두 흐름
# 원본 gray에서 Sobel을 계산하여 지정 위치의 dx/dy/강도/그레디언트 방향을 출력한다.
# 그 뒤 별도로 gray를 Gaussian 3×3, sigmaX=sigmaY=1로 평활화하고 Canny 두 임계값 쌍을 비교한다.
# 뒤의 흐림을 바꿔도 앞에서 이미 계산한 미분 출력이 자동으로 바뀌는 것은 아니다.

#endregion

import cv2 as cv
from cv_assets import read_image
import numpy as np
import sys

#%%
# [함수 사용: 입출력·창]
# img=cv.imread(path, flags=cv.IMREAD_COLOR): 파일→배열. 기본 BGR, 읽기 실패 시 None인지 확인한다.
# cv.imshow(winname,img): 이름이 winname인 창에 배열을 표시한다. 저장 함수는 아니다.
# key=cv.waitKey(delay=0): GUI 이벤트·키 처리. 0은 계속 대기, 양수는 ms, 입력 없으면 보통 -1이다.
# cv.destroyAllWindows(): OpenCV 창을 닫는다. 캡처 자원 해제는 cap.release()이다.
img=read_image('food.jpg')
if img is None:
    sys.exit('No File exists.')

# [함수 사용] dst=cv.cvtColor(src, code)
# src=입력 영상, code=색 변환 상수. 예: BGR2GRAY·BGR2HSV·HSV2BGR. 변환 영상을 반환한다.
# [PPT 인자 보충] 영상처리_1 16쪽: dstCn=0은 출력 채널 수 자동 결정. 출력 높이·너비와 깊이는 같고 채널 수는 변환에 따라 달라진다.
gray=cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# CV_32F는 부호 보존. atan2(dy,dx)는 그레디언트 방향이고 실제 에지 접선은 수직이다.
# 배열 [30,40]은 (y,x). 현재 출력값과 계산 예는 아래 print 묶음에서 확인한다.
# [함수 사용] dst=cv.Sobel(src, ddepth, dx, dy, ksize=3)
# dx·dy=각 축의 미분 차수, ksize=커널 크기. 예: CV_32F,1,0은 음수를 보존한 x 미분 결과이다.
# [PPT 인자 보충] 에지 검출 14쪽: dx/dy는 미분 차수. ksize는 보통 1,3,5,7, scale=1은 배율, delta=0은 더할 값, borderType은 가장자리 처리이다.
grad_x=cv.Sobel(gray, cv.CV_32F, 1, 0, ksize=3)
grad_y=cv.Sobel(gray, cv.CV_32F, 0, 1, ksize=3)

# [함수 사용] dst=cv.convertScaleAbs(src, alpha=1, beta=0)
# abs(alpha*src+beta)를 uint8로 포화 변환하여 반환한다. 계산용 부호는 사라지므로 주로 표시에 쓴다.
sobel_x=cv.convertScaleAbs(grad_x)
sobel_y=cv.convertScaleAbs(grad_y)

# [함수 사용] dst=np.sqrt(a)
# 원소별 제곱근을 반환한다. np.sqrt(dx**2+dy**2)는 그레디언트 크기 계산이다.
edge_strength=np.sqrt(grad_x**2+grad_y**2)
# [함수 사용] degree=np.degrees(radian)
# 라디안 각도를 도 단위로 바꿔 반환한다. 영상의 y축 방향은 별도로 고려한다.
# [함수 사용] angle=np.arctan2(y,x)
# x·y 부호로 사분면까지 고려한 각도를 라디안으로 반환한다. 영상 미분에서는 인자 순서가 dy,dx이다.
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
# [함수 사용] value=np.max(a, axis=None)
# axis 생략 시 전체 원소 중 최댓값, 지정 시 해당 축의 최댓값을 반환한다.
max_strength=np.max(edge_strength)
# [함수 사용] dst=np.uint8(values)
# 값을 8비트 부호 없는 정수로 변환한다. 안전한 범위 표시에는 먼저 clip을 고려하며 좌표값을 8비트로 제한하지 않도록 주의한다.
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
# [함수 사용] dst=cv.GaussianBlur(src, ksize, sigmaX, sigmaY=0)
# ksize=(너비,높이) 커널, sigmaX/Y=표준편차. sigmaY=0이면 sigmaX를 사용하며 흐린 영상을 반환한다.
# [PPT 인자 보충] 영상처리_2 27쪽: ksize의 각 값은 양의 홀수 또는 sigma로 자동 결정하는 0. sigmaX=0은 커널 크기로 추정, borderType은 가장자리 확장 방식이다.
blur=cv.GaussianBlur(gray, (3,3), sigmaX=1, sigmaY=1)

# ch4_2_Canny.py와 같은 두 가지 임계값 사용
# [함수 사용] edges=cv.Canny(src, threshold1, threshold2, apertureSize=3, L2gradient=False)
# 수업 예제의 src=uint8 명암 영상, 두 threshold=히스테리시스 기준. 단일 채널 에지 맵을 반환한다.
# [PPT 인자 보충] 에지 검출 28쪽: apertureSize는 Sobel 크기(3/5/7). L2gradient=True면 sqrt(dx²+dy²), False면 |dx|+|dy|로 강도를 계산한다. 반환은 0/255의 uint8 에지 맵이다.
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
