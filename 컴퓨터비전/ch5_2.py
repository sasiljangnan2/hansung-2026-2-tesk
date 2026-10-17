import cv2 as cv
import numpy as np

#%% OpenCV 함수로 해리스 응답 계산 [5장 지역특징_1 PPT 23, 34쪽]
# ch5_1.py와 같은 10x10 삼각형이다. float32는 음수/소수 응답을 다루기 위한 자료형이다.
img=np.array([[0,0,0,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0,0,0],
              [0,0,0,1,0,0,0,0,0,0],
              [0,0,0,1,1,0,0,0,0,0],
              [0,0,0,1,1,1,0,0,0,0],
              [0,0,0,1,1,1,1,0,0,0],
              [0,0,0,1,1,1,1,1,0,0],
              [0,0,0,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0,0,0]], dtype=np.float32)

# blockSize=3은 미분 곱을 모을 이웃 범위, ksize=3은 Sobel 미분 커널 크기이다.
# k=0.04는 det(A)-k*trace(A)^2의 계수이다. 반환값은 좌표 목록이 아닌 응답 맵이다.
# PPT는 blockSize를 가우시안 크기로 설명하지만 OpenCV 함수 자체의 가우시안 sigma는 아니다.
# 직접 구현과는 미분 커널/이웃 집계/스케일이 달라 응답 숫자가 같다고 가정하면 안 된다.
dst=cv.cornerHarris(img,3,3,0.04)

#%% 응답 맵 표시 [PPT 34쪽]
# 상수는 이 작은 영상의 음수/양수 응답을 화면 밝기로 옮기는 표시용 값이다.
# uint8 변환 전에 clip하여 범위 밖 값의 순환을 막았다. 이 표시는 특징점 검출/NMS가 아니다.
np.set_printoptions(precision=4, suppress=True)
print('OpenCV Harris response:\n',dst)
img_H=np.zeros([160,160],np.uint8)
for j in range(160):
    for i in range(160):
        img_H[j,i]=np.uint8(np.clip((dst[j//16,i//16]+0.0178)*2000,0,255))
cv.imshow('Img_H_Harris Display',img_H)
cv.waitKey()
cv.destroyAllWindows()
