import cv2 as cv
import numpy as np

#%% 작은 삼각형 영상 준비 [5장 지역특징_1 PPT 27~32쪽, 프로그램 5-1]
# 파일 대신 10x10 실수 배열을 사용하므로 입력 영상 없이 바로 실행할 수 있다.
# 해리스는 물체 전체의 면적이 아니라, 주변 밝기가 여러 방향으로 변하는 점을 찾는다.
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

#%% 미분 커널과 가우시안 가중치 [PPT 14~18, 28, 33쪽]
# ux는 x방향, uy는 y방향 차이를 구한다. k는 3x1이므로 외적으로 3x3 가중치 g를 만든다.
# np.outer는 벡터를 펼쳐 모든 원소 쌍의 곱을 만든다. 이 경우 k @ k.T도 같은 결과이다.
ux=np.array([[-1,0,1]])
uy=np.array([[-1,0,1]]).transpose()
k=cv.getGaussianKernel(3,1)
g=np.outer(k,k.transpose())

#%% 구조 행렬의 세 성분 계산 [PPT 16~18, 29쪽]
# 부호 있는 dx, dy를 구하고 dxx=dx^2, dyy=dy^2, dyx=dy*dx를 만든다.
# 여기서 dxx/dyy는 '1차 미분의 제곱'이다. 헤시안의 2차 미분과 혼동하지 않는다.
# 가우시안으로 모은 gdxx, gdyy, gdyx가 구조 행렬 A의 두 대각 성분과 비대각 성분이다.
dy=cv.filter2D(img,cv.CV_32F,uy)
dx=cv.filter2D(img,cv.CV_32F,ux)
dyy=dy*dy
dxx=dx*dx
dyx=dy*dx
gdyy=cv.filter2D(dyy,cv.CV_32F,g)
gdxx=cv.filter2D(dxx,cv.CV_32F,g)
gdyx=cv.filter2D(dyx,cv.CV_32F,g)

#%% 해리스 응답 [PPT 19~20, 29쪽]
# C=det(A)-0.04*trace(A)^2. 고유값을 직접 계산하지 않고 코너 가능성을 구한다.
# 두 고유값이 모두 크면 코너, 한쪽만 크면 에지, 둘 다 작으면 평탄한 영역이다.
C=(gdyy*gdxx-gdyx*gdyx)-0.04*(gdyy+gdxx)*(gdyy+gdxx)

#%% 비최대 억제로 대표점 선택 [PPT 21, 24, 29쪽]
# 임계값만 적용하면 한 코너 주변의 여러 픽셀이 남으므로 3x3 안에서 혼자 가장 큰 점만 남긴다.
# 자신과의 비교는 False, 주변 8개는 모두 True여야 하므로 합이 8이다. 동률이면 모두 제외한다.
# 0.1은 이 입력에 맞춘 임계값이다. 특징점을 9로 표시하는 것은 검출 후의 시각화이다.
for j in range(1,C.shape[0]-1):
    for i in range(1,C.shape[1]-1):
        if C[j,i]>0.1 and np.count_nonzero(C[j,i]>C[j-1:j+2,i-1:i+2])==8:
            img[j,i]=9

#%% 중간 행렬과 결과 확인 [PPT 30~32쪽]
# 행렬 출력은 식의 각 항을 대조하기 위한 것이다. 확대와 밝기 변환은 표시만 바꾼다.
# PPT 표시식에 clip을 추가해 uint8 범위를 벗어날 때 값이 순환하지 않도록 했다.
np.set_printoptions(precision=2, suppress=True)
print('dy:\n',dy); print('dx:\n',dx)
print('dyy:\n',dyy); print('dxx:\n',dxx); print('dyx:\n',dyx)
print('gdyy:\n',gdyy); print('gdxx:\n',gdxx); print('gdyx:\n',gdyx)
print('C:\n',C); print('특징점 표시(9):\n',img)
popping=np.zeros([160,160],np.uint8)
for j in range(160):
    for i in range(160):
        popping[j,i]=np.uint8(np.clip((C[j//16,i//16]+0.06)*700,0,255))
points=cv.resize(np.uint8(img/9*255),(160,160),interpolation=cv.INTER_NEAREST)
cv.imshow('Harris response manual',popping)
cv.imshow('Harris representative points',points)
cv.waitKey()
cv.destroyAllWindows()
