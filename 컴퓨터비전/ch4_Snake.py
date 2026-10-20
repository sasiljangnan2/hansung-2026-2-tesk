# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 20:53:14 2025

@author: BigData
"""

# ==================================================================
# [선행 이론] 대화식 분할·능동 외곽선·에너지의 배경
# 근거: 4장 영역 분할_2 28~35쪽
# 읽는 순서: 선행 이론 → 관련 코드 앞의 서술형 요약 → 계산·출력 주석.
# PPT의 배경 이론과 현재 코드의 구현 범위를 구분하여 공부한다.
#
# 1. 영역 분할과 의미 분할
# 영역 분할은 밝기·색·텍스처가 비슷한 픽셀을 묶거나 다른 영역의 경계를 정하는 작업이다.
# 의미 분할은 픽셀에 사람·도로 같은 의미 범주까지 연결한다. 단순 색 군집의 번호가 물체 이름은 아니다.
# 이 강의의 고전적 방법은 주로 픽셀 특성으로 분할한 뒤 해석/인식하는 접근이다.
# 단순 배경에서는 임계값이 유리할 수 있고 복잡한 영상에서는 이웃·그래프·사용자 초기 정보 등을 고려한다.
#
# 2. 왜 초기 곡선을 주는가?
# 영상 전체를 여러 영역으로 나누는 대신 관심 물체 하나의 경계를 찾고 싶을 수 있다.
# 능동 외곽선은 대략적인 곡선에서 시작해 더 좋은 경계로 이동하는 최적화 접근이다.
# g(s)=(y(s),x(s))는 곡선 위 위치를 나타낸다. s는 곡선의 매개변수이고 반복 시간과 다르다.
# 디지털에서는 여러 좌표를 순서대로 저장한다. init/snake는 영상이 아니라 (점 수,2)의 좌표 배열이다.
#
# 3. 총 에너지의 구성
# 강의 E=sum(E_image+E_internal+E_constraint)는 영상 반응·곡선 내부 성질·사전 제약을 함께 고려한다.
# 영상 항 E_image=-||gradient f||**2이면 강한 경계에 갈수록 더 작은 에너지가 된다.
# 강한 그레디언트를 찾으면서 에너지를 최소화하므로 음의 부호가 들어간다는 점을 이해한다.
# 내부 항은 (alpha*||g'(s)||**2+beta*||g''(s)||**2)/2로 길이와 굽힘을 제한한다.
# alpha는 장력/수축, beta는 굽힘의 매끄러움에 관계한다. 영상 반응만 따를 때의 거친 곡선을 억제한다.
# 제약 항은 사용자가 지정한 모양 같은 사전 정보를 뜻한다. 현재 함수 호출에 별도 사용자 제약 항을 구현하지 않았다.
#
# 4. 수렴과 함수 인자의 한계
# 반복해서 에너지를 줄여도 초기 곡선이 멀거나 다른 에지가 강하면 원하는 경계에 가지 못할 수 있다.
# 지역 최소점·잡음·초기화에 민감하므로 초기 위치와 평활화가 중요하다.
# 현재 Gaussian sigma=3은 잡음 완화, active_contour의 gamma는 시간 적분 간격이다.
# 다른 강의 식에서 외부 에너지 가중치에 쓰는 gamma와 함수 인자의 뜻을 섞지 않는다.
# w_line/w_edge는 밝기 자체와 에지에 끌리는 비중이며 현재 기본 0/1을 사용한다.
# ==================================================================


from skimage.segmentation import active_contour
from skimage.filters import gaussian
from skimage import data
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt

# [서술형] Snake(능동 외곽선) 분석
# 관련 PPT: 영역 분할_2 29~35쪽
# 특징: 초기 곡선을 움직이며 영상 경계에 끌리는 항과 곡선의 길이·굽힘 항의 에너지를 줄인다.
# 장점: 곡선의 매끄러움을 제어하면서 관심 물체의 연속적인 경계를 표현할 수 있다.
# 단점: 초기 곡선과 잡음·가짜 경계에 민감하고 원하는 경계 대신 지역 최소점에 머물 수 있다.
# 활용: 대략적인 위치를 아는 물체의 윤곽 정밀화, 이전 프레임 경계를 활용하는 추적의 발상.
# 한 문장 답안: Snake는 내부 매끄러움과 영상 경계를 함께 고려하지만 초기 위치에 따른 수렴 한계가 있다.
# [계산·코드 연결]
# 강의 E=sum(E_image+E_internal), E_image=-||gradient f||**2.
# E_internal=(alpha*||g'(s)||**2+beta*||g''(s)||**2)/2. 길이와 굽힘을 함께 제한한다.
# alpha=장력, beta=굽힘, active_contour의 gamma=시간 간격. 다른 식의 외부 가중치 gamma와 구분.
# 공식 astronaut=(512,512,3), gray와 Gaussian 후=(512,512). 현재 scikit-image 미설치로 전체 미실행.
# [함수 사용] gray=rgb2gray(rgb)
# RGB 영상에서 명암 실수 영상을 반환한다. OpenCV의 BGR 입력 관례와 채널 순서가 다르다.
# [함수 사용] img=data.astronaut()
# 표준 우주비행사 RGB uint8 영상 (512,512,3)을 반환한다.
img = rgb2gray(data.astronaut())
# [함수 사용] dst=gaussian(img, sigma=3)
# sigma=가우시안 표준편차. skimage.filters의 평활화 결과이며 정수 입력의 값 범위 변환 설정도 확인한다.
img = gaussian(img, sigma=3)
print(img.shape)
# 초기 윤곽선 (원형)
# 매개변수 곡선으로 초기 경계 표현 [영역 분할_2 PPT 29~35쪽]
# init은 (행,열)=(y,x) 좌표 400개이다. 그래프에는 x가 먼저이므로 [:,1],[:,0] 순서로 그린다.
# 초기 위치가 경계와 멀면 다른 경계에 수렴할 수 있다.
# [함수 사용] a=np.linspace(start, stop, num)
# 기본은 양 끝을 포함해 num개의 등간격 표본을 만든다. endpoint=False이면 끝값은 제외한다.
s = np.linspace(0, 2 * np.pi, 400)
# [함수 사용] dst=np.sin(x)
# 라디안 단위 x의 원소별 사인 값을 반환한다. 초기 원형 곡선의 y 좌표 계산 등에 사용한다.
r = 100 + 100 * np.sin(s)
# [함수 사용] dst=np.cos(x)
# 라디안 단위 x의 원소별 코사인 값을 반환한다. 초기 원형 곡선의 x 좌표 계산 등에 사용한다.
c = 220 + 100 * np.cos(s)
# [배열 문제] r.shape=c.shape=(400,), np.array([r,c])는 (2,400), .T 후 init은 (400,2)이다.
# s의 양 끝은 0과 2*pi여서 처음/마지막 좌표는 수치 오차를 제외하면 같은 위치이다.
# active_contour 출력 snake도 (400,2)이며 각 행은 (y,x)이다. 좌표 목록과 영상 응답 맵을 구분한다.
# [함수 사용] a=np.array(values, dtype=None)
# values=리스트 등의 입력, dtype=원소 자료형(생략하면 추론). 지정한 값을 가진 ndarray를 만든다.
init = np.array([r, c]).T

# [계산·코드 연결] 앞에서 Gaussian sigma=3으로 잡음을 완화하고 원형 초기 곡선 (400,2)을 준비한다.
# w_line=0,w_edge=1 기본값은 밝기 자체보다 에지에 끌리도록 한다. 결과 snake도 (400,2)의 (y,x) 좌표이다.
# alpha가 커지면 수축 경향, beta가 커지면 더 매끄러운 경향. 값이 클수록 무조건 정확한 것은 아니다.
# [함수 사용] snake=active_contour(img, init, alpha=0.01, beta=0.1, gamma=0.01)
# init=(점 수,2)의 (행,열) 곡선, alpha=장력, beta=굽힘, gamma=시간 간격. 갱신한 곡선 좌표를 반환한다.
# [PPT 인자 보충] 영역 분할 35쪽: w_line은 밝기 유인(음수면 어두운 곳), w_edge는 에지 유인, max_px_move는 1회 최대 이동량. max_num_iter/convergence는 종료 조건, boundary_condition은 periodic/fixed/free 등이다.
# 입력 snake는 (행,열) 좌표의 (K,2) 배열이고 결과도 K개 점이다. periodic은 끝점 중복 없이 준비하는 것이 문서 조건이다(기존 실행문은 유지).
snake = active_contour(img, init, alpha=0.015, beta=10, gamma=0.001)

# 시각화
# [함수 사용] plt.imshow(X,cmap=None,vmin=None,vmax=None)
# 2D 명암에는 cmap="gray", 컬러는 RGB를 사용한다. vmin/vmax는 명암 표시 범위이며 OpenCV BGR은 변환 후 표시한다.
plt.imshow(img, cmap='gray')
# [함수 사용] plt.show()
# 준비된 그래프 창을 표시한다. 원본 배열을 저장하거나 픽셀 값을 바꾸는 함수가 아니다.
plt.show()
plt.imshow(img, cmap='gray')
# [함수 사용] plt.plot(x,y,fmt,color=...,linewidth=...) 또는 plt.plot(y)
# 좌표·곡선을 그린다. x 생략 시 0부터의 인덱스. fmt="--r"는 빨간 파선, lw는 linewidth의 별칭이다.
plt.plot(init[:, 1], init[:, 0], '--r', lw=3)
plt.show()
plt.imshow(img, cmap='gray')
plt.plot(init[:, 1], init[:, 0], '--r', lw=3)
plt.plot(snake[:, 1], snake[:, 0], '-b', lw=3)
plt.show()

# 이 예제에서는 w_line=0, w_edge=1이 기본값으로 설정되어 있어 경계(edge)를 중심으로 윤곽선을 조정합니다1