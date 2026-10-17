# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 20:53:14 2025

@author: BigData
"""

from skimage.segmentation import active_contour
from skimage.filters import gaussian
from skimage import data
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt

img = rgb2gray(data.astronaut())
img = gaussian(img, sigma=3)
print(img.shape)
# 초기 윤곽선 (원형)
# 매개변수 곡선으로 초기 경계 표현 [영역 분할_2 PPT 29~35쪽]
# init은 (행,열)=(y,x) 좌표 400개이다. 그래프에는 x가 먼저이므로 [:,1],[:,0] 순서로 그린다.
# 초기 위치가 경계와 멀면 다른 경계에 수렴할 수 있다.
s = np.linspace(0, 2 * np.pi, 400)
r = 100 + 100 * np.sin(s)
c = 220 + 100 * np.cos(s)
init = np.array([r, c]).T

# Snake 알고리즘 적용
# 에너지 최소화로 곡선 이동 [영역 분할_2 PPT 30~35쪽]
# alpha는 길이/장력, beta는 굽힘의 매끄러움, 이 함수의 gamma는 시간 적분 간격이다.
# 영상 경계로 끌어당기는 항과 곡선 내부 제약을 함께 사용한다. 앞의 Gaussian은 잡음 완화용이다.
snake = active_contour(img, init, alpha=0.015, beta=10, gamma=0.001)

# 시각화
plt.imshow(img, cmap='gray')
plt.show()
plt.imshow(img, cmap='gray')
plt.plot(init[:, 1], init[:, 0], '--r', lw=3)
plt.show()
plt.imshow(img, cmap='gray')
plt.plot(init[:, 1], init[:, 0], '--r', lw=3)
plt.plot(snake[:, 1], snake[:, 0], '-b', lw=3)
plt.show()

# 이 예제에서는 w_line=0, w_edge=1이 기본값으로 설정되어 있어 경계(edge)를 중심으로 윤곽선을 조정합니다1