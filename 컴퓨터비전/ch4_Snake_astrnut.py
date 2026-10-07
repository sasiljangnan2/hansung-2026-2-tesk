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
s = np.linspace(0, 2 * np.pi, 400)
r = 100 + 100 * np.sin(s)
c = 220 + 100 * np.cos(s)
init = np.array([r, c]).T

# Snake 알고리즘 적용
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