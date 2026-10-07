# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 15:51:43 2023

@author: BigData
"""

import skimage
import numpy as np
import cv2 as cv
import time
#%%
img = skimage.data.coffee()
cv.imshow("Coffee", cv.cvtColor(img, cv.COLOR_RGB2BGR))
cv.waitKey()

start=time.time()
slic=skimage.segmentation.slic(img, compactness=20, n_segments=600, start_label=1)
print(slic.shape)

g=skimage.graph.rag_mean_color(img, slic, mode='similarity')
ncut=skimage.graph.cut_normalized(slic, g)
print(img.shape, '분할 소요 시간', time.time()-start)
print(np.min(ncut)); print(np.max(ncut))
print(ncut);print(ncut.shape);
print(np.unique(ncut))

marking=skimage.segmentation.mark_boundaries(img, ncut)
ncut_img=np.uint8(marking*255)

cv.imshow("Normalized Cut", cv.cvtColor(ncut_img, cv.COLOR_RGB2BGR))
cv.waitKey()

cv.destroyAllWindows()
