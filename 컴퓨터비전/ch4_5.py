# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 13:58:38 2023

@author: BigData
"""

import skimage
import numpy as np
import cv2 as cv
#%%
img = skimage.data.coffee()
print(img.shape)
cv.imshow("Coffee", cv.cvtColor(img, cv.COLOR_RGB2BGR))
cv.waitKey()

slic1=skimage.segmentation.slic(img, compactness=20, n_segments=600) #result of clustering
print(np.min(slic1)); print(np.max(slic1))

sp_img1=skimage.segmentation.mark_boundaries(img, slic1) #mode='outer'
print(np.min(sp_img1)); print(np.max(sp_img1)); print(sp_img1.shape)
sp_img1=np.uint8(sp_img1*255)

slic2=skimage.segmentation.slic(img, compactness=40, n_segments=600)

sp_img2=skimage.segmentation.mark_boundaries(img, slic2)
print(np.min(sp_img2)); print(np.max(sp_img2))
sp_img2=np.uint8(sp_img2*255)


cv.imshow("Super Pixels, Com=20", cv.cvtColor(sp_img1, cv.COLOR_RGB2BGR))
cv.imshow("Super Pixels, Com=40", cv.cvtColor(sp_img2, cv.COLOR_RGB2BGR))
cv.waitKey()

cv.destroyAllWindows()
