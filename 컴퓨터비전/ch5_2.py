# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 21:29:04 2023

@author: BigData
"""
import numpy as np
import cv2 as cv

img=cv.imread("mot_color70.jpg")
gray=cv.cvtColor(img, cv.COLOR_BGR2GRAY)
print("Gray shape: ", gray.shape)

sift=cv.SIFT_create()
#sift=cv.SIFT_create(500)
kp, des = sift.detectAndCompute(gray, None)
print("Keypoints:\n", cv.KeyPoint_convert(kp))
print("Number of Keypointd: ", len(kp))

gray = cv.drawKeypoints(gray, kp, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv.imshow("SIFT", gray)

cv.waitKey()
cv.destroyAllWindows()

