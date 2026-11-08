# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 13:59:25 2024

@author: BigData
"""

import cv2 as cv
import numpy as np
#%%
img1=cv.imread('model3.jpg')
gray1=cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
img2=cv.imread('scene.jpg')
gray2=cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
cv.imshow("IMG!", img1);cv.imshow("IMG2", img2)
cv.waitKey(); cv.destroyAllWindows()
#%% detect and compute Keypoints
sift=cv.SIFT_create()
kp1, des1 =sift.detectAndCompute(gray1, None)
kp2, des2 =sift.detectAndCompute(gray2, None)
print("특징점 개수", len(kp1), len(kp2))
#%% Keypoints of img1
#gray1 = cv.drawKeypoints(gray1, kp1, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#cv.imshow("SIFT", gray1)
img = cv.drawKeypoints(img1, kp1, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv.imshow("SIFT", img)
cv.waitKey(); cv.destroyAllWindows()
#%% Matches
flann_matcher=cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
knn_match=flann_matcher.knnMatch(des1, des2, k=2)

T=0.7
good_match=[]
for nearest1, nearest2 in knn_match:
    if(nearest1.distance/nearest2.distance) < T:
       good_match.append(nearest1)
#%%
print(len(good_match)); type(good_match)
print(good_match[0].queryIdx, good_match[0].trainIdx,good_match[0].distance, good_match[0].imgIdx)
#%% DrawMatches
img_mat=np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], 3), dtype=np.uint8)
cv.drawMatches(img1, kp1, img2, kp2, good_match, img_mat, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv.imshow("Good Matches", img_mat)
cv.waitKey(); cv.destroyAllWindows()
#%% Instance recognition
points1=np.float32([kp1[gm.queryIdx].pt for gm in good_match])
points2=np.float32([kp2[gm.trainIdx].pt for gm in good_match])
H, mask=cv.findHomography(points1, points2, cv.RANSAC)

h1, w1 = img1.shape[0], img1.shape[1]
h2, w2 = img2.shape[0], img2.shape[1]

box1=np.float32([[0,0], [0, h1-1], [w1-1, h1-1], [w1-1,0]]).reshape(4,1,2)
box2=cv.perspectiveTransform(box1,H)

img2=cv.polylines(img2, [np.int32(box2)], True, (0,255,0),8)

img_match=np.empty((max(h1, h2), w1+w2, 3), dtype=np.uint8)
cv.drawMatches(img1, kp1, img2, kp2, good_match, img_match, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv.imshow("matches and Homography", img_match)
cv.waitKey(); cv.destroyAllWindows()