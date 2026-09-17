import cv2 as cv
import sys
import numpy as np
#%%
img=cv.imread('Balloon.png') ; type(img)
if img is None:
    sys.exit("No File exists.")
cv.imshow("Original", img) ; cv.waitKey()
#%%
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
mask1 = cv.inRange(hsv, (90, 80, 50), (130, 255, 255))
cv.imshow("mask1", mask1)
mask2 = cv.inRange(hsv, (90, 80, 50), (130, 255, 255))
cv.imshow("mask2", mask2)
mask = mask1 | mask2
cv.imshow("Mask", mask) ; cv.waitKey()
#%%
detected = cv.copyTo(img, mask)
#detected=cv.bitwise_and(img, img, mask)
cv.imshow("Result", detected); cv.waitKey()
cv.destroyAllWindows()