import cv2 as cv
import sys
import numpy as np
#%%
img=cv.imread('BALLOON.bmp') ; type(img)
if img is None:
    sys.exit("No File exists.")
cv.imshow("Original", img) ; cv.waitKey()
#%%
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
mask = cv.inRange(hsv, (90, 60, 50), (130, 255, 255))
dst = np.full_like(img, 255) 
#%%
detected = cv.copyTo(img, mask, dst)  
#detected=cv.bitwise_and(img, img, mask)
cv.imshow("Result", detected); cv.waitKey()
cv.destroyAllWindows()