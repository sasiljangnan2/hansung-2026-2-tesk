# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 15:46:50 2026

@author: BigData
"""

import cv2 as cv
import numpy as np
import sys
#%% ROI
def region_of_interest(image):

    height = image.shape[0]
    width = image.shape[1]

    # 도로 부분을 삼각형으로 설정
    polygons = np.array([
        [(0, height),
         (width // 2, int(height * 0.4)),
         (width, height)]
    ])

    mask = np.zeros_like(image)

    cv.fillPoly(mask, polygons, 255)

    masked_image = cv.bitwise_and(image, mask)

    return masked_image

#%% Classify Left/Right lane and calculate slope and intercept

def average_slope_intercept(image, lines):
    left_lines = []
    right_lines = []

    if lines is None:
        return None

    for line in lines:
        x1, y1, x2, y2 = line
        # 기울기 계산
        if x2 == x1: #horizontal line
            continue
        slope = (y2 - y1) / (x2 - x1)
        # 너무 수평인 선 제거
        if abs(slope) < 0.5:
            continue
        intercept = y1 - slope * x1
        # 좌측 차선
        if slope < 0:
            left_lines.append((slope, intercept))
        # 우측 차선
        else:
            right_lines.append((slope, intercept))

    lane_lines = []
    if left_lines:
        left_avg = np.mean(left_lines, axis=0)
        lane_lines.append(make_coordinates(image, left_avg))
    if right_lines:
        right_avg = np.mean(right_lines, axis=0)
        lane_lines.append(make_coordinates(image, right_avg))

    return lane_lines

#%% extenr lane to the bottom of the image
def make_coordinates(image, line_parameters):

    slope, intercept = line_parameters

    y1 = image.shape[0]

    # 소실점 부근
    y2 = int(y1 * 0.6)

    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)

    return np.array([x1, y1, x2, y2])

#%% Draw lanes on image
def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line
            cv.line(line_image,(x1, y1), (x2, y2),(0, 0, 255), 8)

    return line_image
#%%
img=cv.imread("solidWhiteRight.jpg")
if img is None:
     sys.exit("No File exists.")
     

cv.imshow("Road",img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
blur = cv.GaussianBlur(gray,(5, 5), 0)
edges = cv.Canny(blur, 50, 150)
cv.imshow("Canny Edges", edges)

ROI_edges = region_of_interest(edges)
cv.imshow("Road_ROI", ROI_edges)

cv.waitKey()
#cv.destroyAllWindows()

#%%Hough Transform

lines = cv.HoughLinesP( ROI_edges, rho=2, theta=np.pi / 180, threshold=50, minLineLength=40, maxLineGap=100)
averaged_lines = average_slope_intercept(img, lines)
line_image = display_lines(img, averaged_lines)
cv.imshow("Line Image", line_image)

result = cv.addWeighted(img, 0.8, line_image, 1.0, 1)

cv.imshow("Lane Detection", result)

cv.waitKey()
cv.destroyAllWindows()

