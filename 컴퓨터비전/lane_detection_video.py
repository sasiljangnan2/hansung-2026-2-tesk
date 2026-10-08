# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 15:46:50 2026

@author: BigData
"""

import cv2 as cv
import numpy as np
import sys
from pathlib import Path
#%% ROI
def region_of_interest(image):

    height = image.shape[0]
    width = image.shape[1]

    # 도로 부분을 삼각형으로 설정
    polygons = np.array([
        [(int(width * 0.12), height - 1),
         (width // 2, int(height * 0.55)),
         (int(width * 0.95), height - 1)]
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

    # OpenCV 버전에 따른 배열 모양 차이를 (선분 수, 4)로 통일
    for line in lines.reshape(-1, 4):
        x1, y1, x2, y2 = line
        # 기울기 계산
        if x2 == x1: # 수직선은 기울기를 계산할 수 없으므로 제외
            continue
        slope = (y2 - y1) / (x2 - x1)
        # 너무 수평인 선 제거
        if abs(slope) < 0.3:
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

#%% Extend lane to the bottom of the image
def make_coordinates(image, line_parameters):

    slope, intercept = line_parameters

    y1 = image.shape[0] - 1

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
#%% 동영상 입력
# 실행 위치와 관계없이 이 파이썬 파일 옆의 동영상을 읽음
folder = Path(__file__).resolve().parent
cap = cv.VideoCapture(str(folder / 'second_HW_lane_detection.mp4'))
if not cap.isOpened():
    sys.exit('No video file exists.')

fps = cap.get(cv.CAP_PROP_FPS)
if fps <= 0:
    fps = 25

#%% 프레임마다 차선 검출
try:
    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        blur = cv.GaussianBlur(gray, (5, 5), 0)
        edges = cv.Canny(blur, 50, 150)
        ROI_edges = region_of_interest(edges)

        lines = cv.HoughLinesP(ROI_edges, rho=2, theta=np.pi / 180,
                              threshold=50, minLineLength=40, maxLineGap=100)
        averaged_lines = average_slope_intercept(img, lines)
        line_image = display_lines(img, averaged_lines)
        result = cv.addWeighted(img, 0.8, line_image, 1.0, 1)

        cv.imshow('Lane Detection', result)
        if cv.waitKey(max(1, int(1000 / fps))) == ord('q'):
            break
finally:
    cap.release()
    cv.destroyAllWindows()
