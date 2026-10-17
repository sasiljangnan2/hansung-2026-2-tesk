# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 15:46:50 2026

@author: BigData
"""

# ==================================================================
# [선행 이론] 정지 영상 차선 검출의 가정·단계·한계
# 근거: 4장 에지 검출_1 37~52쪽
# 읽는 순서: 선행 이론 → 관련 코드 앞의 서술형 요약 → 계산·출력 주석.
# PPT의 배경 이론과 현재 코드의 구현 범위를 구분하여 공부한다.
#
# 1. 점을 연결하지 않고 같은 모델을 지지하는지 모으는 발상
# 경계점이 가능한 직선 후보에 투표하고 많은 표가 모이는 후보를 선으로 선택한다.
# 일부 경계가 끊겨 있어도 여러 점이 같은 모델을 지지하면 찾을 수 있다. 무늬·잡음도 후보에 투표할 수 있다.
# y=a*x+b를 쓰면 한 영상 점 (x,y)이 b=-x*a+y라는 매개변수 공간의 직선이 된다.
# 같은 영상 직선 위 점들은 공통 a,b 위치에 모인다. 하지만 수직선은 기울기 a가 무한대라 표현이 어렵다.
# 이를 rho=x*cos(theta)+y*sin(theta)로 표현한다. theta는 직선의 접선이 아니라 법선 방향이다.
# 누적 공간을 유한 간격으로 나누므로 rho/theta 해상도와 투표 기준이 결과·계산량에 영향을 준다.
#
# 2. 허프 함수의 결과와 모델을 구분
# HoughLines는 (rho,theta)의 무한 직선, HoughLinesP는 (x1,y1,x2,y2)의 선분을 반환한다.
# minLineLength는 최소 선분 길이, maxLineGap은 한 선분으로 연결할 수 있는 틈이다.
# 원의 일반 매개변수는 (중심a,중심b,반지름r)이고 (x-a)**2+(y-b)**2=r**2를 만족한다.
# 직선의 두 매개변수와 원의 세 매개변수 공간을 혼동하지 않는다.
#
# 3. ROI·기울기 분류·대표선의 가정 [PPT 51~52쪽]
# 도로는 영상의 아래쪽 특정 구역에 있다는 가정으로 삼각형 ROI 밖의 에지를 제외한다.
# 직선 선분의 기울기 a=(y2-y1)/(x2-x1), 절편 b=y1-a*x1을 구해 좌/우로 나눈다.
# 영상 y축이 아래로 증가하는 현재 구도에서 왼쪽은 음수, 오른쪽은 양수 기울기를 갖는다.
# 수직선은 분모 0이라 제외하고 너무 수평인 선분도 차선 후보에서 제외한다.
# 각 그룹 a,b의 평균으로 대표선을 만들고 x=(y-b)/a로 정한 위·아래 높이의 끝점을 구한다.
# 이 모델은 구도와 직선 근사에 의존하므로 곡선·가림·그림자·잘못된 ROI에서 오차가 생길 수 있다.
#
# 4. 전체 처리 흐름과 표시
# 프레임→Gray→Gaussian→Canny→ROI→HoughLinesP→좌우 분류/평균→연장→원본 합성이다.
# 앞 단계의 오검출은 뒤 평균 직선에도 영향을 준다. 허프 검출만으로 차선의 의미를 모두 이해한 것은 아니다.
# 선의 BGR (0,0,255)은 빨강. addWeighted는 원본/선 영상의 가중합을 포화 변환한 표시 결과이다.
# 시간 변화나 모든 프레임의 정확도를 보장하는 추적 모델을 별도로 구현한 것은 아니다.
#
# 5. 현재 정지 영상 파일의 실행 주의점
# solidWhiteRight.jpg가 없으며 허프 반환 모양을 그대로 언패킹하는 부분도 확인이 필요하다.
# 동영상 예제는 reshape(-1,4)로 통일한다. 이 정리는 기존 실행문을 수정하지 않고 원리와 범위를 설명한다.
# ==================================================================


import cv2 as cv
import numpy as np
import sys
#%% ROI
# [서술형] 허프 직선과 기울기 기반 차선 분석
# 관련 PPT: 에지 검출_1 37~46,51~52쪽
# 특징: 도로 ROI 에지에서 직선 선분을 찾고 기울기별 좌/우 평균 직선으로 차선을 표현한다.
# 장점: 끊어진 직선 경계의 투표를 모으고 ROI로 도로 밖의 많은 불필요한 후보를 줄일 수 있다.
# 단점: 곡선 차선·가림·그림자·카메라 구도 변화에 취약하고 ROI/기울기/허프 인자 설정에 의존한다.
# 활용: 이 강의처럼 도로 전방을 보며 주행 차선을 직선으로 근사할 수 있는 영상.
# 한 문장 답안: 허프 기반 차선 검출은 직선 투표와 ROI로 단순화하지만 곡선·조명·구도 변화에 한계가 있다.
# [계산·코드 연결]
# rho=x*cos(theta)+y*sin(theta), theta는 법선. HoughLines는 rho/theta, P는 두 끝점이다.
# 반환이 None일 수 있고 배열 모양도 확인한다. ROI는 같은 크기 0/255 마스크와 AND한다.
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

    # 원래 실행문 유지: HoughLinesP가 (N,1,4)를 반환하면 이 반복문의 언패킹과 맞지 않는다.
    # 동영상 파일에는 reshape(-1,4)로 통일한 예가 있으므로 반환 배열 모양을 비교한다.
    for line in lines:
        x1, y1, x2, y2 = line
        # 기울기 계산
        if x2 == x1: # 수직선: x 변화가 0이므로 기울기 계산 제외
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
# [작은 계산 문제] y=a*x+b에서 x=(y-b)/a. a=-1,b=800,y=500이면 x=300이다.
# 영상의 y축은 아래가 양수라 이 구도의 왼쪽 선분은 음의 기울기이다.
# 평균 a,b를 사용하므로 마지막 대표 차선은 검출 선분 하나의 끝점을 그대로 그린 것이 아니다.
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

