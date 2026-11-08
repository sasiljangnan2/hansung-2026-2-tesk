import cv2 as cv
from cv_assets import read_image
import numpy as np

#%% 영상 읽기
img=read_image('soccer1.JPG')
model=read_image('soccer_ball.JPG')
gray=cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#%% 모델 크기를 바꿔가며 축구공 찾기
# 모델 속 공이 더 크므로 0.5~1배로 비교한다.
best_score=-1
for scale in np.arange(0.5, 1.01, 0.05):
    resized=cv.resize(model, (0,0), fx=scale, fy=scale)
    template=cv.cvtColor(resized, cv.COLOR_BGR2GRAY)

    # matchTemplate: 위치별 유사도, minMaxLoc: 가장 높은 점수와 위치
    response=cv.matchTemplate(gray, template, cv.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc=cv.minMaxLoc(response)
    if max_val>best_score:
        best_score=max_val
        x,y=max_loc
        h,w=template.shape

#%% 찾은 사각형의 중심과 반지름으로 원 그리기
center=(x+w//2, y+h//2)
radius=min(w,h)//2
result=img.copy()
cv.circle(result, center, radius, (0,0,255), 2)

print('중심 좌표:', center)
print('반지름:', radius)
print('유사도:', round(best_score, 4))

#%% 결과 표시
cv.imshow('Model', model)
cv.imshow('Soccer detection', result)
cv.waitKey()
cv.destroyAllWindows()
