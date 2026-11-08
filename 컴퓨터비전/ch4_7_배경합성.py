# -*- coding: utf-8 -*-
from pathlib import Path
import cv2 as cv
import numpy as np

#%% 영상 읽기
HERE = Path(__file__).resolve().parent
INPUT_FILE = HERE / 'background_images/김준휘.jpg'
BACKGROUND_FILE = HERE / 'background_images/background.png'
OUTPUT_DIR = HERE / 'background_results'
original = cv.imdecode(np.frombuffer(INPUT_FILE.read_bytes(), np.uint8), cv.IMREAD_COLOR)
background = cv.imdecode(np.frombuffer(BACKGROUND_FILE.read_bytes(), np.uint8), cv.IMREAD_COLOR)
if original is None or background is None:
    raise ValueError('입력 사진과 배경 사진을 확인하세요.')

scale = min(1.0, 736 / max(original.shape[:2]))
img = cv.resize(original, (round(original.shape[1]*scale), round(original.shape[0]*scale)))
height, width = img.shape[:2]
img_show = img.copy()

#%% 마스크 설정
mask = np.full((height, width), cv.GC_PR_FGD, np.uint8)
# 흰색은 배경 후보로 지정
white = cv.inRange(img, (235,235,235), (255,255,255))
mask[white != 0] = cv.GC_PR_BGD
mask[:max(1, int(height*0.03)), :] = cv.GC_BGD
# 셔츠가 배경으로 빠지지 않도록 전경 지정
cv.rectangle(mask, (int(width*.35), int(height*.62)),
             (int(width*.66), height-1), cv.GC_FGD, -1)
for a, b in [((.50,.30),(.50,.62)), ((.25,.86),(.75,.86))]:
    cv.line(mask, (int(width*a[0]), int(height*a[1])),
            (int(width*b[0]), int(height*b[1])), cv.GC_FGD, 6)

#%% 마우스 입력
def painting(event, x, y, flags, param):
    left = event == cv.EVENT_LBUTTONDOWN or (event == cv.EVENT_MOUSEMOVE and flags & cv.EVENT_FLAG_LBUTTON)
    right = event == cv.EVENT_RBUTTONDOWN or (event == cv.EVENT_MOUSEMOVE and flags & cv.EVENT_FLAG_RBUTTON)
    if left or right:
        label, color = (cv.GC_FGD, (255,0,0)) if left else (cv.GC_BGD, (0,0,255))
        cv.circle(mask, (x,y), 7, label, -1)
        cv.circle(img_show, (x,y), 7, color, -1)
        cv.imshow('Painting', img_show)

cv.namedWindow('Painting')
cv.setMouseCallback('Painting', painting)
cv.imshow('Painting', img_show)
print('왼쪽=남기기 / 오른쪽=지우기 / G=다시 분할·저장 / Q=종료')
OUTPUT_DIR.mkdir(exist_ok=True)
key = ord('g')

#%% 분할 및 합성
while True:
    if key in (ord('g'), ord('G')):
        if not np.any((mask == 1) | (mask == 3)) or not np.any((mask == 0) | (mask == 2)):
            print('전경과 배경을 모두 지정하세요.')
        else:
            bg_model = np.zeros((1,65), np.float64)
            fg_model = np.zeros((1,65), np.float64)
            cv.setRNGSeed(0)
            cv.grabCut(img, mask, None, bg_model, fg_model, 5, cv.GC_INIT_WITH_MASK)
            binary = np.where((mask == 0) | (mask == 2), 0, 255).astype('uint8')
            x, y, w, h = cv.boundingRect(binary)
            if w == 0 or h == 0:
                print('남길 부분을 왼쪽 버튼으로 칠하세요.')
                key = 0
                continue

            bh, bw = background.shape[:2]
            ratio = min(bh*0.90/h, bw*0.70/w)
            size = (max(1, int(w*ratio)), max(1, int(h*ratio)))
            person = cv.resize(img[y:y+h, x:x+w], size,
                               interpolation=cv.INTER_AREA if ratio < 1 else cv.INTER_LINEAR)
            selected = cv.resize(binary[y:y+h, x:x+w], size, interpolation=cv.INTER_NEAREST)
            left, top = (bw-size[0])//2, bh-size[1]
            result = background.copy()
            cv.copyTo(person, selected, result[top:top+size[1], left:left+size[0]])

            for name, image in [('01_background.png', background), ('02_original.png', original),
                                ('03_result.png', result), ('04_mask.png', binary)]:
                (OUTPUT_DIR/name).write_bytes(cv.imencode('.png', image)[1].tobytes())
            cv.imshow('Mask', binary)
            cv.imshow('Result', result)

    key = cv.waitKey(20) & 0xff
    if key in (ord('q'), ord('Q'), 27) or cv.getWindowProperty('Painting', cv.WND_PROP_VISIBLE) < 1:
        break
cv.destroyAllWindows()
