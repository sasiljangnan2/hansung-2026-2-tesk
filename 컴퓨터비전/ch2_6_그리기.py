import cv2 as cv
from cv_assets import read_image
import sys

# [함수 사용: 입출력·창]
# img=cv.imread(path, flags=cv.IMREAD_COLOR): 파일→배열. 기본 BGR, 읽기 실패 시 None인지 확인한다.
# cv.imshow(winname,img): 이름이 winname인 창에 배열을 표시한다. 저장 함수는 아니다.
# key=cv.waitKey(delay=0): GUI 이벤트·키 처리. 0은 계속 대기, 양수는 ms, 입력 없으면 보통 -1이다.
# cv.destroyAllWindows(): OpenCV 창을 닫는다. 캡처 자원 해제는 cap.release()이다.
img=read_image('girl_laughing.jpg') 
  
if img is None:
    sys.exit('파일을 찾을 수 없습니다.')

# [함수 사용] cv.rectangle(img, pt1, pt2, color, thickness=1)
# pt1·pt2=(x,y) 꼭짓점, color=BGR 색. img에 그리며 -1은 채우기이다. (x,y,w,h) 사각형 인자 형식도 있다.
# [PPT 인자 보충] OpenCV 기본 30쪽: lineType은 선 연결·안티앨리어싱, shift는 좌표의 소수 비트 수. thickness<0이면 채우며 img를 직접 수정한다.
cv.rectangle(img,(830,30),(1000,200),(0,0,255),2)	# 직사각형 그리기
# [함수 사용] cv.putText(img, text, org, fontFace, fontScale, color, thickness=1)
# text=문자열, org=(x,y) 기준선 시작점, fontFace=글꼴, fontScale=배율. 지정한 BGR 색으로 img에 쓴다.
# [PPT 인자 보충] OpenCV 기본 28쪽: org는 글자의 왼쪽 아래 기준점. lineType은 선 모양, bottomLeftOrigin=False는 영상 원점을 왼쪽 위로 해석한다.
cv.putText(img,'laugh',(830,24),cv.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)	# 글씨 쓰기

cv.imshow('Draw',img)

cv.waitKey()
cv.destroyAllWindows()