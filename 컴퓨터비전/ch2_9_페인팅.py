import cv2 as cv 
import sys

# [함수 사용: 입출력·창]
# img=cv.imread(path, flags=cv.IMREAD_COLOR): 파일→배열. 기본 BGR, 읽기 실패 시 None인지 확인한다.
# cv.imshow(winname,img): 이름이 winname인 창에 배열을 표시한다. 저장 함수는 아니다.
# cv.namedWindow(winname): 마우스 콜백 등을 등록할 이름 있는 창을 만든다.
# key=cv.waitKey(delay=0): GUI 이벤트·키 처리. 0은 계속 대기, 양수는 ms, 입력 없으면 보통 -1이다.
# cv.destroyAllWindows(): OpenCV 창을 닫는다. 캡처 자원 해제는 cap.release()이다.
img=cv.imread('soccer.jpg')

if img is None:
    sys.exit('파일을 찾을 수 없습니다.')

BrushSiz=5					# 붓의 크기
LColor,RColor=(255,0,0),(0,0,255)		# 파란색과 빨간색

def painting(event,x,y,flags,param):
    if event==cv.EVENT_LBUTTONDOWN:   
        # [함수 사용] cv.circle(img, center, radius, color, thickness=1)
        # center=(x,y), radius=반지름, color=BGR 색. img에 직접 그리며 thickness=-1이면 원 내부를 채운다.
        # [PPT 인자 보충] OpenCV 기본 37쪽: lineType=LINE_4/LINE_8/LINE_AA는 선 연결·안티앨리어싱 방식. shift는 좌표의 소수 비트 수이며 thickness<0이면 채운다.
        cv.circle(img,(x,y),BrushSiz,LColor,-1)# 마우스 왼쪽 버튼 클릭하면 파란색
    elif event==cv.EVENT_RBUTTONDOWN: 
        cv.circle(img,(x,y),BrushSiz,RColor,-1)# 마우스 오른쪽 버튼 클릭하면 빨간색
    elif event==cv.EVENT_MOUSEMOVE and flags==cv.EVENT_FLAG_LBUTTON:
        cv.circle(img,(x,y),BrushSiz,LColor,-1)# 왼쪽 버튼 클릭하고 이동하면 파란색
    elif event==cv.EVENT_MOUSEMOVE and flags==cv.EVENT_FLAG_RBUTTON:
        cv.circle(img,(x,y),BrushSiz,RColor,-1)# 오른쪽 버튼 클릭하고 이동하면 빨간색

    cv.imshow('Painting',img)		# 수정된 영상을 다시 그림

cv.namedWindow('Painting')
cv.imshow('Painting',img)

# [함수 사용] cv.setMouseCallback(winname, onMouse, param=None)
# winname=창 이름, onMouse=콜백 함수 자체. 이벤트가 나면 onMouse(event,x,y,flags,param)를 호출하도록 등록한다.
cv.setMouseCallback('Painting',painting)

while(True):
    if cv.waitKey(1)==ord('q'):
        cv.destroyAllWindows()      
        break