
#%% 개념 요약 — 정의·장점·단점, 함수 사용법은 해당 코드 옆 참고

#%% [개념] 최근접·양선형·양3차 보간 비교
# 정의: 영상 크기나 좌표를 바꿀 때 격자 사이의 픽셀 값을 주변의 기존 픽셀로 추정하는 방법들이다.
# 장점: 최근접은 범주 보존과 단순 계산, 양선형은 부드러운 확대, 양3차는 더 넓은 이웃 활용에 유리하다.
# 단점: 최근접은 계단 현상, 양선형은 흐림, 양3차는 계산량과 오버슈트·링잉 가능성이 있다.
# PPT: 영상처리_2 38~42쪽

#region 핵심 이론·함수 참고
# [핵심 이론] 기하 변환·동차 좌표·역방향 사상·보간의 배경
# 근거: 3장 영상처리_2 32~42쪽
# 1. 기하 연산은 좌표를 바꾸고 밝기를 가져오는 작업
# 평행 이동·회전·크기 변화는 목적 좌표에 대응하는 원본 위치를 정한다.
# (x,y,w)에서 w!=0이면 (x/w,y/w)로 해석한다. 세 성분에 같은 0이 아닌 배수를 곱해도 같은 점이다.
# 열벡터 기준 p'=H*p. T=[[1,0,tx],[0,1,ty],[0,0,1]], S=[[sx,0,0],[0,sy,0],[0,0,1]].
# PPT의 영상 좌표(y 아래가 +)에서 화면상 반시계 회전 R=[[cos,sin,0],[-sin,cos,0],[0,0,1]].
# 2. 복합 변환과 곱 순서
# 이동 후 회전은 R*T*p, 회전 후 이동은 T*R*p. 오른쪽부터 작용하고 보통 두 결과는 다르다.
# 행렬 곱은 결합법칙은 있지만 교환법칙은 없다. H=R*T를 미리 구하면 각 점에는 H 한 번을 적용한다.
# 3. 순방향과 역방향 사상
# 원본 점을 목적 위치로 보내는 순방향은 빈칸/중복이 생길 수 있다.
# 4. 보간법의 가정과 차이
# 최근접은 한 픽셀의 값을 선택한다. 빠르고 레이블을 보존하지만 계단 모양이 보일 수 있다.
# x=i+u,y=j+v의 가중치는 (1-u)(1-v),u(1-v),(1-u)v,uv이며 합은 1이다.
# 보간은 새로 측정한 세부 정보가 아니다. 영역 번호처럼 범주 값에는 보통 최근접을 선택한다.

#endregion

import cv2 as cv
import sys

#%% 영상의 일부를 잘라 보간법 비교 [영상처리_2 PPT 38~42쪽, 프로그램 3-8]
# PPT의 rose.png 대신 프로젝트에 있는 soccer.jpg를 사용한다(run.py에서 실행).
# 배열은 [y,x]이며 끝 인덱스는 포함하지 않는다. copy는 원본에 사각형을 그릴 때 패치를 보호한다.
# [함수 사용: 입출력·창]
# img=cv.imread(path, flags=cv.IMREAD_COLOR): 파일→배열. 기본 BGR, 읽기 실패 시 None인지 확인한다.
# cv.imshow(winname,img): 이름이 winname인 창에 배열을 표시한다. 저장 함수는 아니다.
# key=cv.waitKey(delay=0): GUI 이벤트·키 처리. 0은 계속 대기, 양수는 ms, 입력 없으면 보통 -1이다.
# cv.destroyAllWindows(): OpenCV 창을 닫는다. 캡처 자원 해제는 cap.release()이다.
img=cv.imread('soccer.jpg')
if img is None:
    sys.exit('No File exists.')

# [기하 변환 이론: 영상처리_2 PPT 32~39쪽]
# 동차 좌표 p=[x,y,1]^T에서 이동·회전·크기는 행렬 곱으로 합성한다. 이동 후 회전은 R*T*p이다.
# 열벡터에서는 오른쪽 행렬부터 적용되고 일반적으로 R*T와 T*R은 다르다.
# 순방향 사상은 목적 영상에 빈칸이 생길 수 있어 목적 좌표에서 원본을 찾는 역방향 사상을 쓴다.
# 역으로 찾은 원본 좌표가 실수이면 보간이 필요하다. 아래 resize는 보간의 결과를 비교한다.
# 현재 patch.shape=(100,100,3), dtype=uint8이다. copy 덕분에 파란 사각형이 패치를 덮지 않는다.
patch=img[250:350,170:270,:].copy()
# [함수 사용] cv.rectangle(img, pt1, pt2, color, thickness=1)
# pt1·pt2=(x,y) 꼭짓점, color=BGR 색. img에 그리며 -1은 채우기이다. (x,y,w,h) 사각형 인자 형식도 있다.
# [PPT 인자 보충] OpenCV 기본 30쪽: lineType은 선 연결·안티앨리어싱, shift는 좌표의 소수 비트 수. thickness<0이면 채우며 img를 직접 수정한다.
img=cv.rectangle(img,(170,250),(270,350),(255,0,0),3)

#%% 같은 패치를 5배 확대
# x=i+u,y=j+v의 양선형 가중치: (1-u)(1-v),u(1-v),(1-u)v,uv. 합은 1이다.
# u=v=0.5면 네 이웃 평균. 0,100,200,100이면 결과는 100이다.
# 세 결과는 모두 (500,500,3). 레이블을 LINEAR로 섞으면 의미 없는 새 영역 번호가 생길 수 있다.
# [함수 사용] dst=cv.resize(src, dsize, fx=0, fy=0, interpolation=cv.INTER_LINEAR)
# dsize=(너비,높이). (0,0)이면 fx·fy 배율로 크기를 정하며, interpolation은 보간법이다. 결과 영상을 반환한다.
patch1=cv.resize(patch,(0,0),fx=5,fy=5,interpolation=cv.INTER_NEAREST)
patch2=cv.resize(patch,(0,0),fx=5,fy=5,interpolation=cv.INTER_LINEAR)
patch3=cv.resize(patch,(0,0),fx=5,fy=5,interpolation=cv.INTER_CUBIC)

cv.imshow('Original',img)
cv.imshow('Nearest',patch1)
cv.imshow('Linear',patch2)
cv.imshow('Cubic',patch3)
cv.waitKey()
cv.destroyAllWindows()
