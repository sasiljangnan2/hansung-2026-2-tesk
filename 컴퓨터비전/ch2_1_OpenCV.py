# -*- coding: utf-8 -*-
# ==================================================================
# 2장: OpenCV 설치 및 기본 사용법
# 기준: 2장 OpenCV 설치 및 기본 사용법_26.pptx, 38슬라이드.
# 쪽수는 표지를 포함한 PowerPoint 순서이다. 관련 내용끼리 #%%로 묶었다.
# 이 파일은 공부용 설명 주석만 담고 있다. 아래 명령·코드 예시는 자동 실행되지 않는다.
# 출력 예시는 PPT의 배열과 프로젝트의 soccer.jpg를 사용하여 확인했다.
# 기존 ch2_2_픽셀.py, ch2_5_동영상.py~ch2_9_페인팅.py의 실행 코드는 그대로 두었다.
# ==================================================================


#%% 1. OpenCV의 뜻·역할·특징 [PPT 2쪽]
# OpenCV는 Open Source Computer Vision Library의 약자이다.
# 영상처리·컴퓨터비전·기계학습에 필요한 자료 구조와 알고리즘을 제공하는 라이브러리이다.
# 예: 영상 읽기·필터링·색 변환·에지·특징점·기하 변환·물체 검출·추적 등.
# 라이브러리는 필요한 기능을 호출해 쓰는 도구 모음이며, Python 언어나 개발 도구 자체와는 다르다.
# Windows·Linux·macOS·Android·iOS 등 여러 환경에서 사용하며 언어·기능 지원은 빌드에 따라 다를 수 있다.
#
# PPT의 역사 표기: 2006년 1.0 / 2009년 2.0 / 2015년 3.0. 초기 C API에서 C++ 중심으로 발전했다.
# Python에서는 import cv2 또는 import cv2 as cv로 사용한다.
# cv2의 '2'는 현재 설치 버전이 반드시 2.x라는 뜻이 아니다. 실제 버전은 cv2.__version__으로 확인한다.
# import cv2 as cv에서 cv는 이 파일 안에서 사용하는 별칭이다. 기능이나 설치된 패키지가 바뀌지 않는다.
#
# 특징: 반복적으로 구현해야 하는 비전 기능을 함수로 호출할 수 있고 NumPy 배열과 연결하기 쉽다.
# 장점: 기존 구현을 활용해 개발 시간을 줄이고 최적화된 내부 계산을 사용할 수 있다.
# 한계: 함수의 입력 형식·인자·실행 환경을 맞춰야 하며 호출만으로 모든 장면의 정답을 보장하지 않는다.
#
# [현재 프로젝트에서 확인한 버전: 2026-10-18]
# Python 3.13.5 / cv2.__version__은 '5.0.0' / NumPy 2.5.3.
# 위 값은 현재 가상환경의 확인값이다. 다른 PC나 나중에 설치한 환경에서도 같은 값이라는 뜻은 아니다.


#%% 2. 주요 모듈과 담당 기능 [PPT 3~4쪽]
# core: 배열·행렬 등 기본 자료 구조와 산술·수학 연산.
# imgproc: 필터링·히스토그램·색 변환·기하 변환·그리기 등 영상처리.
# highgui: namedWindow·imshow·waitKey·마우스 콜백 등 간단한 창과 사용자 입력.
# imgcodecs: imread·imwrite 등 정지 영상의 파일 읽기·쓰기.
# videoio: VideoCapture·VideoWriter 등 카메라·동영상 입출력.
# video: 움직임 분석·광류·추적·배경 모델 등 동영상 내용 분석.
#
# ml: Bayes·SVM·결정 트리·EM 등 기계학습 알고리즘.
# features2d: 지역 특징의 검출·기술·매칭 관련 기능. 함수별 실제 소속은 문서에서 확인한다.
# objdetect: 얼굴·사람 등 물체 검출 관련 기능.
# calib3d: 카메라 보정·스테레오·3D 기하 관련 기능.
# flann: 근사 최근접 이웃 등을 빠르게 찾는 기능. 특징 기술자 매칭에 연결된다.
# stitching: 여러 장의 사진을 연결한 파노라마 제작.
# dnn: 학습된 신경망을 읽고 추론하는 기능. 영상처리 함수와 신경망 추론은 역할이 다르다.
# contrib: 추가 모듈 모음. ximgproc·text·face 등 기본 배포 외의 기능을 포함할 수 있다.
#
# [PPT 표기와 실제 역할 구분]
# PPT는 highgui에 영상 입출력, video에 비디오 입출력을 함께 적었다.
# 공부할 때는 정지 파일 입출력=imgcodecs, 카메라/동영상 입출력=videoio, 창·키·마우스=highgui로 나누면 명확하다.
# contrib 전체를 '특허 알고리즘만 모은 하나의 모듈'로 이해하지 않는다. 추가 기능의 모음이다.
# 설치 패키지·빌드 옵션에 따라 사용할 수 있는 함수가 달라질 수 있다.
# Python의 cv.rectangle처럼 cv 아래로 보인다고 모든 기능이 같은 내부 모듈에 속하는 것은 아니다.


#%% 3. C++ 라이브러리 파일과 함수 선언 읽기 [PPT 5~7,30,37쪽]
# C++ 헤더(.hpp): 함수·클래스의 선언을 제공하여 컴파일할 때 사용할 수 있게 한다.
# Windows의 .lib: 링크에 사용되는 라이브러리 파일이다. DLL용 import library일 수도 있다.
# .dll: 실행 시 동적으로 불러오는 라이브러리이다.
# 파일명 끝의 d는 PPT의 예에서 Debug용을 뜻한다. Release용 이름에는 해당 d가 없다.
# 예: opencv_core500d.lib / opencv_core500.lib. 숫자 부분은 버전·빌드의 표기와 관계된다.
# opencv_world는 여러 모듈을 묶는 통합 라이브러리 형태이며, opencv.hpp는 통합 헤더의 예이다.
# PPT는 core500 예시와 world460 예시를 함께 쓴다. 모든 설치에서 파일 숫자가 같다고 외우지 않는다.
# Python의 import cv2 예제에 C++의 #include나 .lib 링크 설정을 그대로 붙이지 않는다.
#
# [공식 문서의 선언을 읽는 순서]
# ① 언어: C++ 선언과 Python 호출 형식을 구분한다.
# ② 함수명: 어떤 작업을 하는지 확인한다.
# ③ 입력·출력: 반환값인지, 인자로 받은 배열을 수정하는지 확인한다.
# ④ 자료형·크기·채널: 허용하는 입력 형식을 확인한다.
# ⑤ 필수 인자·선택 인자·기본값: 생략하면 어떤 설정인지 확인한다.
# ⑥ 제한: 영상 크기·커널 크기·자료형 등에 조건이 있는지 확인한다.
#
# [함수 사용] cv.rectangle(img, pt1, pt2, color, thickness=1)
# pt1·pt2=(x,y) 꼭짓점, color=BGR 색. img에 그리며 -1은 채우기이다. (x,y,w,h) 사각형 인자 형식도 있다.
# [PPT 인자 보충] OpenCV 기본 30쪽: lineType은 선 연결·안티앨리어싱, shift는 좌표의 소수 비트 수. thickness<0이면 채우며 img를 직접 수정한다.
# cv.rectangle(img, pt1, pt2, color[, thickness[, lineType[, shift]]]) → img
# 문서의 대괄호는 선택 인자를 표시한다. 대괄호까지 그대로 입력하라는 뜻은 아니다.
# C++의 cv::Point는 (x,y) 좌표를 담는 형식이며 Python에서는 보통 (x,y) 튜플로 전달한다.
# InputArray는 입력 배열, OutputArray는 출력, InputOutputArray는 읽고 수정하는 인자의 성격을 나타낸다.
# rectangle·circle은 입력 영상에 그려 넣는다. cvtColor·resize는 보통 반환 영상을 받아 사용한다.
#
# [Canny 선언 예: PPT 7쪽]
# image는 입력, edges는 결과 에지 영상, threshold1/threshold2는 두 임계값이다.
# apertureSize=3은 기본 Sobel 크기, L2gradient=False는 기본 강도 계산 방식의 선택이다.
# [함수 사용] edges=cv.Canny(src, threshold1, threshold2, apertureSize=3, L2gradient=False)
# 수업 예제의 src=uint8 명암 영상, 두 threshold=히스테리시스 기준. 단일 채널 에지 맵을 반환한다.
# [PPT 인자 보충] 에지 검출 28쪽: apertureSize는 Sobel 크기(3/5/7). L2gradient=True면 sqrt(dx²+dy²), False면 |dx|+|dy|로 강도를 계산한다. 반환은 0/255의 uint8 에지 맵이다.
# cv.Canny(gray,100,200)처럼 Python에서는 반환값으로 에지 영상을 받을 수 있다.
# 단순히 인자 순서를 외우기보다 역할과 기본값을 함께 본다. 에지 이론은 ch4_2_Canny.py와 연결된다.


#%% 4. Python·IDE·라이브러리·가상환경 [PPT 8~12쪽]
# Python: 작성한 코드를 실행하는 언어와 인터프리터.
# IDE: 편집·실행·디버깅 등을 돕는 개발 환경. PPT는 Spyder를 사용한다.
# 라이브러리: NumPy·OpenCV·TensorFlow처럼 필요한 계산 기능을 제공하는 패키지.
# Anaconda: Python·도구·패키지와 환경 관리 기능을 묶어 제공하는 배포판이다.
# 가상환경: 프로젝트마다 사용할 Python·패키지 구성을 분리하기 위한 환경이다.
#
# 가상환경의 장점: 프로젝트 사이의 버전 충돌을 줄이고 어떤 환경에서 실행했는지 관리하기 쉽다.
# 주의점: 설치한 환경과 실행하는 환경이 다르면 같은 PC에서도 모듈을 못 찾을 수 있다.
# 가상환경을 만들었다고 필요한 라이브러리가 모두 자동으로 설치되는 것은 아니다.
# IDE를 켠 환경, 선택한 Python 인터프리터, pip가 설치하는 환경의 관계를 확인한다.
#
# [PPT의 명령 흐름: 설명용이며 이 파일에서 실행되지 않음]
# python --version                  : 현재 Python 버전 확인.
# conda --version                   : conda 버전 확인. Anaconda 전체의 단일 버전과 같다고 보지는 않는다.
# conda create -n cv python=3.13     : cv라는 환경 생성. -n은 이름 지정.
# conda activate cv                 : 이후 명령이 사용할 환경 활성화.
# conda install spyder              : 해당 환경에 Spyder 설치.
# python -m pip install opencv-python  : 선택한 Python 환경에 OpenCV 설치.
# python -m pip install tensorflow    : TensorFlow 설치 예시.
# conda list                        : 활성 환경의 패키지 목록 확인.
# import cv2; print(cv2.__version__)  : Python 안에서 실제 불러온 OpenCV 버전 확인.
#
# opencv-python과 opencv-contrib-python은 둘 다 cv2를 제공하므로 필요한 종류 하나를 선택한다.
# headless 패키지는 GUI 의존성을 줄인 형태로, imshow 창을 쓰는 이 수업 예제의 환경과 구분한다.
# TensorFlow·Python의 지원 버전 관계는 배포 시점에 따라 달라진다. PPT의 특정 시점 설명을 영구 조건으로 외우지 않는다.
# 위 설치 명령은 PPT를 이해하기 위한 예시이며 기존 환경을 다시 설치하거나 바꾸지 않았다.
#
# [현재 프로젝트 연결]
# 수업 PPT는 conda의 cv 환경을, 이 프로젝트는 컴퓨터비전/.venv의 Python을 사용한다.
# 환경 이름 자체보다 실제 실행 중인 Python과 패키지가 맞는지가 중요하다.
# 현재 설정은 VS Code에서 Python 실행 버튼이나 F5를 사용할 수 있고 run.py에는 예제 선택 메뉴가 있다.
# 영상의 상대 경로는 실행 위치 기준이다. run.py는 영상이 있는 프로젝트 최상위 폴더에서 예제를 실행한다.


#%% 5. Spyder 화면과 셀 실행 [PPT 13쪽]
# Editor: 코드를 쓰고 저장하는 곳.
# Variable Explorer: 현재 실행 환경의 변수·배열 내용을 확인하는 곳.
# IPython Console: 코드를 실행하고 출력·오류·표현식 값을 확인하는 곳.
# #%%는 Spyder 등에서 코드를 셀로 나누는 표시이며 Python 문법에서는 주석이다.
#
# PPT의 단축키: 파일 실행 F5 / 현재 셀 실행 Ctrl+Enter / 실행 후 다음 셀 Shift+Enter / 선택 실행 F9.
# IDE·설정에 따라 단축키는 다를 수 있다. VS Code의 키 동작과 모두 같다고 보지 않는다.
# 셀 실행은 현재 세션의 변수 상태를 이어 쓴다. 앞 셀을 생략하면 변수가 없거나 이전 실행값이 남을 수 있다.
# 따라서 시험의 출력 문제도 변수에 어떤 값이 들어 있는지부터 순서대로 확인해야 한다.
# 이 파일은 설명 주석만 있으므로 셀을 실행해도 새 계산 결과나 영상 창은 생기지 않는다.


#%% 6. 객체·클래스·속성·메소드 [PPT 14~16쪽]
# 클래스(class): 어떤 데이터와 동작을 갖는 객체인지 정의하는 형식.
# 객체(object): 그 클래스에 속하는 실제 인스턴스. a,b,c 배열은 각각 별도의 ndarray 객체가 될 수 있다.
# 속성(attribute): 객체가 가지고 있는 정보. a.shape·a.dtype·a.ndim 등을 예로 볼 수 있다.
# 메소드(method): 객체와 연결되어 호출하는 함수. a.sort()·a.copy() 등이 예이다.
# 속성 확인과 함수 호출을 구분한다. a.shape는 크기 정보이고 a.sort()는 정렬 동작을 호출한다.
# PPT의 '객체가 능동적'이라는 표현은 객체의 메소드를 통해 동작을 수행한다는 발상이다.
# 객체가 프로그램의 호출 없이 스스로 자동으로 동작을 시작한다는 뜻은 아니다.
#
# type(a): a의 클래스·자료형 종류를 확인. ndarray라면 <class 'numpy.ndarray'>.
# dir(a): 사용 가능한 이름들의 목록. 메소드뿐 아니라 shape 같은 속성 이름도 포함된다.
# help(a.sort): sort의 사용법을 확인. 이 단계에서 실제 정렬을 수행하지는 않는다.
# a.sort: 메소드 자체에 대한 참조. a.sort(): 실제 메소드 호출.
# type(img)와 img.dtype은 다른 질문이다. 앞은 객체의 클래스, 뒤는 배열 원소의 자료형이다.
#
# [장점·주의점]
# 배열 객체 하나에서 데이터와 관련 기능을 함께 다룰 수 있고 같은 인터페이스로 여러 배열을 처리할 수 있다.
# 다만 어떤 메소드는 원본을 바꾸고, 어떤 함수는 새 값을 반환한다. 반환값을 무조건 원본에 대입하면 안 된다.


#%% 7. 프로그램 2-1: 정렬과 출력값 [PPT 14,16쪽]
# [함수 사용] a=np.array(values, dtype=None)
# values=리스트 등의 입력, dtype=원소 자료형(생략하면 추론). 지정한 값을 가진 ndarray를 만든다.
# a=np.array([4,5,0,1,2,3,6,7,8,9,10,11])
# print(a)          → [ 4  5  0  1  2  3  6  7  8  9 10 11]
# print(type(a))    → <class 'numpy.ndarray'>
# print(a.shape)    → (12,)
# a.sort()
# print(a)          → [ 0  1  2  3  4  5  6  7  8  9 10 11]
#
# (12,)는 길이 12인 1차원 배열의 shape 튜플이다. (12,1) 같은 2차원 열 배열과 다르다.
# a.sort()는 a를 제자리에서 정렬하고 반환값은 None이다.
# 따라서 print(a.sort())의 출력은 정렬된 배열이 아니라 None이다. 호출 후 a는 이미 정렬되어 있다.
# a=a.sort()라고 쓰면 이름 a가 None을 가리켜 원래처럼 배열을 사용하기 어렵다.
# [함수 사용] dst=np.sort(a, axis=-1)
# 지정 축으로 정렬한 복사본을 반환한다. 원본을 정렬하고 None을 반환하는 a.sort()와 구분한다.
# 원본을 유지하고 정렬된 결과를 따로 받는 발상은 result=np.sort(a)와 비교한다.
# 2차원 a.sort()의 기본 axis=-1은 각 행의 마지막 축을 정렬하며 전체 원소를 한 줄로 합치는 뜻이 아니다.
#
# b=np.array([-4.3,-2.3,12.9,8.99,10.1,-1.2]); b.sort()
# print(b)          → [-4.3  -2.3  -1.2   8.99 10.1  12.9 ]
# c=np.array(['one','two','three','four','five','six','seven']); c.sort()
# print(c)          → ['five' 'four' 'one' 'seven' 'six' 'three' 'two']
# 문자열은 영어 숫자의 수치 의미가 아니라 문자열의 사전식 순서로 정렬된다.
# 배열 print는 원소를 쉼표 없이 나타내며 열 정렬을 위한 공백이 들어갈 수 있다.
# 대화형 콘솔에서 a만 평가한 표현과 print(a)의 출력 표기(array(...) 등)는 같지 않을 수 있다.


#%% 8. 영상 읽기·실패 검사·표시·저장 [PPT 17쪽, 프로그램 2-2]
# import cv2 as cv / import sys로 필요한 모듈을 불러온다.
# [함수 사용: 입출력·창]
# img=cv.imread(path, flags=cv.IMREAD_COLOR): 파일→배열. 기본 BGR, 읽기 실패 시 None인지 확인한다.
# cv.imshow(winname,img): 이름이 winname인 창에 배열을 표시한다. 저장 함수는 아니다.
# key=cv.waitKey(delay=0): GUI 이벤트·키 처리. 0은 계속 대기, 양수는 ms, 입력 없으면 보통 -1이다.
# cv.destroyAllWindows(): OpenCV 창을 닫는다. 캡처 자원 해제는 cap.release()이다.
# ok=cv.imwrite(path,img): 배열을 영상 파일로 저장한다. ok는 성공 여부이며 표시와 별개이다.
# cv.namedWindow(winname): 마우스 콜백 등을 등록할 이름 있는 창을 만든다.
# img=cv.imread('soccer.jpg')는 파일을 읽어 영상 배열을 얻는다.
# 일반적인 기본 컬러 읽기는 BGR 3채널이며 img가 단순한 파일 이름 문자열인 것은 아니다.
# 파일이 없거나 읽을 수 없으면 Python에서는 None이 반환될 수 있다.
#
# if img is None: sys.exit('파일을 찾을 수 없습니다.')
# 이 검사는 유효한 영상이 아닌데 shape 조회·cvtColor·imshow 등을 진행하는 오류를 막는다.
# None은 값이 없음을 나타내는 객체이다. 숫자 0이나 모든 픽셀이 0인 검정 영상과 다르다.
# print(type(None)) → <class 'NoneType'>. 정상으로 읽은 영상의 type은 numpy.ndarray이다.
# img is None은 None 객체인지 확인한다. if not img는 ndarray의 참/거짓 판정 문제를 일으킬 수 있어 같은 검사로 쓰지 않는다.
#
# cv.imshow('Image Display',img): 지정한 창에 영상을 표시한다. 창 이름과 표시할 영상을 인자로 받는다.
# cv.waitKey(): 키 입력을 기다리고 GUI 이벤트 처리를 진행한다. 기본 delay=0은 기한 없이 기다린다.
# cv.destroyAllWindows(): OpenCV의 표시 창들을 닫는다. 배열 변수를 삭제하거나 카메라를 해제하는 함수는 아니다.
# cv.imwrite('result.jpg',img): 영상을 파일에 저장한다. imshow로 보여 주기만 해서는 저장되지 않는다.
# imwrite의 성공 여부는 반환값으로 확인할 수 있고, 인코딩 조건 등의 문제는 예외로 나타날 수도 있다.
# JPEG처럼 손실 압축되는 형식은 저장 후 다시 읽었을 때 픽셀 값이 달라질 수 있다.
#
# 상대 경로는 현재 작업 폴더를 기준으로 찾는다. 파일이 코드 옆에 있는지와 실행 위치는 따로 확인한다.
# 실제 읽은 영상의 채널·자료형은 읽기 옵션과 파일 형식에 따라 달라질 수 있다.
# 읽기 옵션의 예: IMREAD_GRAYSCALE=명암으로 읽기 / IMREAD_UNCHANGED=알파 등 보존.


#%% 9. 영상 배열의 구조와 픽셀 좌표 [PPT 18~19쪽 · ch2_2_픽셀.py]
# OpenCV의 일반적인 Python 영상은 NumPy의 ndarray로 표현된다.
# 명암 영상은 보통 shape=(높이,너비)의 2차원 배열, BGR 영상은 (높이,너비,3)의 3차원 배열이다.
# 배열의 차원 수와 물리적인 세계의 3차원은 다른 개념이다. 채널 축이 있다고 깊이가 측정된 것은 아니다.
# 컬러 영상의 세 채널은 B=0, G=1, R=2 순서이다.
#
# 배열 접근: img[행,열,채널] = img[y,x,c].
# 그리기 좌표: (x,y). 같은 지점이라도 배열 접근과 튜플 순서가 다르다.
# 원점은 왼쪽 위이며 x는 오른쪽으로, y는 아래쪽으로 증가한다.
# (H,W,3)이면 y 범위는 0~H-1, x 범위는 0~W-1이다.
#
# [현재 soccer.jpg에서 확인한 출력]
# print(type(img))     → <class 'numpy.ndarray'>
# print(img.shape)     → (948, 1434, 3)
# print(img.ndim)      → 3
# print(img.dtype)     → uint8
# print(img.size)      → 4078296   # 948*1434*3, 채널 원소까지 포함한 전체 원소 수.
# print(img.nbytes)    → 4078296   # uint8의 한 원소가 1바이트인 현재 배열의 메모리 바이트 수.
# print(img[0,0])      → [162 104  98]
# print(img[0,1])      → [163 105  99]
# print(img[0,0,0],img[0,0,1],img[0,0,2]) → 162 104 98
# 픽셀의 개수는 948*1434이고 img.size는 그 세 배이다. 압축된 JPG 파일 크기와 nbytes도 다르다.
#
# uint8은 원소가 0~255인 8비트 부호 없는 정수라는 뜻이다. 255단계가 아니라 256개 값이 있다.
# 위 크기·픽셀 값은 현재 파일의 확인값이다. 다른 이미지에서도 같다고 외우지 않는다.
# img[:,:,2]는 R 채널 전체이고 img[30,40]은 y=30,x=40인 위치의 BGR 벡터이다.
# img[250:350,170:270,:].shape → (100,100,3). 슬라이스의 끝 350,270은 포함하지 않는다.
# 기본 슬라이스는 원본과 메모리를 공유할 수 있다. 따로 그려도 원본을 보존하려면 copy()를 고려한다.


#%% 10. 명암 변환·크기 축소·보간 [PPT 20~21쪽, 프로그램 2-3]
# [함수 사용] dst=cv.cvtColor(src, code)
# src=입력 영상, code=색 변환 상수. 예: BGR2GRAY·BGR2HSV·HSV2BGR. 변환 영상을 반환한다.
# [PPT 인자 보충] 영상처리_1 16쪽: dstCn=0은 출력 채널 수 자동 결정. 출력 높이·너비와 깊이는 같고 채널 수는 변환에 따라 달라진다.
# gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)는 컬러의 세 성분을 하나의 명암 값으로 바꾼다.
# 강의 식: I=round(0.299*R+0.587*G+0.114*B).
# 사람의 밝기 지각을 반영한 가중합으로 설명하며 단순한 세 채널 산술평균은 아니다.
# 입력은 BGR이므로 배열의 가중합으로 쓰면 0.114*img[:,:,0]+0.587*img[:,:,1]+0.299*img[:,:,2]이다.
# B=G=R=v이면 가중치 합이 1이어서 이상적인 명암 값은 v이다.
# 직접 계산한 실수를 uint8로 절삭하는 코드와 OpenCV의 반올림 처리에는 작은 수치 차이가 있을 수 있다.
#
# [현재 파일의 출력·형태]
# print(gray.shape)     → (948,1434)
# print(gray.ndim)      → 2
# print(gray[0,0],gray[0,1]) → 109 110
# 명암 영상은 한 채널이므로 shape에 채널 축이 없어질 수 있다. (948,1434,1)과도 배열 모양은 다르다.
#
# [함수 사용] dst=cv.resize(src, dsize, fx=0, fy=0, interpolation=cv.INTER_LINEAR)
# dsize=(너비,높이). (0,0)이면 fx·fy 배율로 크기를 정하며, interpolation은 보간법이다. 결과 영상을 반환한다.
# gray_small=cv.resize(gray,dsize=(0,0),fx=0.5,fy=0.5)
# dsize=(0,0)은 크기를 확정하지 않고 fx·fy 배율로 출력 크기를 구하는 호출이다.
# fx는 너비(x), fy는 높이(y)의 배율. 명시적인 dsize는 (너비,높이)라 shape의 앞 두 항과 순서가 반대이다.
# print(gray_small.shape) → (474,717). 양쪽 길이가 반이므로 픽셀 수는 원래의 1/4이다.
# cv.resize(gray,(100,200))의 출력 shape는 (200,100)이다.
# dsize를 명시적으로 주면 출력 크기는 그 크기를 기준으로 정한다. 서로 모순되는 크기와 배율을 혼용하지 않는다.
#
# 보간은 새 격자의 픽셀 값을 주변에서 추정하는 계산이다. 단순히 파일의 표시 크기만 바꾸는 것은 아니다.
# 기본 resize는 INTER_LINEAR. 최근접은 값의 종류 보존, 선형은 이웃 값의 가중합이라는 차이가 있다.
# 클래스 번호 마스크는 보간으로 새 클래스 번호를 만들지 않도록 최근접 등을 선택한다.
# 명암 변환은 색 정보를 줄이고 축소는 공간 정보를 줄인다. 나중에 확대해도 잃은 세부가 모두 복원되지는 않는다.
# 연결 예제: ch3_8_보간.py의 보간 비교 / ch3_9_속도.py의 명암 변환·시간 비교.


#%% 11. 카메라·동영상 열기와 프레임 읽기 [PPT 22~23쪽, 프로그램 2-4]
# [함수 사용] cap=cv.VideoCapture(source)
# source=카메라 번호 또는 동영상 경로. cap.isOpened()로 열기 확인, ret,frame=cap.read()로 읽고 cap.release()로 해제한다.
# [PPT 인자 보충] OpenCV 기본 23쪽: apiPreference=cv.CAP_ANY는 백엔드 자동 선택. cap.isOpened()는 열기 성공 여부, ret,frame=cap.read()는 다음 프레임, cap.release()는 자원 해제이다.
# cap=cv.VideoCapture(0)는 기본 카메라 인덱스 0을 여는 시도이다.
# cap=cv.VideoCapture('tunnel.mpeg')는 동영상 파일을 여는 시도이다.
# 괄호 안의 숫자 0과 문자열 '0'은 다른 입력이다. '0'을 카메라 번호라고 생각하지 않는다.
# cv.VideoCapture(0,cv.CAP_DSHOW)는 Windows의 DirectShow 백엔드를 지정하는 예이다.
# 카메라 인덱스·백엔드 지원은 환경에 따라 달라질 수 있어 모든 PC의 장치 순서를 고정해서 외우지 않는다.
# VideoCapture의 반환값은 캡처 객체이며 실제 영상 한 프레임과 다르다.
#
# cap.isOpened(): 카메라/파일을 성공적으로 열었는지 True/False로 확인한다.
# ret,frame=cap.read(): 다음 프레임을 읽고 성공 여부 ret와 영상 frame을 함께 받는다.
# 성공 시 frame은 보통 BGR ndarray이다. 실패 시 ret가 False이므로 표시·가공 전에 확인한다.
# 파일이 정상으로 열려도 끝까지 읽으면 ret=False가 될 수 있다. 장치 오류·스트림 문제도 읽기 실패의 원인이 된다.
# '열기 성공'과 '매번 프레임 읽기 성공'은 서로 다른 검사이다.
#
# 처리 순서: 열기 → isOpened 검사 → 반복 read → ret 검사 → 표시/가공 → 키 입력 → 반복 종료 → 자원 정리.
# cap.release()는 카메라/파일 캡처 자원을 해제한다. destroyAllWindows()는 창을 닫는다. 역할이 다르다.
# 이 정리 파일은 카메라를 실제로 열거나 동영상 캡처를 실행하지 않는다.


#%% 12. waitKey·ord·반복문의 의미 [PPT 22,25,31,34,36쪽]
# key=cv.waitKey(1): 최소 대기와 GUI 이벤트 처리를 거쳐 키 코드를 받는다. 키가 없으면 보통 -1이다.
# 양수 delay는 밀리초 단위이며 실제 대기 시간은 운영체제·처리 시간의 영향을 받는다.
# waitKey(1)이 있다고 동영상 전체가 1ms 간격이나 1000fps로 재생된다고 단정하지 않는다.
# waitKey(0) 또는 인자 없는 waitKey()는 키가 들어올 때까지 기다린다.
# 보통 OpenCV 창이 활성화되어 있어야 키 입력을 받을 수 있다.
#
# ord('q') → 113 / ord('c') → 99 / ord('Q') → 81. 대문자와 소문자의 코드는 다르다.
# key==ord('q')는 입력이 소문자 q인지 확인한다. key 자체가 문자 'q'인 것은 아니다.
# ord는 한 글자 문자열을 받아 정수 코드를 돌려준다. ord('quit')처럼 여러 글자를 주면 오류가 난다.
# 예제의 break는 현재 반복문을 끝낸다. 그 아래 프로그램 전체의 모든 줄을 없애거나 종료시키는 명령은 아니다.
# sys.exit(...)는 실행을 종료하는 용도여서 반복문만 빠져나오는 break와 구분한다.
#
# [GUI 연결]
# imshow만 호출하고 이벤트 처리를 하지 않으면 창이 갱신되지 않거나 곧 종료될 수 있다.
# 마우스 예제의 while+waitKey는 사용자 입력을 계속 처리하며 종료 키를 기다리기 위해 있다.
# 대기하는 줄은 그림을 그리는 함수와 역할이 달라도 창이 유지되고 반응하도록 하는 데 필요하다.


#%% 13. 프레임 수집·리스트·hstack [PPT 24~26쪽 · ch2_5_동영상.py]
# frames=[]는 프레임 배열들을 보관할 Python 리스트를 만든다.
# c키를 누르면 frames.append(frame)으로 현재 프레임을 리스트에 추가하고 q키로 수집 반복을 끝낸다.
# len(frames)는 저장한 프레임 개수, frames[0]은 첫 영상 ndarray이다.
# 리스트 자체의 type은 list이며 각 영상의 type은 numpy.ndarray이다. 서로 다른 층의 자료 구조이다.
# append는 배열을 자동으로 복사하는 함수가 아니다. 같은 배열을 나중에 수정하며 보존하려면 frame.copy()를 생각한다.
#
# if len(frames)>0 검사는 저장한 영상이 없는데 frames[0]에 접근하는 IndexError를 막는다.
# imgs=frames[0] 다음 for i in range(1,min(3,len(frames))): ...는 처음 것을 포함해 최대 3장만 연결한다.
# range의 끝은 포함하지 않는다. 7장을 저장했더라도 i는 1,2이고 최종 연결은 0,1,2번 세 장이다.
# len=1이면 추가 연결 반복은 0회. len=2이면 1회. len>=3이면 2회이다.
#
# [함수 사용] dst=np.hstack((a,b))
# 영상은 너비 축으로 연결하므로 높이·채널 등 다른 축이 같아야 한다. 새 연결 배열을 반환한다.
# imgs=np.hstack((imgs,frames[i]))는 영상을 가로로 이어 붙여 새 결과 배열을 만든다.
# 컬러 영상 hstack은 axis=1, 즉 너비 방향의 연결이다. 높이와 채널 수 등 나머지 축이 맞아야 한다.
# 영상 vstack은 높이 방향 연결이므로 너비·채널 수 등 나머지 축이 맞아야 한다.
# hstack은 평균·합성·블렌딩이 아니라 배열 연결이다. 시간 축을 새로 만들어 동영상으로 저장하는 것도 아니다.
#
# [PPT 26쪽의 조건부 출력 예시]
# 7장을 수집하고 각 프레임이 (480,640,3)이라면 len(frames) → 7.
# print(frames[0].shape) → (480,640,3), print(type(imgs)) → <class 'numpy.ndarray'>.
# 최대 3장 연결 후 print(imgs.shape) → (480,1920,3).
# (480,640,3)을 세 장 vstack한다면 shape는 (1440,640,3)이다.
# 실제 영상 해상도와 c키를 누른 횟수가 다르면 값도 달라진다. 현재 파일이 언제나 이 크기를 출력하는 것은 아니다.
#
# [작은 배열로 계산 확인]
# a=np.array([[3,5,7],[5,7,9]])일 때 a.shape는 (2,3).
# np.hstack((a,a)) → [[3,5,7,3,5,7],[5,7,9,5,7,9]], shape=(2,6).
# [함수 사용] dst=np.vstack((a,b))
# 영상은 높이 축으로 연결하므로 너비·채널 등 다른 축이 같아야 한다. 새 연결 배열을 반환한다.
# np.vstack((a,a)).shape → (4,3). print 배열은 실제 표시에서 쉼표 없이 줄이 나뉠 수 있다.
# 보충: 1차원 배열의 hstack은 axis=0으로 잇는다. 모든 차원에서 무조건 axis=1이라고 외우지 않는다.
#
# [현재 코드와 PPT의 차이]
# PPT는 웹캠을 사용하지만 ch2_5_동영상.py는 slow_traffic_small.mp4를 읽는다. 이후 수집·연결의 흐름은 같다.
# 현재 코드는 수집한 사진을 창에 보여 주며 별도의 동영상 저장 파일을 만들지는 않는다.


#%% 14. 그리기 함수·색·좌표·두께 [PPT 27~30쪽 · ch2_6_그리기.py]
# line=직선 / rectangle=직사각형 / polylines=연속 선분 / circle=원 / ellipse=타원 / putText=문자열.
# 대부분의 이 수업 예제는 입력 img를 직접 수정한다. 원본 보존이 필요하면 복사본에 그린다.
# 표시 창을 닫는다고 배열에 그려 넣은 픽셀이 되돌아가지는 않는다.
#
# cv.rectangle(img,(830,30),(1000,200),(0,0,255),2)
# 첫 두 점은 꼭짓점의 (x,y) 좌표, (0,0,255)는 BGR 빨강, 마지막 2는 선 두께이다.
# 좌표 차이는 가로 170, 세로 170이지만 끝점·선 두께·영상 경계 처리까지 고려하면 덮는 픽셀 수와는 구분된다.
# 색의 예: (255,0,0)=파랑 / (0,255,0)=초록 / (0,0,255)=빨강 / (255,255,255)=흰색.
# 배열 접근은 [y,x], 그리기 좌표는 (x,y)이다. 좌표를 바꾸면 다른 위치에 그려진다.
#
# [함수 사용] cv.putText(img, text, org, fontFace, fontScale, color, thickness=1)
# text=문자열, org=(x,y) 기준선 시작점, fontFace=글꼴, fontScale=배율. 지정한 BGR 색으로 img에 쓴다.
# [PPT 인자 보충] OpenCV 기본 28쪽: org는 글자의 왼쪽 아래 기준점. lineType은 선 모양, bottomLeftOrigin=False는 영상 원점을 왼쪽 위로 해석한다.
# cv.putText(img,'laugh',(830,24),cv.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
# 'laugh'를 파랑으로 쓰며 1은 글꼴 배율, 2는 두께이다.
# org=(830,24)는 기본 설정에서 글자의 기준선 쪽 시작 위치이며 글자 영역의 왼쪽 위 점이라고 단정하지 않는다.
# 기본 Hershey 글꼴의 지원 범위를 고려한다. 임의의 한글 문자열이 그대로 출력된다고 기대하지 않는다.
#
# 특징: 결과 위치·선·글씨를 원본 위에 빠르게 표시할 수 있다.
# 장점: 검출·분할 결과의 시각화나 간단한 사용자 인터페이스에 편리하다.
# 주의점: 덧그린 배열을 후속 분석에 쓰면 원래 픽셀이 바뀐 상태이다. 분석용과 표시용을 구분한다.


#%% 15. 콜백 함수와 마우스 이벤트 [PPT 31~33쪽 · ch2_7_클릭.py]
# 콜백(callback): 특정 일이 생겼을 때 호출되도록 미리 등록해 두는 함수이다.
# def draw(event,x,y,flags,param): ...로 처리 함수를 만든다.
# [함수 사용] cv.setMouseCallback(winname, onMouse, param=None)
# winname=창 이름, onMouse=콜백 함수 자체. 이벤트가 나면 onMouse(event,x,y,flags,param)를 호출하도록 등록한다.
# cv.namedWindow('Drawing')로 창을 만들고 cv.setMouseCallback('Drawing',draw)로 함수를 등록한다.
# 여기서는 draw라는 함수 자체를 전달한다. draw()로 지금 실행한 결과를 전달하는 것과 다르다.
# OpenCV가 그 창의 마우스 이벤트를 처리할 때 등록된 draw를 인자와 함께 호출한다.
# 창 이름은 등록할 실제 창과 맞아야 하며 waitKey 등으로 이벤트를 계속 처리해야 한다.
#
# event: 이번에 일어난 이벤트 종류. 누름·뗌·이동·더블클릭·휠 등이 있다.
# x,y: 이벤트가 발생한 영상 위치의 좌표. 배열 접근에 사용할 때는 img[y,x] 순서를 생각한다.
# flags: 누른 마우스 버튼·Ctrl·Shift·Alt 등의 상태를 비트로 담는다. 특수 키만 나타내는 것은 아니다.
# param: 등록 시 추가로 전달한 사용자 데이터. C++ 문서의 userdata와 같은 역할이다.
#
# [이벤트 상수: PPT 33쪽과 현재 환경에서 확인]
# MOUSEMOVE=0, LBUTTONDOWN=1, RBUTTONDOWN=2, MBUTTONDOWN=3.
# LBUTTONUP=4, RBUTTONUP=5, MBUTTONUP=6.
# LBUTTONDBLCLK=7, RBUTTONDBLCLK=8, MBUTTONDBLCLK=9.
# MOUSEWHEEL=10, MOUSEHWHEEL=11. 실제 코드에는 cv.EVENT_를 앞에 붙인다.
#
# [상태 플래그]
# cv.EVENT_FLAG_LBUTTON=1 / RBUTTON=2 / MBUTTON=4 / CTRLKEY=8 / SHIFTKEY=16 / ALTKEY=32.
# event=1과 flags=1은 숫자가 같아도 뜻이 다르다. 앞은 '왼쪽 누름 사건', 뒤는 '왼쪽이 눌린 상태'이다.
# 왼쪽 버튼+Shift가 함께 눌리면 flags=1|16=17처럼 여러 비트가 합쳐질 수 있다.
# flags==cv.EVENT_FLAG_LBUTTON은 오직 값이 1인 경우만 통과하므로 Shift가 함께 있으면 False이다.
# flags & cv.EVENT_FLAG_LBUTTON은 왼쪽 비트가 있는지 확인한다. bool(17 & 1) → True.
# 이것은 설명용 비교이며 기존 페인팅 코드의 == 조건을 여기서 수정한 것은 아니다.
#
# [ch2_7_클릭.py의 동작]
# 왼쪽 누름: 클릭 점을 한 꼭짓점으로 가로·세로 200인 빨간 사각형을 그린다.
# 오른쪽 누름: 가로·세로 100인 파란 사각형을 그린다.
# 클릭마다 img에 추가로 그려지므로 이전 사각형도 누적된다. 단순한 자동 인식 결과가 아니라 사용자 입력이다.


#%% 16. 드래그·좌표 저장·global [PPT 34쪽 · ch2_8_드래그.py]
# 고정 크기 사각형 대신 마우스 누른 곳과 뗀 곳을 두 꼭짓점으로 사용하는 예제이다.
# EVENT_LBUTTONDOWN: ix,iy=x,y로 시작 위치를 저장한다.
# EVENT_LBUTTONUP: cv.rectangle(img,(ix,iy),(x,y),...)로 시작점과 현재점을 연결한다.
# 두 이벤트는 서로 다른 콜백 호출이므로 시작 위치를 호출 사이에 보존해야 한다.
#
# global ix,iy는 함수 안에서 대입하는 이름을 전역 이름으로 연결한다.
# 이 선언 없이 한 호출에서만 지역 변수로 저장하면 다른 호출에서 그 시작값을 그대로 사용할 수 없다.
# 시작점이 아직 저장되지 않은 상태에서는 그 이름을 읽을 수 없으므로 초기화·이벤트 순서가 중요하다.
# 반면 img의 픽셀을 수정하는 것과 함수 안에서 img 이름 자체에 새 값을 대입하는 것은 구분한다.
# 현재 함수가 원본 배열에 그려 넣는다고 해서 항상 global img 선언이 필요한 것은 아니다.
#
# 현재 코드는 버튼을 뗄 때 사각형을 확정한다. 드래그하는 모든 순간의 미리보기를 그리는 구현은 아니다.
# 실시간 미리보기는 시작점과 마우스 이동을 관리하면서 원본 복사본에 표시하는 등의 별도 처리가 필요하다.
# 현재 ch2_8_드래그.py의 버튼을 뗐을 때를 가리키는 주석은 실제 조건 EVENT_LBUTTONUP 기준으로 읽는다.


#%% 17. 페인팅·원 채우기·lineType·shift [PPT 35~38쪽 · ch2_9_페인팅.py]
# BrushSiz=5는 원형 붓의 반지름을 뜻한다. 선 두께가 5라는 설정과는 다르다.
# LColor=(255,0,0)은 파랑, RColor=(0,0,255)는 빨강이다.
# 왼쪽 누름은 파란 원, 오른쪽 누름은 빨간 원을 그리고, 누른 상태로 이동하면 현재 위치에 원을 반복해 그린다.
# 작은 원들이 이어지면서 붓으로 칠한 것처럼 보인다. 빠른 이동 시 점 사이 간격이 생길 수 있다.
#
# [함수 사용] cv.circle(img, center, radius, color, thickness=1)
# center=(x,y), radius=반지름, color=BGR 색. img에 직접 그리며 thickness=-1이면 원 내부를 채운다.
# [PPT 인자 보충] OpenCV 기본 37쪽: lineType=LINE_4/LINE_8/LINE_AA는 선 연결·안티앨리어싱 방식. shift는 좌표의 소수 비트 수이며 thickness<0이면 채운다.
# cv.circle(img,(x,y),BrushSiz,LColor,-1)
# img=그릴 배열 / (x,y)=중심 / BrushSiz=반지름 / LColor=BGR 색 / -1=내부까지 채우기.
# thickness가 양수이면 테두리 두께, -1 또는 cv.FILLED이면 원 내부를 채운다.
# lineType 예: LINE_4·LINE_8·LINE_AA. 4/8은 선 연결 방식, AA는 가장자리 계단 현상을 완화하는 표시이다.
# 선 연결의 4/8과 영상 배열의 채널 수는 관계가 없다. AA의 중간 색도 분류 마스크의 새 클래스와 구분한다.
# shift는 좌표의 소수부를 고정소수점 비트로 나타내는 설정이며 창을 옮기는 인자가 아니다.
# 예: shift=1에서 정수 좌표 값 3은 실제 좌표 3/2를 표현한다. 기본 shift=0은 보통의 정수 좌표이다.
#
# [출력·효과 예: 실제 영상 대신 5×5 검정 배열로 확인]
# [함수 사용] a=np.zeros(shape, dtype=float)
# shape=배열 크기, dtype=자료형. 0으로 채운 배열을 만든다. 영상 마스크에는 보통 dtype=np.uint8을 지정한다.
# a=np.zeros((5,5,3),np.uint8)
# cv.circle(a,(2,2),1,(0,0,255),-1)
# print(a[2,2]) → [  0   0 255]. 원의 중심 픽셀이 빨강으로 바뀐다.
# 반환 영상을 따로 받지 않아도 a 자체가 수정된다. 원본 보존이 필요하면 a.copy()에 그린다.
#
# 특징: 마우스 콜백·버튼 상태·채워진 원을 조합해 간단한 페인팅을 만든다.
# 장점: 적은 코드로 사용자 입력을 영상에 바로 반영한다.
# 한계: 기본 예제에는 레이어·실행 취소·자동 저장·매끄러운 선분 보간이 구현되어 있지 않다.
# 창을 닫아도 그린 결과가 자동으로 파일에 저장되는 것은 아니다. 저장은 imwrite 등 별도 호출의 역할이다.


#%% 18. 기존 예제 연결과 공부 순서
# ch2_1_OpenCV.py: 이 파일. 설치·객체·배열·입출력·GUI 이론을 주석으로 모았다.
# ch2_2_픽셀.py: 영상 읽기와 표시, ndarray 형태, 픽셀 값과 패치 확인.
# 프로그램 2-3의 명암·축소는 이 파일 10번 묶음에서 설명한다. ch3_8·ch3_9와도 연결된다.
# ch2_5_동영상.py: 동영상 프레임을 리스트에 수집하고 최대 세 장을 가로로 연결.
# ch2_6_그리기.py: 고정 좌표의 사각형과 문자열. ch2_7_클릭.py: 클릭 이벤트별 고정 크기 사각형.
# ch2_8_드래그.py: 누른 점·뗀 점으로 사각형 크기 결정. ch2_9_페인팅.py: 버튼·드래그와 원형 붓 페인팅.
#
# 예제 번호와 강의 절 번호는 구분한다. '2.6절'은 영상 수집을 설명하지만 ch2_6_그리기.py는 프로그램 2-6 그리기 예제이다.
# 순서: 환경·입력 확인 → ndarray·좌표 → 명암·크기 → 프레임·키 → 그리기 → 콜백 → 드래그·페인팅.


#%% 19. 출력·서술형 복습 문답
# Q1. OpenCV·Python·IDE는 각각 무엇인가?
# A. 비전 기능 라이브러리 / 실행 언어·인터프리터 / 편집·실행·디버깅 도구이다.
#
# Q2. 가상환경을 쓰는 이유와 주의점은?
# A. 프로젝트별 패키지 충돌을 줄인다. 설치한 Python과 실제 실행하는 Python이 맞는지 확인해야 한다.
#
# Q3. import cv2 as cv의 의미는?
# A. cv2 모듈을 이 파일에서 cv라는 별칭으로 참조한다. 설치 버전 번호를 변경하는 명령이 아니다.
#
# Q4. Debug 라이브러리의 파일명 끝 d는 무엇인가?
# A. PPT의 명명 예에서 Debug용을 나타낸다. 숫자 부분은 버전·빌드 표기와 관계된다.
#
# Q5. 함수 문서의 [, thickness]는 무엇인가?
# A. 선택 인자를 표시한다. 실제 호출에 대괄호를 그대로 적는 문법은 아니다.
#
# Q6. type(a)·a.dtype·a.shape·a.ndim의 차이는?
# A. 객체 클래스 / 원소 자료형 / 축별 크기 / 축의 개수이다.
#
# Q7. dir(a)는 메소드만 보여 주는가?
# A. 속성 등 객체가 제공하는 이름들도 포함한다. shape 같은 이름도 보일 수 있다.
#
# Q8. a.shape가 (12,)이면 무엇인가?
# A. 길이 12인 1차원 배열이다. 12행 1열의 2차원 배열 (12,1)과 다르다.
#
# Q9. print(a.sort())의 출력은?
# A. None. a 자체는 정렬된다. a=a.sort()로 쓰면 a는 None을 가리킨다.
#
# Q10. 영어 숫자 문자열 정렬은 숫자 순서인가?
# A. 문자열의 사전식 순서이다. PPT의 예는 five,four,one,seven,six,three,two 순서이다.
#
# Q11. imread에 실패한 None과 검정 영상은 같은가?
# A. None은 유효한 배열이 없다는 뜻이며, 검정 영상은 원소 값이 0인 실제 배열이다.
#
# Q12. img.shape=(948,1434,3)에서 높이·너비·채널은?
# A. 높이 948, 너비 1434, 채널 3. 일반적인 컬러 읽기의 순서는 BGR이다.
#
# Q13. img[30,40,2]는 어떤 값인가?
# A. y=30,x=40인 픽셀의 R 성분이다. 그리기 좌표로는 (40,30)인 지점이다.
#
# Q14. 현재 soccer.jpg의 첫 픽셀 세 값과 명암 값은?
# A. BGR은 162,104,98이고 cvtColor의 명암 값은 109이다. 특정 파일의 확인값이다.
#
# Q15. 명암 변환은 세 채널의 산술평균인가?
# A. 아니다. 강의는 0.299R+0.587G+0.114B의 반올림 가중합으로 설명한다.
#
# Q16. 양쪽 크기를 반으로 줄이면 픽셀 수는?
# A. 1/4. 현재 명암 영상 (948,1434)은 (474,717)이 된다.
#
# Q17. cv.resize(gray,(100,200))의 shape는?
# A. (200,100). dsize는 (너비,높이), shape는 (높이,너비)이다.
#
# Q18. VideoCapture 객체와 read의 반환값은?
# A. 앞은 캡처 자원을 다루는 객체, read는 성공 여부와 현재 프레임 배열의 두 값을 반환한다.
#
# Q19. 파일이 열렸는데도 ret=False가 될 수 있는가?
# A. 파일 끝까지 읽거나 프레임을 얻지 못하면 가능하다. 열기 검사와 프레임 검사는 별개이다.
#
# Q20. release와 destroyAllWindows의 차이는?
# A. 캡처 자원 해제와 표시 창 닫기의 차이이다. 서로의 역할을 대신하지 않는다.
#
# Q21. ord('q'), ord('c'), ord('Q')는?
# A. 113, 99, 81. 대문자와 소문자는 다른 코드이다.
#
# Q22. 7장 중 최대 3장을 잇는 range(1,min(3,7))의 i 값은?
# A. 1,2. frames[0]을 먼저 썼으므로 전체로는 0,1,2번 세 장이다.
#
# Q23. (480,640,3) 세 장을 hstack하면?
# A. (480,1920,3). 높이·채널은 유지되고 너비가 더해진다. vstack이면 (1440,640,3).
#
# Q24. setMouseCallback('Drawing',draw)에서 draw()를 쓰지 않는 이유는?
# A. 지금 호출한 결과가 아니라 나중에 이벤트가 날 때 호출할 함수 자체를 등록해야 하기 때문이다.
#
# Q25. event와 flags는 같은 값인가?
# A. event는 사건 종류, flags는 버튼·수정 키의 상태 비트이다. 숫자가 같아도 역할은 다르다.
#
# Q26. 왼쪽+Shift의 flags=17일 때 flags==1과 bool(flags & 1)은?
# A. False와 True. 비트 검사로 왼쪽 버튼이 눌린 상태를 확인할 수 있다.
#
# Q27. ch2_8_드래그.py에서 시작 좌표를 저장하는 이유는?
# A. 버튼을 누를 때와 뗄 때가 다른 콜백 호출이어서 첫 위치를 유지해야 사각형을 그릴 수 있다.
#
# Q28. cv.circle의 radius=5와 thickness=-1은?
# A. 반지름 5인 원을 내부까지 채워 그린다. 붓 반지름과 테두리 두께를 구분한다.
#
# Q29. 마우스로 그린 img가 창을 닫으면 원본으로 돌아오는가?
# A. 아니다. 배열을 직접 수정한 것이므로 원본 보존에는 복사가, 파일 저장에는 별도 저장 호출이 필요하다.
#
# Q30. #%% 셀 실행에서 이전 실행값을 조심해야 하는 이유는?
# A. 현재 세션의 변수가 남아 있을 수 있어 셀을 실행한 순서에 따라 결과가 달라질 수 있기 때문이다.


#%% 참고: 강의와 대조한 공식 자료
# 주된 학습 범위: 2장 OpenCV 설치 및 기본 사용법_26.pptx 1~38쪽.
# 출력 예시는 프로젝트 가상환경과 실제 soccer.jpg, 작은 배열로 확인했다.
# 공식 함수 문서에서는 버전·Python 호출 형식·지원 환경을 함께 확인한다.
# 영상 파일 입출력: https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html
# 창·키·마우스: https://docs.opencv.org/4.x/d7/dfc/group__highgui.html
# 그리기: https://docs.opencv.org/4.x/d6/d6e/group__imgproc__draw.html
# VideoCapture: https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html
# 크기 변환: https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html
# 명암 변환: https://docs.opencv.org/4.x/de/d25/imgproc_color_conversions.html
# Python 배포 패키지: https://pypi.org/project/opencv-python/
# ndarray.sort: https://numpy.org/doc/stable/reference/generated/numpy.ndarray.sort.html
# hstack: https://numpy.org/doc/stable/reference/generated/numpy.hstack.html

#%% [PPT 전체 확인 보충] 함수 선언·메소드·그리기 (모두 설명 주석)
# 관련 PPT: 2장 OpenCV 설치 및 기본 사용법_26 7·14~16·23·27·30·37쪽.
# [함수 사용법 읽기] C++의 InputArray/OutputArray와 Python의 반환값 표현을 구분한다.
# Python 설명에서 [인자]는 선택 인자라는 뜻이며 대괄호를 실제 호출에 쓰는 것이 아니다.
# 출력 dst를 생략할 수 있는 함수는 보통 dst=cv.함수(입력,...)로 쓴다. 나머지 선택 인자는 키워드로 적으면 위치 혼동을 줄인다.
# 아래 예시는 설명용이다. 이 파일 전체는 주석이며 실행·창 표시·파일 저장을 하지 않는다.
#
# [함수 사용] a.sort(axis=-1): 원본을 지정 축으로 정렬하고 None을 반환한다. np.sort(a)는 정렬한 새 배열이다.
# [함수 사용] type(obj)는 클래스, dir(obj)는 멤버 이름 목록, help(obj)는 도움말을 보여준다.
# [함수 사용] ord('q') -> 113, chr(113) -> 'q'. 키 입력 정수와 문자를 비교할 때 연결한다.
# [함수 사용] cap.get(propId)는 영상 속성 값(float)을 읽는다. 예: cap.get(cv.CAP_PROP_FPS).
# FPS/프레임 수 등의 값은 장치·파일·백엔드에 따라 없거나 부정확할 수 있어 고정 출력으로 외우지 않는다.
#
# [함수 사용] cv.polylines(img, pts, isClosed, color, thickness=1)
# pts는 int32 점 배열들의 목록, isClosed=True는 끝점과 시작점을 잇는다. img를 직접 수정한다.
# 사용 예: cv.polylines(img,[np.array([[10,10],[50,10],[30,40]],np.int32)],True,(0,255,0),2)
# [함수 사용] cv.ellipse(img, center, axes, angle, startAngle, endAngle, color, thickness=1)
# center=(x,y), axes=(가로 반축,세로 반축), angle=회전각(도), start/endAngle=그릴 호의 범위(도).
# 사용 예: cv.ellipse(img,(100,100),(60,30),0,0,360,(0,0,255),2) -> 빨간 전체 타원.
# 두 함수는 PPT 27쪽에 이름이 나온 그리기 기능이며 기존 실행 예제에 새 호출을 넣은 것은 아니다.

