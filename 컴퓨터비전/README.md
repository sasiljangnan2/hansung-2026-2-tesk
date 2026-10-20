# 컴퓨터 비전 실행 방법

## VS Code에서 바로 실행

1. VS Code에서 프로젝트 전체 또는 `컴퓨터비전` 폴더를 엽니다.
2. 실행할 예제 파일(예: `ch3_3_이진화.py`)을 엽니다.
3. 오른쪽 위의 ▶ **Run Python File** 버튼 또는 `Ctrl+F5`를 누릅니다. 디버깅은 `F5`로 실행합니다.
4. 이미지 창에서 키를 누르면 다음 단계로 넘어갑니다. 마우스 그리기 예제는 이미지 창에서 `q`를 누르면 종료됩니다.

권장 Python/Python Debugger 확장이 필요합니다. 인터프리터를 이미 별도로 선택했다면 `Ctrl+Shift+P` → `Python: Select Interpreter`에서 `컴퓨터비전/.venv/Scripts/python.exe`를 선택하세요.

설정 변경 전에 열어 둔 터미널은 닫고 새 터미널을 열어 주세요. 실행 경로는 이미지 파일이 있는 프로젝트 최상위 폴더로 설정되어 있습니다.

`#%%` 셀은 Jupyter 확장을 설치하고 같은 가상환경을 선택한 뒤 **Run Cell**로 실행할 수 있습니다. 설정 변경 전에 열어 둔 Interactive 창은 닫고 다시 여세요.

파일 선택 메뉴를 사용하려면 `run.py`를 열고 실행하세요. 새 `.py` 파일도 메뉴에 자동으로 표시됩니다.

## 예제 파일 이름과 순서

장 번호 뒤에 핵심 단어 하나를 붙였습니다. 예전 코드나 Word 문서의 파일명을 찾을 때는 아래 표를 보세요.

| 예전 파일명 | 현재 파일명 |
|---|---|
| `ch1_1_intro.py` | [ch1_1_소개.py](ch1_1_소개.py) |
| `ch2_1_opencv_basics.py` | [ch2_1_OpenCV.py](ch2_1_OpenCV.py) |
| `ch2_2.py` | [ch2_2_픽셀.py](ch2_2_픽셀.py) |
| `ch2_5.py` | [ch2_5_동영상.py](ch2_5_동영상.py) |
| `ch2_6.py` | [ch2_6_그리기.py](ch2_6_그리기.py) |
| `ch2_7.py` | [ch2_7_클릭.py](ch2_7_클릭.py) |
| `ch2_8.py` | [ch2_8_드래그.py](ch2_8_드래그.py) |
| `ch2_9.py` | [ch2_9_페인팅.py](ch2_9_페인팅.py) |
| `ch3_1_soccer.py` | [ch3_1_HSV.py](ch3_1_HSV.py) |
| `ch3_2_detect.py` | [ch3_2_풍선.py](ch3_2_풍선.py) |
| `ch3_3.py` | [ch3_3_이진화.py](ch3_3_이진화.py) |
| `ch3_4.py` | [ch3_4_모폴로지.py](ch3_4_모폴로지.py) |
| `ch3_5.py` | [ch3_5_감마.py](ch3_5_감마.py) |
| `ch3_6.py` | [ch3_6_히스토그램.py](ch3_6_히스토그램.py) |
| `ch3_7.py` | [ch3_7_필터.py](ch3_7_필터.py) |
| `ch3_8.py` | [ch3_8_보간.py](ch3_8_보간.py) |
| `ch3_9.py` | [ch3_9_속도.py](ch3_9_속도.py) |
| `ch4_1.py` | [ch4_1_Sobel.py](ch4_1_Sobel.py) |
| `ch4_1_food_edge.py` | [ch4_1_음식에지.py](ch4_1_음식에지.py) |
| `ch4_2.py` | [ch4_2_Canny.py](ch4_2_Canny.py) |
| `ch4_3.py` | [ch4_3_외곽선.py](ch4_3_외곽선.py) |
| `ch4_4.py` | [ch4_4_허프.py](ch4_4_허프.py) |
| `ch4_4_detect_road_lane.py` | [ch4_4_정지차선.py](ch4_4_정지차선.py) |
| `ch4_4_lane_detection_video.py` | [ch4_4_동영상차선.py](ch4_4_동영상차선.py) |
| `ch4_5.py` | [ch4_5_SLIC.py](ch4_5_SLIC.py) |
| `ch4_6.py` | [ch4_6_Ncut.py](ch4_6_Ncut.py) |
| `ch4_7.py` | [ch4_7_GrabCut.py](ch4_7_GrabCut.py) |
| `ch4_8.py` | [ch4_8_모멘트.py](ch4_8_모멘트.py) |
| `ch4_findContour_binsryGroup_실습용.py` | [ch4_외곽선실습.py](ch4_외곽선실습.py) |
| `ch4_Snake_astrnut.py` | [ch4_Snake.py](ch4_Snake.py) |
| `ch4_watershed.py` | [ch4_워터셰드.py](ch4_워터셰드.py) |
| `ch4_watershed_hand.py` | [ch4_손워터셰드.py](ch4_손워터셰드.py) |
| `ch5_1.py` | [ch5_1_해리스직접.py](ch5_1_해리스직접.py) |
| `ch5_2.py` | [ch5_2_해리스함수.py](ch5_2_해리스함수.py) |

`run.py`는 예제 선택 메뉴이므로 이름을 유지합니다. 변경한 파일명도 메뉴에 자동으로 표시됩니다.

주석에만 설명한 주제도 있으므로 자세한 구현 범위는 파일 안의 설명과 [PPT 전체 확인표](시험공부/PPT_전체_확인표.md)를 함께 보세요.

## 실행 환경

- Python 3.13 기반 가상환경: `컴퓨터비전/.venv`
- 패키지: `requirements.txt`에 고정된 OpenCV, NumPy, Matplotlib, IPython/Jupyter 등
- 이미지 파일은 프로젝트 최상위 폴더에 그대로 두세요.
- 가상환경은 Git에 포함하지 않습니다.

패키지 복구가 필요한 경우 프로젝트 최상위 폴더의 PowerShell 터미널에서 실행합니다 (인터넷 필요):

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\컴퓨터비전\setup.ps1
```

새 PC에서는 Python 3.13 또는 Anaconda를 먼저 설치하세요. `setup.ps1`은 사용자 폴더의 Anaconda/Miniconda, `py`, `python` 순으로 탐색합니다.

## Spyder에서 실행

1. 이 폴더의 `start_spyder.cmd`를 더블클릭합니다.
2. Spyder에서 실습 `.py` 파일을 열고 `F5`를 누릅니다.
3. `#%%`로 나뉜 셀은 `Ctrl+Enter`로 실행합니다.
4. OpenCV 이미지 창이 열리면 해당 창을 클릭한 뒤 키를 눌러 다음 단계로 진행합니다.

실행기는 이 폴더의 `.venv`에 설치한 Spyder와 CV 패키지를 사용합니다.
설정은 `.spyder-cv`에 저장되며 시작 작업 폴더는 이 폴더입니다.

기존에 설치한 다른 Spyder를 쓰려면 `Tools > Preferences > Python interpreter`에서
이 폴더의 `.venv\Scripts\python.exe`를 선택하고 콘솔의 커널을 다시 시작하세요.
다른 버전의 Spyder는 다른 버전의 spyder-kernels를 요구할 수 있으므로
위 실행기를 사용하는 것이 편합니다.

## 이미지와 동영상 경로

파일을 읽는 모든 실습은 `cv_assets.py`의 공통 함수를 사용합니다.
Spyder의 현재 작업 폴더 대신 `컴퓨터비전` 폴더, 그 상위 폴더 순서로 자료를 찾습니다.
이미지는 한글 경로를 지원하며 흑백·알파 채널 읽기 옵션을 그대로 유지합니다.
`cv_assets.py`는 실습 파일과 함께 보관하세요.

현재 프로젝트에 없는 자료는 `mistyroad.jpg`, `apples.jpg`,
`solidWhiteRight.jpg`, `slow_traffic_small.mp4`입니다.
해당 예제를 실행하려면 수업 원본 자료를 위 두 폴더 중 하나에 넣어 주세요.
자료가 없으면 필요한 파일명과 검색한 경로를 오류 메시지로 표시합니다.
