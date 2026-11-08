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

## 5번째 과제: 배경 합성

[ch4_7_배경합성.py](ch4_7_배경합성.py)를 실행하면 `김준휘.jpg`의 인물을 분리해 제공한 초원 배경에 합성합니다. `run.py`의 선택 메뉴에서도 실행할 수 있습니다.

- **기존 코드 활용:** `ch4_7_GrabCut.py`의 마우스 칠하기·`grabCut`·`np.where`, `ch3_2_풍선.py`의 `inRange`·`copyTo`, `ch4_8_모멘트.py`의 `boundingRect`, 보간 예제의 `resize`를 연결했습니다.
- **동작:** 입력 → 초기 마스크 → GrabCut 영역 분할 → 전경 크기 조절 → 새 배경에 복사. 전경은 종횡비를 유지하고 마스크는 최근접 보간합니다.
- **현재 사진의 값:** BGR 세 채널이 각각 235~255이면 배경 후보(2), 맨 위 3%는 확실한 배경(0)으로 둡니다. 셔츠·넥타이 안쪽 사각형과 얼굴·정장의 선은 확실한 전경(1)입니다. 나머지는 GrabCut이 분할합니다. 좌우 가장자리의 정장 어깨는 배경으로 강제 지정하지 않습니다.
- **수정 방법:** `Painting` 창에서 왼쪽 드래그는 남길 부분, 오른쪽 드래그는 지울 부분입니다. `G`로 다시 분할하고 저장하며, `Q` 또는 Esc로 종료합니다.
- **다른 사진:** `INPUT_FILE`은 사진 경로입니다. 현재 초기 마커의 위치와 흰색 기준은 이 증명사진에 맞춘 값이므로, 다른 구도에서는 마커도 조정하거나 마우스로 수정해야 합니다.
- **배경·크기:** `BACKGROUND_FILE`은 배경 경로입니다. 합성 단계의 `bh*0.90`에서 `0.90`이 배경 높이에 대한 인물 높이 비율입니다. 기본 배경은 [제공된 초원 사진](background_images/background.png)입니다.
- **한글 경로:** Python의 `read_bytes`·`write_bytes`와 OpenCV의 `imdecode`·`imencode`를 사용합니다. 앞서 오류가 났던 NumPy의 `tofile`은 사용하지 않습니다.

입력 영상은 `background_images`, 결과는 `background_results`에 있습니다. 실행 또는 `G`를 누를 때 다음 파일을 갱신합니다.

| 파일 | 내용 |
|---|---|
| [01_background.png](background_results/01_background.png) | 원래 배경 |
| [02_original.png](background_results/02_original.png) | 원본 본인 사진 |
| [03_result.png](background_results/03_result.png) | 배경을 바꾼 결과 |
| [04_mask.png](background_results/04_mask.png) | GrabCut 전경 마스크(흰색=전경) |

코드를 짧게 읽을 수 있도록 실행 옵션과 여러 보조 함수는 없애고 `#%%` 단계별로 정리했습니다. 다른 사진·배경은 파일 위쪽의 `INPUT_FILE`·`BACKGROUND_FILE`에서 바꾸세요.

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
