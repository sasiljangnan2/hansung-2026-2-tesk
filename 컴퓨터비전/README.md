# 컴퓨터 비전 실행 방법

## VS Code에서 바로 실행

1. VS Code에서 프로젝트 전체 또는 `컴퓨터비전` 폴더를 엽니다.
2. 실행할 예제 파일(예: `ch3_3.py`)을 엽니다.
3. 오른쪽 위의 ▶ **Run Python File** 버튼 또는 `Ctrl+F5`를 누릅니다. 디버깅은 `F5`로 실행합니다.
4. 이미지 창에서 키를 누르면 다음 단계로 넘어갑니다. 마우스 그리기 예제는 이미지 창에서 `q`를 누르면 종료됩니다.

권장 Python/Python Debugger 확장이 필요합니다. 인터프리터를 이미 별도로 선택했다면 `Ctrl+Shift+P` → `Python: Select Interpreter`에서 `컴퓨터비전/.venv/Scripts/python.exe`를 선택하세요.

설정 변경 전에 열어 둔 터미널은 닫고 새 터미널을 열어 주세요. 실행 경로는 이미지 파일이 있는 프로젝트 최상위 폴더로 설정되어 있습니다.

`#%%` 셀은 Jupyter 확장을 설치하고 같은 가상환경을 선택한 뒤 **Run Cell**로 실행할 수 있습니다. 설정 변경 전에 열어 둔 Interactive 창은 닫고 다시 여세요.

파일 선택 메뉴를 사용하려면 `run.py`를 열고 실행하세요. 새 `.py` 파일도 메뉴에 자동으로 표시됩니다.

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
