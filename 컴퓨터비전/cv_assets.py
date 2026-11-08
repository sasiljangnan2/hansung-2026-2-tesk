"""실습 자료를 코드 폴더 및 상위 폴더에서 읽는 공통 함수."""
from pathlib import Path
import cv2 as cv
import numpy as np

HERE = Path(__file__).resolve().parent


def resolve_asset(filename):
    path = Path(filename)
    candidates = [path] if path.is_absolute() else [HERE / path, HERE.parent / path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    locations = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(
        f"실습 자료 '{path.name}'를 찾을 수 없습니다. 다음 위치 중 하나에 파일을 넣어 주세요:\n{locations}"
    )


def read_image(filename, flags=cv.IMREAD_COLOR):
    """cv.imread와 같은 색상 옵션을 사용하고 한글 경로를 지원한다."""
    path = resolve_asset(filename)
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        raise ValueError(f"이미지 파일이 비어 있습니다: {path}")
    image = cv.imdecode(data, flags)
    if image is None:
        raise ValueError(f"이미지 형식을 읽을 수 없거나 파일이 손상되었습니다: {path}")
    return image


def open_video(source, *args, **kwargs):
    """파일 경로를 찾은 뒤 열고, 숫자 카메라 인덱스는 그대로 사용한다."""
    resolved = str(resolve_asset(source)) if isinstance(source, (str, Path)) else source
    cap = cv.VideoCapture(resolved, *args, **kwargs)
    if not cap.isOpened():
        cap.release()
        raise OSError(f"동영상 또는 카메라를 열 수 없습니다: {resolved}")
    return cap
