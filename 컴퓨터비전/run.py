"""수업 예제를 프로젝트 루트에서 실행하는 메뉴."""

from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent


def main():
    scripts = sorted(p for p in HERE.glob('*.py') if p.name != Path(__file__).name)
    if len(sys.argv) > 1:
        target = Path(sys.argv[1]).name
        if not target.endswith('.py'):
            target += '.py'
        matches = [p for p in scripts if p.name == target]
        if not matches:
            print(f'실행할 예제를 찾을 수 없습니다: {target}')
            return 1
        return subprocess.call([sys.executable, str(matches[0]), *sys.argv[2:]], cwd=HERE.parent)

    while True:
        print('\n=== 컴퓨터 비전 예제 실행 ===')
        for number, script in enumerate(scripts, 1):
            print(f'{number:2}. {script.name}')
        print(' 0. 종료')
        try:
            choice = input('실행할 번호: ').strip()
        except (EOFError, KeyboardInterrupt):
            return 0
        if choice == '0':
            return 0
        if not choice.isdecimal() or not 1 <= int(choice) <= len(scripts):
            print('목록에 있는 번호를 입력하세요.')
            continue
        result = subprocess.call([sys.executable, str(scripts[int(choice) - 1])], cwd=HERE.parent)
        print(f'실행 종료 (종료 코드: {result})')


if __name__ == '__main__':
    raise SystemExit(main())
