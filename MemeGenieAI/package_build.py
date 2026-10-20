"""Archive native builds without losing Unix executable bits or app symlinks."""
from pathlib import Path
import platform
import shutil
import sys
import tarfile


def main():
    root = Path(__file__).resolve().parent
    release = root / 'release'
    target = root / 'artifacts'
    target.mkdir(exist_ok=True)
    name = f'MemeGenieAI-{platform.system()}-{platform.machine()}'
    if sys.platform == 'win32':
        result = shutil.make_archive(str(target / name), 'zip', release, 'MemeGenieAI')
    else:
        folder = 'MemeGenieAI.app' if sys.platform == 'darwin' else 'MemeGenieAI'
        result = target / (name + '.tar.gz')
        with tarfile.open(result, 'w:gz', dereference=False) as archive:
            archive.add(release / folder, arcname=folder)
    print(result)


if __name__ == '__main__':
    main()
