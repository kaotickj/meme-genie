"""Build on the target OS: python build.py (dependencies must be installed)."""
from pathlib import Path
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parent
    subprocess.run([
        sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean",
        "--distpath", str(root / "release"),
        "--workpath", str(root / "build" / "native"),
        str(root / "MemeGenieAI.spec"),
    ], cwd=root, check=True)
    print("Build ready in", root / "release")


if __name__ == "__main__":
    main()
