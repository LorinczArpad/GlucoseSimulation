import subprocess
import sys
from pathlib import Path

def install_requirements(requirements_path):
    if not Path(requirements_path).exists():
        print(f"File '{requirements_path}' not found.")
        return

    with open(requirements_path, "r") as file:
        for line in file:
            package = line.strip()
            if package and not package.startswith("#"):
                print(f"Installing {package}...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    print(f'Installation of {requirements_path} complete!')


if __name__ == "__main__":
    PATH = "requirements.txt"
    install_requirements(PATH)