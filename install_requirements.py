import subprocess
import sys

def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

packages = ['numpy', 'scipy', 'matplotlib']

print("Installing required packages...")
for package in packages:
    try:
        install_package(package)
        print(f"✓ Successfully installed {package}")
    except Exception as e:
        print(f"✗ Failed to install {package}: {e}")

print("\nAll packages installed. Now you can run main4.py")