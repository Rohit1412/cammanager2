from setuptools import setup, find_packages

setup(
    name="cam_api",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "opencv-python-headless",
        "numpy",
        "psutil",
        "pytest",
        "pytest-asyncio",
        "pytest-cov"
    ],
) 