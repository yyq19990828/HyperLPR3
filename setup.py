# !/usr/bin/env python
from setuptools import find_packages, setup
import re
from pathlib import Path

# Read version from __init__.py without importing the module
def get_version():
    init_file = Path(__file__).parent / "hyperlpr3" / "__init__.py"
    content = init_file.read_text(encoding="utf-8")
    match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)
    if match:
        return match.group(1)
    raise RuntimeError("Unable to find version string.")

__version__ = get_version()

if __name__ == "__main__":
    setup(
        name="hyperlpr3",
        version=__version__,
        description="vehicle license plate recognition.",
        url="https://github.com/szad670401/HyperLPR",
        author="HyperInspire",
        author_email="tunmxy@163.com",
        keywords="vehicle license plate recognition",
        packages=find_packages(),
        classifiers=[
            "License :: OSI Approved :: Apache Software License",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 3",
        ],
        install_requires=[
            "opencv-python",
            "onnxruntime",
            "tqdm",
            "requests",
            "fastapi",
            "uvicorn",
            "python-multipart",
            "loguru"
        ],
        license="Apache License 2.0",
        zip_safe=False,
        entry_points="""
                [console_scripts]
                lpr3=hyperlpr3.command.cli:cli
            """
    )