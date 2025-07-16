"""
语音筛选Pipeline包安装脚本
"""

from setuptools import setup, find_packages

# 读取README文件
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# 读取requirements文件
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="speech-filter",
    version="1.0.0",
    author="Speech Filter Team",
    author_email="contact@speechfilter.com",
    description="一个基于多AI模型的语音筛选工具",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/speech-filter",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.990",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "speech-filter=speech_filter.main:main",
        ],
    },
    keywords="speech audio ai machine-learning vad whisper mos",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/speech-filter/issues",
        "Source": "https://github.com/yourusername/speech-filter",
        "Documentation": "https://github.com/yourusername/speech-filter#readme",
    },
) 