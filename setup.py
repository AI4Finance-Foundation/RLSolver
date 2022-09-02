from __future__ import annotations

from setuptools import find_packages
from setuptools import setup

# Read requirements.txt, ignore comments
try:
    REQUIRES = list()
    f = open("requirements.txt", "rb")
    for line in f.read().decode("utf-8").split("\n"):
        line = line.strip()
        if "#" in line:
            line = line[: line.find("#")].strip()
        if line:
            REQUIRES.append(line)
except FileNotFoundError:
    print("'requirements.txt' not found!")
    REQUIRES = list()

setup(
    name="RLSolver",
    version="0.0.1",
    include_package_data=True,
    author="Shixun Wu, Xiaoyang Liu, Ming Zhu",
    author_email="zhumingpassional@gmail.com",
    url="https://github.com/AI4Finance-Foundation/RLSolver",
    license="MIT",
    packages=find_packages(),
    install_requires=REQUIRES
    + [
        
    ],
    # install_requires=REQUIRES,
    description="RLSolver: High-performance RL solvers.",
    long_description="Version 0.0.0 notes: High-performance RL solvers",
    # It is developed by `AI4Finance`_. \
    # _AI4Finance: https://github.com/AI4Finance-Foundation",
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
    ],
    keywords="Reinforcement Learning, Solver, Combinatorial optimization, Non-convex optimization, ",
    platform=["any"],
    python_requires=">=3.7",
)
