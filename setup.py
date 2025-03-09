import io
import os
import re

from setuptools import find_packages
from setuptools import setup

with open("requirements.txt") as f:
    install_requires = []
    for l in f.readlines():
        l = l.strip()
        if "git+" in l:
            url, name = l.split("#egg=")
            install_requires.append(f"{name} @ {url}")
        else:
            install_requires.append(l)


def read(filename):
    filename = os.path.join(os.path.dirname(__file__), filename)
    text_type = type("")
    with io.open(filename, mode="r", encoding="utf-8") as fd:
        return re.sub(text_type(r":[a-z]+:`~?(.*?)`"), text_type(r"``\1``"), fd.read())


setup(
    name="subsampling_significance_tests",
    version="0.1.0",
    url="https://github.com/Kallemakela/",
    author="Kalle Mäkelä",
    author_email="kalle@a.a",
    description="Minimal cookiecutter template for Python packages",
    long_description=read("README.md"),
    packages=find_packages(exclude=("tests",)),
    install_requires=install_requires,
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.11",
    ],
)
