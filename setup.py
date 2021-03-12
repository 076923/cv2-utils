from distutils.core import setup
from setuptools import find_packages, setup, Extension


def get_version():
    return '0.0.4'

def get_requirements():
    with open('requirements.txt', 'rU') as fhan:
        requires = [line.strip() for line in fhan.readlines()]
    return requires

def get_long_description():
    with open("README.md", "r") as fh:
        long_description = fh.read()
    return long_description

setup(
    name='python-opencv-utils',
    description='Python OpenCV Utilities',
    version=get_version(),
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests", "images"]),
    license='Apache License 2.0',
    author='Daehee Yun',
    author_email='s076923@gmail.com',
    url='https://github.com/076923/cv2-utils',
    long_description=get_long_description(),
    long_description_content_type="text/markdown", 
    install_requires=get_requirements(),
)
