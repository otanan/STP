#!/usr/bin/env python3
"""Setup for pip install stp.

**Author: Jonathan Delgado**

"""
from setuptools import setup, find_packages

setup(
    name='stp',
    version='0.0.1.2',
    description='Stochastic Thermodynamics in Python',
    license='MIT',
    url='https://github.com/otanan/STP',
    author='Jonathan Delgado',
    author_email='jonathan.delgado@uci.edu',
    keywords=['stochastic', 'thermodynamics', 'python', 'information'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.5',
    ],
    download_url='https://github.com/otanan/STP/archive/refs/tags/0.0.1.0.tar.gz',
    # packages=find_packages(),
    packages=[
        'stp',
        'stp.tools',
    ],
    install_requires=[
        # External packages
        'numpy',
        'scipy',
    ],
)

