#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from setuptools import setup

setup(
    name='clare',
    python_requires=">=3.7",
    install_requires=[
        "aim==2.0.27",
        "fire",
        "loguru==0.6.0",
        "gym==0.23.1",
        "sklearn==0.0",
        "gtimer",
        "numpy==1.21.6",
        "tianshou==0.4.8",
        "tqdm",
        "mujoco-py<2.2,>=2.1",
        "scikit-learn==1.0.2",
        "scipy==1.7.3",
        "ray==1.12.0",
        "patchelf"
        #"mujoco-py==2.0.2.13",
    ],
)
