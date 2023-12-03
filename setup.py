#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 11:53:26 2023

@author: mtross
"""

from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()


setup(
    name='autoencoder_trainer',
    version='0.1',
    author='Michael Tross',
    author_email='mtross2@huskers.unl.edu',
    packages=find_packages(),
    install_requires=requirements,
    scripts=['scripts/train_autoencoder.py'])