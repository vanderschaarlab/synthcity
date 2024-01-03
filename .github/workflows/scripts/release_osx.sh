#!/bin/sh

export MACOSX_DEPLOYMENT_TARGET=10.14

python -m pip install --upgrade pip
pip install setuptools wheel twine auditwheel

python3 setup.py build bdist_wheel --plat-name macosx_10_14_x86_64 --dist-dir wheel
twine upload --skip-existing wheel/*
