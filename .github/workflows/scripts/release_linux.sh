#!/bin/bash

set -e

yum makecache -y
yum install centos-release-scl -y
yum-config-manager --enable rhel-server-rhscl-7-rpms
yum install llvm-toolset-7.0 python3 python3-devel -y

# Python
python3 -m pip install --upgrade pip
python3 -m pip install setuptools wheel twine auditwheel

# Publish
python3 -m pip wheel . -w dist/ --no-deps
twine upload --verbose --skip-existing dist/*
