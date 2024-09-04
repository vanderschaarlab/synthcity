#!/bin/bash

set -e

# Update the package list
apt-get update

# Install necessary packages and build tools
apt-get install -y \
    software-properties-common \
    python3 \
    python3-dev \
    python3-pip \
    build-essential \
    llvm \
    clang \
    lsb-release

# Add the LLVM repository to get the latest version of LLVM (if needed)
wget https://apt.llvm.org/llvm.sh
chmod +x llvm.sh
./llvm.sh 14  # Replace 14 with the required version if necessary

# Upgrade pip to the latest version
python3 -m pip install --upgrade pip

# Install Python packaging tools
python3 -m pip install setuptools wheel twine auditwheel

# Build Python wheels
python3 -m pip wheel . -w dist/ --no-deps

# Publish the built wheels to PyPI
twine upload --verbose --skip-existing dist/*
