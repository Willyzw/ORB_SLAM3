#!/bin/bash

# Build script for ORB-SLAM3 Python bindings

set -e

echo "Building ORB-SLAM3 Python bindings..."

# Check if main library was built successfully
if [ ! -f "lib/libORB_SLAM3.so" ]; then
    echo "Error: Main ORB-SLAM3 library not found. Please run ./build.sh first"
    exit 1
fi

# Build the Python bindings
echo "Building Python bindings..."
cd python
mkdir -p build
cd build

# Configure with CMake
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
make -j4
