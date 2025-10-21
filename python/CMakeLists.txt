cmake_minimum_required(VERSION 3.12)
project(orbslam3_python)

# Find pybind11
find_package(pybind11 REQUIRED)

# Find Python
find_package(Python COMPONENTS Interpreter Development REQUIRED)

find_package(OpenCV 4.4)
   if(NOT OpenCV_FOUND)
      message(FATAL_ERROR "OpenCV > 4.4 not found.")
   endif()

find_package(Eigen3 3.1.0 REQUIRED)

# Include directories
include_directories(
    ${PROJECT_SOURCE_DIR}/../
    ${PROJECT_SOURCE_DIR}/../include
    ${PROJECT_SOURCE_DIR}/../include/CameraModels
    ${PROJECT_SOURCE_DIR}/../Thirdparty/Sophus
    ${EIGEN3_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

# Create the Python module
pybind11_add_module(orbslam3
    orbslam3_module.cpp
    orbslam3_python.h
)

# Link against the main ORB_SLAM3 library
target_link_libraries(orbslam3 PRIVATE
    ${PROJECT_SOURCE_DIR}/../lib/libORB_SLAM3.so
    ${OpenCV_LIBS}
    ${EIGEN3_LIBS}
    ${PROJECT_SOURCE_DIR}/../Thirdparty/DBoW2/lib/libDBoW2.so
    ${PROJECT_SOURCE_DIR}/../Thirdparty/g2o/lib/libg2o.so
    -lboost_serialization
)

# Set the output directory
set_target_properties(orbslam3 PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/../python
)

# Compiler flags
target_compile_definitions(orbslam3 PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
target_compile_features(orbslam3 PRIVATE cxx_std_14)
