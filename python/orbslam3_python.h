/**
 * This file is part of ORB-SLAM3
 *
 * Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
 * Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
 *
 * ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
 * License as published by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
 * the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with ORB-SLAM3.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef ORBSLAM3_PYTHON_H
#define ORBSLAM3_PYTHON_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <opencv2/opencv.hpp>
#include "System.h"
#include "Tracking.h"

namespace py = pybind11;

class NDArrayConverter {
public:
    static void init_numpy() {
        // Numpy initialization is handled by pybind11 automatically
    }
    
    static cv::Mat numpy_to_mat(py::array_t<unsigned char> input) {
        py::buffer_info buf_info = input.request();
        
        if (buf_info.ndim != 3 && buf_info.ndim != 2) {
            throw std::runtime_error("Number of dimensions must be 2 or 3");
        }
        
        if (buf_info.ptr == nullptr) {
            throw std::runtime_error("Input array is null");
        }
        
        int rows = buf_info.shape[0];
        int cols = buf_info.shape[1];
        int channels = buf_info.ndim == 3 ? buf_info.shape[2] : 1;
        
        // Create a copy of the data to avoid memory issues
        cv::Mat result(rows, cols, CV_8UC(channels));
        memcpy(result.data, buf_info.ptr, rows * cols * channels * sizeof(unsigned char));
        
        return result;
    }
    
    static cv::Mat numpy_to_mat_float(py::array_t<float> input) {
        py::buffer_info buf_info = input.request();
        
        if (buf_info.ndim != 2) {
            throw std::runtime_error("Depth array must be 2D");
        }
        
        if (buf_info.ptr == nullptr) {
            throw std::runtime_error("Input depth array is null");
        }
        
        int rows = buf_info.shape[0];
        int cols = buf_info.shape[1];
        
        // Create a copy of the data to avoid memory issues
        cv::Mat result(rows, cols, CV_32F);
        memcpy(result.data, buf_info.ptr, rows * cols * sizeof(float));
        
        return result;
    }
    
    static py::array_t<unsigned char> mat_to_numpy(const cv::Mat& mat) {
        std::vector<size_t> shape;
        std::vector<size_t> strides;
        
        if (mat.channels() == 1) {
            shape = {static_cast<size_t>(mat.rows), static_cast<size_t>(mat.cols)};
            strides = {static_cast<size_t>(mat.cols), 1};
        } else {
            shape = {static_cast<size_t>(mat.rows), static_cast<size_t>(mat.cols), static_cast<size_t>(mat.channels())};
            strides = {static_cast<size_t>(mat.cols * mat.channels()), static_cast<size_t>(mat.channels()), 1};
        }
        
        return py::array_t<unsigned char>(
            shape,
            strides,
            mat.data
        );
    }
};

class ORBSLAM3Python {
public:
    ORBSLAM3Python(const std::string& vocab_file, const std::string& settings_file, ORB_SLAM3::System::eSensor sensor_type)
        : mSensorType(sensor_type), mVocabFile(vocab_file), mSettingsFile(settings_file), mpSystem(nullptr) {
    }
    
    ~ORBSLAM3Python() {
        if (mpSystem) {
            mpSystem->Shutdown();
            delete mpSystem;
        }
    }
    
    void initialize() {
        if (mpSystem) {
            delete mpSystem;
        }
        mpSystem = new ORB_SLAM3::System(mVocabFile, mSettingsFile, mSensorType);
    }
    
    py::array_t<double> processMono(py::array_t<unsigned char> image, double timestamp) {
        if (!mpSystem) {
            throw std::runtime_error("System not initialized. Call initialize() first.");
        }
        
        cv::Mat cv_image = NDArrayConverter::numpy_to_mat(image);
        Sophus::SE3f pose = mpSystem->TrackMonocular(cv_image, timestamp);
        
        // Convert SE3f to numpy array (4x4 transformation matrix)
        Eigen::Matrix4f transform = pose.matrix();
        
        // Create a copy of the matrix data to avoid issues with temporary objects
        Eigen::Matrix4d transform_double = transform.cast<double>();
        
        // Create numpy array manually to avoid nullptr issues
        py::array_t<double> result = py::array_t<double>({4, 4});
        auto buf = result.mutable_unchecked<2>();
        
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                buf(i, j) = transform_double(i, j);
            }
        }
        
        return result;
    }
    
    py::array_t<double> processStereo(py::array_t<unsigned char> left_image, py::array_t<unsigned char> right_image, double timestamp) {
        if (!mpSystem) {
            throw std::runtime_error("System not initialized. Call initialize() first.");
        }
        
        cv::Mat cv_left = NDArrayConverter::numpy_to_mat(left_image);
        cv::Mat cv_right = NDArrayConverter::numpy_to_mat(right_image);
        Sophus::SE3f pose = mpSystem->TrackStereo(cv_left, cv_right, timestamp);
        
        // Convert SE3f to numpy array (4x4 transformation matrix)
        Eigen::Matrix4f transform = pose.matrix();
        
        // Create a copy of the matrix data to avoid issues with temporary objects
        Eigen::Matrix4d transform_double = transform.cast<double>();
        
        // Create numpy array manually to avoid nullptr issues
        py::array_t<double> result = py::array_t<double>({4, 4});
        auto buf = result.mutable_unchecked<2>();
        
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                buf(i, j) = transform_double(i, j);
            }
        }
        
        return result;
    }
    
    py::array_t<double> processRGBD(py::array_t<unsigned char> image, py::array_t<float> depth, double timestamp) {
        if (!mpSystem) {
            throw std::runtime_error("System not initialized. Call initialize() first.");
        }
        
        cv::Mat cv_image = NDArrayConverter::numpy_to_mat(image);
        cv::Mat cv_depth = NDArrayConverter::numpy_to_mat_float(depth);
        Sophus::SE3f pose = mpSystem->TrackRGBD(cv_image, cv_depth, timestamp);
        
        // Convert SE3f to numpy array (4x4 transformation matrix)
        Eigen::Matrix4f transform = pose.matrix();
        
        // Create a copy of the matrix data to avoid issues with temporary objects
        Eigen::Matrix4d transform_double = transform.cast<double>();
        
        // Create numpy array manually to avoid nullptr issues
        py::array_t<double> result = py::array_t<double>({4, 4});
        auto buf = result.mutable_unchecked<2>();
        
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                buf(i, j) = transform_double(i, j);
            }
        }
        
        return result;
    }
    
    void shutdown() {
        if (mpSystem) {
            mpSystem->Shutdown();
        }
    }
    
    bool isRunning() {
        return mpSystem && !mpSystem->isShutDown();
    }
    
    void reset() {
        if (mpSystem) {
            mpSystem->Reset();
        }
    }
    
    void setUseViewer(bool use_viewer) {
        // This is a placeholder - ORB_SLAM3 doesn't have a direct viewer control in the System class
        // The viewer is typically controlled through the Pangolin library in the examples
    }
    
    std::vector<std::vector<double>> getTrajectory() {
        // This is a placeholder - would need to implement trajectory extraction
        // For now, return empty vector
        return std::vector<std::vector<double>>();
    }
    
    int getTrackingState() {
        if (mpSystem) {
            return mpSystem->GetTrackingState();
        }
        return -1;
    }
    
    bool isLost() {
        if (mpSystem) {
            return mpSystem->isLost();
        }
        return true;
    }
    
    bool isFinished() {
        if (mpSystem) {
            return mpSystem->isFinished();
        }
        return true;
    }
    
    // Get current frame keypoints
    py::array_t<float> getCurrentFrameKeypoints() {
        if (!mpSystem) {
            throw std::runtime_error("System not initialized. Call initialize() first.");
        }
        
        // Get tracked keypoints from the system
        std::vector<cv::KeyPoint> keypoints = mpSystem->GetTrackedKeyPointsUn();
        
        if (keypoints.empty()) {
            return py::array_t<float>(std::vector<size_t>{0, 2});
        }
        
        // Create numpy array for keypoints (x, y coordinates)
        py::array_t<float> result = py::array_t<float>(std::vector<size_t>{static_cast<size_t>(keypoints.size()), 2});
        auto buf = result.mutable_unchecked<2>();
        
        for (size_t i = 0; i < keypoints.size(); i++) {
            buf(i, 0) = keypoints[i].pt.x;
            buf(i, 1) = keypoints[i].pt.y;
        }
        
        return result;
    }
    
    // Get current frame map points with 3D coordinates
    py::array_t<double> getCurrentFrameMapPoints() {
        if (!mpSystem) {
            throw std::runtime_error("System not initialized. Call initialize() first.");
        }
        
        // Get tracked map points from the system
        std::vector<ORB_SLAM3::MapPoint*> mapPoints = mpSystem->GetTrackedMapPoints();
        
        // Count valid map points
        int validCount = 0;
        for (const auto& mp : mapPoints) {
            if (mp && !mp->isBad()) {
                validCount++;
            }
        }
        
        if (validCount == 0) {
            return py::array_t<double>(std::vector<size_t>{0, 3});
        }
        
        // Create numpy array for map points (x, y, z coordinates)
        py::array_t<double> result = py::array_t<double>(std::vector<size_t>{static_cast<size_t>(validCount), 3});
        auto buf = result.mutable_unchecked<2>();
        
        int idx = 0;
        for (const auto& mp : mapPoints) {
            if (mp && !mp->isBad()) {
                Eigen::Vector3f worldPos = mp->GetWorldPos();
                buf(idx, 0) = static_cast<double>(worldPos(0));
                buf(idx, 1) = static_cast<double>(worldPos(1));
                buf(idx, 2) = static_cast<double>(worldPos(2));
                idx++;
            }
        }
        
        return result;
    }
    
    // Get current frame keypoints with additional information (x, y, response, octave)
    py::array_t<float> getCurrentFrameKeypointsDetailed() {
        if (!mpSystem) {
            throw std::runtime_error("System not initialized. Call initialize() first.");
        }
        
        // Get tracked keypoints from the system
        std::vector<cv::KeyPoint> keypoints = mpSystem->GetTrackedKeyPointsUn();
        
        if (keypoints.empty()) {
            return py::array_t<float>(std::vector<size_t>{0, 4});
        }
        
        // Create numpy array for keypoints (x, y, response, octave)
        py::array_t<float> result = py::array_t<float>(std::vector<size_t>{static_cast<size_t>(keypoints.size()), 4});
        auto buf = result.mutable_unchecked<2>();
        
        for (size_t i = 0; i < keypoints.size(); i++) {
            buf(i, 0) = keypoints[i].pt.x;
            buf(i, 1) = keypoints[i].pt.y;
            buf(i, 2) = keypoints[i].response;
            buf(i, 3) = static_cast<float>(keypoints[i].octave);
        }
        
        return result;
    }
    
private:
    ORB_SLAM3::System::eSensor mSensorType;
    std::string mVocabFile;
    std::string mSettingsFile;
    ORB_SLAM3::System* mpSystem;
};

#endif // ORBSLAM3_PYTHON_H
