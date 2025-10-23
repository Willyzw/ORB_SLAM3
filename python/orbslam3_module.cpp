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
 * You should have received a copy of the GNU General Public License along with ORB_SLAM3.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#include "orbslam3_python.h"

PYBIND11_MODULE(orbslam3, m) {
    // Initialize numpy
    NDArrayConverter::init_numpy();
    
    py::enum_<ORB_SLAM3::Tracking::eTrackingState>(m, "TrackingState")
        .value("SYSTEM_NOT_READY", ORB_SLAM3::Tracking::eTrackingState::SYSTEM_NOT_READY)
        .value("NO_IMAGES_YET", ORB_SLAM3::Tracking::eTrackingState::NO_IMAGES_YET)
        .value("NOT_INITIALIZED", ORB_SLAM3::Tracking::eTrackingState::NOT_INITIALIZED)
        .value("OK", ORB_SLAM3::Tracking::eTrackingState::OK)
        .value("RECENTLY_LOST", ORB_SLAM3::Tracking::eTrackingState::RECENTLY_LOST)
        .value("LOST", ORB_SLAM3::Tracking::eTrackingState::LOST)
        .value("OK_KLT", ORB_SLAM3::Tracking::eTrackingState::OK_KLT);

    py::enum_<ORB_SLAM3::System::eSensor>(m, "Sensor")
        .value("MONOCULAR", ORB_SLAM3::System::eSensor::MONOCULAR)
        .value("STEREO", ORB_SLAM3::System::eSensor::STEREO)
        .value("RGBD", ORB_SLAM3::System::eSensor::RGBD)
        .value("IMU_MONOCULAR", ORB_SLAM3::System::eSensor::IMU_MONOCULAR)
        .value("IMU_STEREO", ORB_SLAM3::System::eSensor::IMU_STEREO)
        .value("IMU_RGBD", ORB_SLAM3::System::eSensor::IMU_RGBD);

    py::class_<ORBSLAM3Python>(m, "system")
        .def(py::init<std::string, std::string, ORB_SLAM3::System::eSensor>(), 
             py::arg("vocab_file"), py::arg("settings_file"), py::arg("sensor_type"))
        .def("initialize", &ORBSLAM3Python::initialize)
        .def("process_image_mono", &ORBSLAM3Python::processMono, 
             py::arg("image"), py::arg("time_stamp"))
        .def("process_image_stereo", &ORBSLAM3Python::processStereo, 
             py::arg("left_image"), py::arg("right_image"), py::arg("time_stamp"))
        .def("process_image_rgbd", &ORBSLAM3Python::processRGBD, 
             py::arg("image"), py::arg("depth"), py::arg("time_stamp"))
        .def("shutdown", &ORBSLAM3Python::shutdown)
        .def("is_running", &ORBSLAM3Python::isRunning)
        .def("reset", &ORBSLAM3Python::reset)
        .def("set_use_viewer", &ORBSLAM3Python::setUseViewer)
        .def("get_trajectory", &ORBSLAM3Python::getTrajectory)
        .def("get_tracking_state", &ORBSLAM3Python::getTrackingState)
        .def("is_lost", &ORBSLAM3Python::isLost)
        .def("is_finished", &ORBSLAM3Python::isFinished)
        .def("get_current_frame_keypoints", &ORBSLAM3Python::getCurrentFrameKeypoints)
        .def("get_current_frame_mappoints", &ORBSLAM3Python::getCurrentFrameMapPoints)
        .def("get_current_frame_keypoints_detailed", &ORBSLAM3Python::getCurrentFrameKeypointsDetailed)
        .def("get_all_mappoints", &ORBSLAM3Python::getAllMapPoints);
}
