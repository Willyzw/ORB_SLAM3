#!/usr/bin/env python3
"""
Example usage of ORB-SLAM3 Python bindings
"""

import numpy as np
import cv2
import orbslam3

def main():
    # Initialize the system
    vocab_file = "Vocabulary/ORBvoc.txt"
    settings_file = "Examples/Monocular/EuRoC.yaml"
    sensor_type = orbslam3.Sensor.MONOCULAR
    
    slam = orbslam3.system(vocab_file, settings_file, sensor_type)
    slam.initialize()
    
    print("ORB-SLAM3 Python bindings initialized successfully!")
    print(f"System running: {slam.is_running()}")
    print(f"Tracking state: {slam.get_tracking_state()}")
    print(f"Is lost: {slam.is_lost()}")
    
    # Example with a dummy image
    dummy_image = cv2.imread("Datasets/MH01/mav0/cam0/data/1403636579813555456.png", cv2.IMREAD_UNCHANGED)
    # dummy_image = np.zeros((480, 752, 3), dtype=np.uint8)
    print(dummy_image.shape)
    timestamp = 1403636579813555456.0
    
    try:
        # Process the image
        pose = slam.process_image_mono(dummy_image, timestamp)
        print(f"Pose matrix shape: {pose.shape}")
        print(f"Pose matrix:\n{pose}")
    except Exception as e:
        print(f"Error processing image: {e}")
    
    # Shutdown
    slam.shutdown()
    print("System shutdown complete")

if __name__ == "__main__":
    main()
