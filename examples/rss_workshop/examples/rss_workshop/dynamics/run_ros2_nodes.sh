#!/bin/bash

    # Source ROS2 environment
    source /opt/ros/humble/setup.bash

    # Run ROS2 node scripts
    cd ..
python3 dynamics/ros2/plant_model.py &
python3 dynamics/ros2/sensor.py &
python3 dynamics/ros2/estimator.py &
python3 dynamics/ros2/controller.py