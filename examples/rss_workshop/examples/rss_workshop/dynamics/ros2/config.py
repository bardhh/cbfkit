
# Configuration settings for Ros2 application
# Defaults for dynamics

LOG_LEVEL = "INFO"  # Options: DEBUG, INFO, WARN, ERROR, FATAL
QUEUE_SIZE = 10
MODULE_NAME = "dynamics"

# Controller
CONTROLLER_NAME = "controller_1"
TIMER_INTERVAL = 0.1
CONTROLLER_PARAMS = {'k_p': 1.0, 'epsilon': 0.5}

# Estimator
ESTIMATOR_NAME = "naive"

# Plant
PLANT_NAME = "plant"
PLANT_PARAMS = {'epsilon': 0.5}
INTEGRATOR_NAME = "forward_euler"
DT = 0.01

# Sensor
SENSOR_NAME = "perfect"
SENSOR_PARAMS = {}
