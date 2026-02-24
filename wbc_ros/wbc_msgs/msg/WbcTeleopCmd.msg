# Teleoperation command with timestamped 6D end-effector pose and joint targets.

std_msgs/Header header

# Pose command mode.
uint8 POSE_MODE_ABSOLUTE=0
uint8 POSE_MODE_DELTA=1
uint8 pose_mode

# 6D pose value (absolute or delta depending on pose_mode).
geometry_msgs/Pose pose

# Optional joint command (name-aligned vectors).
string[] joint_names
float64[] joint_positions
