# Header timestamp should be update time of controller state
std_msgs/Header header
string[] joint_names

# Direct state vectors instead of JointTrajectoryPoints
float64[] position_desired
float64[] velocity_desired
float64[] position_actual
float64[] velocity_actual
float64[] position_error
float64[] velocity_error

# Control parameters
float64[] stiffness
float64[] damping

# Efforts
float64[] effort_desired
float64[] effort_actual
float64[] effort_ff
float64[] effort_fb

# End-effector force
float64[] force_x
float64[] force_y
float64[] force_z
float64[] force_norm