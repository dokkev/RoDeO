# Command for one motion task instance.

string name

# Desired references.
float64[] desired_position
float64[] desired_velocity
float64[] desired_acceleration

# Optional gain and weight overrides.
float64[] kp
float64[] kd
float64[] ki
float64[] weight
