# Status snapshot of one motion task for WBC monitoring.

string name
int32 dim
int32 priority

# Task references (_des)
float64[] x_des
float64[] xdot_des

# Measured state
float64[] x_curr

# Errors
float64[] x_err

# Operational command (qddot_des + kp*e + kd*edot)
float64[] op_cmd

# Gains
float64[] kp
float64[] kd

# Runtime task weight (used by weighted-QP / prioritization).
float64[] weight

# Instantaneous tracking quality metric.
float64 x_err_norm
