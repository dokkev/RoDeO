# Full WBC pipeline state snapshot for monitoring/debugging.

std_msgs/Header header
int32 state_id

# --- Task references (_des) ---
float64[] q_des
float64[] qdot_des

# --- Measured state ---
float64[] q_curr
float64[] qdot_curr

# --- IK output (_cmd) from FindConfiguration ---
float64[] q_cmd
float64[] qdot_cmd
float64[] qddot_cmd

# --- Dynamics output from MakeTorque ---
float64[] tau_ff
float64[] tau_fb
float64[] tau
float64[] gravity

# --- Per-task details ---
TaskState[] tasks
