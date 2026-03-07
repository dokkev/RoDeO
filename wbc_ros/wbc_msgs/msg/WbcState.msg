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

# --- QP diagnostics (WBIC correction QP) ---
bool qp_solved
int32 qp_status
int32 qp_iter
float64 qp_pri_res
float64 qp_dua_res
float64 qp_obj
float64 qp_setup_time_us
float64 qp_solve_time_us

# --- Tracking performance (instantaneous) ---
float64 joint_pos_err_norm
float64 joint_vel_err_norm
float64 joint_pos_err_max
float64 joint_vel_err_max
float64 tau_fb_norm

# --- Per-task details ---
TaskState[] tasks
