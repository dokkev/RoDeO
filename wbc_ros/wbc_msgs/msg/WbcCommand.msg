# High-level command consumed by WBC controller runtime.

std_msgs/Header header

# Active registry names for this cycle.
string[] active_task_names
string[] active_contact_names
string[] active_force_task_names

# Optional inline task overrides.
MotionTaskCommand[] motion_tasks
ForceTaskCommand[] force_tasks
