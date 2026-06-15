# Controller Variable Naming Convention

This document defines naming conventions for the robot control architecture and
model-based controller layer. It is not a naming convention for actuator-driver
internals.

The goal is to keep the control pipeline explicit:

```txt
task objective -> solver result -> host command -> hardware interface
  -> robot feedback -> accepted state
```

In an ideal system, several values may be numerically equal. In real robot
software, they still belong to different layers and should not be collapsed.

---

## 1. Core Suffix Rule

Use suffixes to describe the control layer of a value.

```txt
_des      = desired by a task, planner, or controller objective
_sol      = computed by a solver
_cmd      = final model-side command produced by the host controller and
            handed to the hardware interface
_meas     = reported by hardware, sensor, or driver
_est      = estimated or filtered value
_applied  = physically applied ground-truth-like value, rarely observable
```

A `_cmd` value is the lowest-level output of the model-based host controller. It
is produced after task arbitration, solver output processing, host-side command
building, and controller-level limits. It should still be expressed in the
model/controller convention, usually joint-space SI units.

A `_cmd` value is not a motor-driver packet value, actuator-internal value,
encoded CAN value, or physically applied value.

Do not add global suffix rules such as `_prelimit`, `_raw`, `_limited`, or
`_sat`. Temporary intermediate variables used inside command construction do
not need a standardized suffix unless they cross a module boundary or are
logged as part of the controller interface.

Use unsuffixed `q`, `qdot`, `qddot`, and `tau` only for the canonical current
robot state accepted by the host controller.

```txt
q     = current configuration used by model/control
qdot  = current velocity used by model/control
qddot = current acceleration used by model/control, if available
tau   = current effort/torque used by model/control
```

The unsuffixed state may come from measured, calibrated, filtered, or estimated
values. The suffix-free name means the host controller has accepted it as the
current state for this control tick.

---

## 2. Struct Field Naming Rule

If the enclosing type already defines the semantic layer, do not repeat the
same suffix in the field name.

Free variables should use suffixes when needed:

```cpp
Eigen::VectorXd q_cmd;
Eigen::VectorXd qdot_cmd;
Eigen::VectorXd tau_cmd;
```

Fields inside a command struct inherit the command meaning from the struct
type:

```cpp
struct RobotCommand {
  Eigen::VectorXd q;
  Eigen::VectorXd qdot;
  Eigen::VectorXd tau;
};
```

Good:

```cpp
RobotCommand cmd;
cmd.q = q_cmd;
cmd.qdot = qdot_cmd;
cmd.tau = tau_cmd;
```

Bad:

```cpp
struct RobotCommand {
  Eigen::VectorXd q_cmd;
  Eigen::VectorXd qdot_cmd;
  Eigen::VectorXd tau_cmd;
};
```

The exception is a mixed-layer struct such as a debug log, solver report, or
analysis record. If one struct contains desired, solved, commanded, and
measured values together, keep suffixes explicit:

```cpp
struct ControlDebugLog {
  Eigen::VectorXd q_des;
  Eigen::VectorXd q_sol;
  Eigen::VectorXd q_cmd;
  Eigen::VectorXd q_meas;
};
```

Use a separate log/report holder when one object needs to record multiple
layers or command components:

```cpp
struct RobotLogger {
  Eigen::VectorXd qddot_sol;
  Eigen::VectorXd q_cmd;
  Eigen::VectorXd qdot_cmd;
  Eigen::VectorXd tau_ff_cmd;
  Eigen::VectorXd tau_fb_cmd;
  Eigen::VectorXd tau_cmd;
};
```

Rule of thumb:

```txt
q_cmd     = free variable at command layer
cmd.q     = field inside RobotCommand
log.q_cmd = field inside RobotLogger or another mixed-layer record
```

---

## 3. Why Layers Are Separate

A task may desire one value, but a solver may compute another because of
conflicts, priorities, constraints, or limits.

```txt
q_des != q_sol
```

A solver result may then be processed by the host before being sent to the
hardware interface.

```txt
q_sol != q_cmd
```

The robot may not exactly track the command because of delay, compliance,
saturation, backlash, contact, actuator limits, or hardware-interface policy.

```txt
q_cmd != q_meas
```

The intended control pipeline is:

```txt
x_des / q_des
  -> solver
  -> q_sol / qdot_sol / qddot_sol / tau_sol
  -> host command builder and controller-level limits
  -> q_cmd / qdot_cmd / tau_cmd
  -> RobotCommand cmd
  -> hardware interface
  -> robot
  -> q_meas / qdot_meas / tau_meas
  -> host state preprocessing
  -> q / qdot / tau
```

---

## 4. Current State And Feedback Values

Use unsuffixed values for the canonical state used by the controller during the
current control tick.

```cpp
struct RobotState {
  Eigen::VectorXd q;
  Eigen::VectorXd qdot;
  Eigen::VectorXd tau;
};
```

Use `*_meas` for values reported by hardware, sensors, or drivers.

```txt
q_meas
qdot_meas
tau_meas
current_meas
```

Example:

```cpp
const JointFeedback feedback = driver.read();

RobotState state;
state.q = apply_joint_calibration(feedback.q_meas);
state.qdot = velocity_filter.update(feedback.qdot_meas);
state.tau = torque_filter.update(feedback.tau_meas);
```

For many motors, reported torque is estimated from measured current:

```txt
tau_meas ~= current_meas * Kt
```

For geared joints, depending on the driver convention:

```txt
tau_meas ~= current_meas * Kt * gear_ratio
```

Use `*_est` for explicit estimates or filtered intermediate values.

```txt
qdot_est
tau_est
contact_force_est
```

---

## 5. Desired Values

Use `*_des` for values desired by a task, planner, or controller objective.

```txt
x_des
xdot_des
q_des
qdot_des
```

These values represent what the objective wants before arbitration or conflict
resolution.

Good:

```cpp
posture_task.q_des = nominal_posture;
hand_task.x_des = target_hand_pose;
```

Bad:

```cpp
q_des = solve_wbc(problem);  // Bad: solver output is not a desired value.
```

---

## 6. Solver Results

Use `*_sol` for raw results from solvers such as IK, QP, WBC, MPC, inverse
dynamics, or trajectory optimization.

```txt
q_sol
qdot_sol
qddot_sol
tau_sol
contact_force_sol
```

Meaning:

```txt
The solver computed this value.
```

It does not mean the value was sent to hardware, accepted by the driver,
physically applied, or measured back.

Mixed-layer solution/report structs may keep suffixes in their fields:

```cpp
struct IDSolution {
  Eigen::VectorXd qddot_ref;
  Eigen::VectorXd qddot_sol;
  Eigen::VectorXd q_cmd;
  Eigen::VectorXd tau_cmd;
};
```

---

## 7. From WBC Acceleration To RobotCommand

In acceleration-based WBC, the solver commonly outputs:

```txt
qddot_sol
```

The host controller converts this solver result into torque, velocity, and
position commands.

Typical flow:

```txt
qddot_sol
  -> inverse dynamics
  -> tau_ff_cmd

qddot_sol
  -> host-side integration and controller-level limits
  -> qdot_cmd, q_cmd

q_cmd, qdot_cmd, q, qdot
  -> optional host-side feedback
  -> tau_fb_cmd

tau_ff_cmd + tau_fb_cmd
  -> tau_cmd

q_cmd, qdot_cmd, tau_cmd
  -> RobotCommand cmd
  -> hardware interface

qddot_sol, q_cmd, qdot_cmd, tau_ff_cmd, tau_fb_cmd, tau_cmd
  -> RobotLogger log

command_kp, command_kd
  -> hardware interface parameters
```

---

## 8. Feedforward Torque Command

Compute model-based feedforward torque from `qddot_sol` using inverse dynamics:

```txt
tau_ff_cmd = M(q) qddot_sol + h(q, qdot)
```

where:

```txt
h(q, qdot) = C(q, qdot) qdot + g(q)
```

In Pinocchio:

```cpp
tau_ff_cmd = pinocchio::rnea(model, data, q, qdot, qddot_sol);
```

Meaning:

```txt
tau_ff_cmd = pure model-based feedforward torque command
```

Do not include feedback inside `tau_ff_cmd`.

If contact force compensation is explicitly modeled, the controller may include
a contact term depending on sign convention:

```txt
tau_ff_cmd = M(q) qddot_sol + h(q, qdot) - J_c(q)^T f_c
```

Only include this term when contact force compensation is intentionally part of
the controller.

---

## 9. Host-Side Integration To Position And Velocity Commands

The host may integrate `qddot_sol` to produce model-side position and velocity
commands.

The integrated values should be clamped before being stored as `_cmd` values:

```cpp
qdot_cmd = clamp_velocity(qdot_cmd_prev + dt * qddot_sol);
q_cmd = clamp_position(q_cmd_prev + dt * qdot_cmd);
```

or, if integrating from the current accepted robot state:

```cpp
qdot_cmd = clamp_velocity(state.qdot + dt * qddot_sol);
q_cmd = clamp_position(state.q + dt * qdot_cmd);
```

Meaning:

```txt
q_cmd    = model-side position command handed to the hardware interface
qdot_cmd = model-side velocity command handed to the hardware interface
```

The `_cmd` values should be the actual model-side values placed into
`RobotCommand`.

---

## 10. Optional Host-Side Feedback Torque

The host may optionally compute a tracking feedback torque:

```txt
tau_fb_cmd = kp_fb * (q_cmd - q)
           + kd_fb * (qdot_cmd - qdot)
```

where:

```txt
kp_fb = host-side feedback proportional gain
kd_fb = host-side feedback derivative gain
```

Then:

```txt
tau_cmd = tau_ff_cmd + tau_fb_cmd
```

In some modes:

```txt
tau_fb_cmd = 0
```

This is appropriate when:

- the model-based torque is intended to be used alone
- the embedded driver handles all impedance tracking
- the user does not want host-side tracking feedback
- the model is assumed to be accurate enough for the task

Naming rule:

```txt
tau_ff_cmd = model-based feedforward torque command
tau_fb_cmd = optional host-side feedback torque command
tau_cmd    = final model-side torque command produced by the host controller
```

Good:

```cpp
tau_ff_cmd = compute_inverse_dynamics(q, qdot, qddot_sol);
tau_fb_cmd = compute_host_feedback(q_cmd, qdot_cmd, q, qdot);
tau_cmd = tau_ff_cmd + tau_fb_cmd;
```

Bad:

```cpp
tau_ff_cmd = tau_ff_cmd + tau_fb_cmd;  // Bad: no longer pure feedforward.
```

---

## 11. Hardware-Owned Impedance Gains, RobotCommand, And RobotLogger

The host controller builds a `RobotCommand` as the final model-side command
handed to the hardware interface.

Because the struct name already carries the command-layer meaning, its fields
do not repeat the `_cmd` suffix. Hardware-owned impedance gains are not part of
`RobotCommand`.

```cpp
struct RobotCommand {
  Eigen::VectorXd q;
  Eigen::VectorXd qdot;
  Eigen::VectorXd tau;
};
```

Meaning:

```txt
cmd.q    = q_cmd
cmd.qdot = qdot_cmd
cmd.tau  = tau_cmd
```

Driver-local gains should live in the hardware interface configuration, for
example `command_kp` and `command_kd` ROS parameters. They are not solver
outputs and should not be carried through the model-side command payload.

Host-side feedback gains must be named separately:

```txt
kp_fb
kd_fb
```

Good:

```cpp
RobotCommand cmd;
cmd.q = q_cmd;
cmd.qdot = qdot_cmd;
cmd.tau = tau_cmd;
```

Use `RobotLogger` for command trace components that should not be part of the
hardware command holder:

```cpp
RobotLogger log;
log.qddot_sol = qddot_sol;
log.q_cmd = q_cmd;
log.qdot_cmd = qdot_cmd;
log.tau_ff_cmd = tau_ff_cmd;
log.tau_fb_cmd = tau_fb_cmd;
log.tau_cmd = tau_cmd;
```

Bad:

```cpp
cmd.kp = command_kp;  // Bad: hardware parameter stored in RobotCommand.
cmd.kd = command_kd;  // Bad.
```

---

## 12. Hardware Interface Boundary

The `_cmd` layer belongs to the robot control architecture, not to actuator
driver internals.

The hardware interface or actuator driver may still apply additional
conversion, clamping, ramping, saturation, packet encoding, gear-ratio
conversion, or unit conversion before sending the command to the physical motor
driver.

These hardware-interface operations can affect tracking performance. For
example, if the hardware interface clamps or ramps `cmd.q`, `cmd.qdot`, or
`cmd.tau`, the robot may not track the model-based controller command exactly.
If the hardware interface owns impedance gains, changes to those parameters can
also affect tracking without changing `RobotCommand`.

However, exposing every actuator-internal command back to the model-based
controller is not always desirable. Doing so couples the robot control
architecture master class to hardware-interface actuator classes and makes the
controller less hardware-agnostic.

Recommended boundary:

```txt
model-based controller
  -> RobotCommand cmd
  -> hardware interface
  -> actuator-specific conversion / clamp / ramp / encoding
  -> physical driver
```

Use `_cmd` for the model-side command handed to the hardware interface. Use
explicit hardware-specific names only inside the hardware interface or actuator
driver implementation.

A hardware-interface-local payload such as `ActuatorCommand` may carry
`command_kp` and `command_kd` as `kp` and `kd` after the `RobotCommand`
boundary.

---

## 13. Measured Vs Applied Values

Use `*_meas` for values reported by hardware, sensors, or drivers.

Use `*_applied` only for values that represent physically applied
ground-truth-like quantities.

For torque:

```txt
tau_cmd     = model-side torque command handed to the hardware interface
tau_meas    = torque reported or estimated by driver/robot
tau_applied = actual physical joint torque, rarely directly observable
```

In most systems:

```txt
tau_applied is not available
```

Do not assume:

```txt
tau_applied = tau_cmd
```

Do not assume:

```txt
tau_applied = tau_meas
```

unless the measurement source is explicitly calibrated and documented as a
physical ground-truth-like torque measurement.

Good:

```cpp
feedback.tau_meas = driver_feedback.tau_meas;
state.tau = torque_filter.update(feedback.tau_meas);
```

Bad:

```cpp
state.tau_applied = driver_feedback.tau_meas;  // Bad: reported estimate is not ground truth.
```

Use `*_applied` only for values from calibrated physical measurement systems,
external ground-truth sensors, or explicitly validated applied-value reports.

---

## 14. Complete WBC Command Example

```cpp
const WbcSolution solution = solve_wbc(problem, state);

// Solver result.
const Eigen::VectorXd qddot_sol = solution.qddot_sol;

// Pure model-based feedforward torque.
const Eigen::VectorXd tau_ff_cmd =
    pinocchio::rnea(model, data, state.q, state.qdot, qddot_sol);

// Integrate acceleration into host-side command references.
Eigen::VectorXd qdot_cmd =
    clamp_velocity(qdot_cmd_prev + dt * qddot_sol);

Eigen::VectorXd q_cmd =
    clamp_position(q_cmd_prev + dt * qdot_cmd);

// Optional host-side feedback torque.
Eigen::VectorXd tau_fb_cmd = Eigen::VectorXd::Zero(model.nv);

if (use_host_feedback) {
  tau_fb_cmd =
      kp_fb.cwiseProduct(q_cmd - state.q) +
      kd_fb.cwiseProduct(qdot_cmd - state.qdot);
}

// Final model-side torque command produced by the host controller.
Eigen::VectorXd tau_cmd = tau_ff_cmd + tau_fb_cmd;
tau_cmd = clamp_torque(tau_cmd);

// Build model-side robot command.
RobotCommand cmd;
cmd.q = q_cmd;
cmd.qdot = qdot_cmd;
cmd.tau = tau_cmd;

// Record the command-building trace separately.
RobotLogger log;
log.qddot_sol = qddot_sol;
log.q_cmd = q_cmd;
log.qdot_cmd = qdot_cmd;
log.tau_ff_cmd = tau_ff_cmd;
log.tau_fb_cmd = tau_fb_cmd;
log.tau_cmd = tau_cmd;

// Hand off to hardware interface.
if (hardware_interface.write(cmd)) {
  last_sent_command = cmd;
  qdot_cmd_prev = qdot_cmd;
  q_cmd_prev = q_cmd;
}
```

---

## 15. Rule Of Thumb

General suffixes:

```txt
_des       = wanted by task/planner/controller objective
_sol       = solved by optimization or model algorithm
_cmd       = final model-side command produced by host controller
_meas      = reported back by robot, sensor, or driver
_est       = estimated or filtered
_applied   = physically applied ground-truth-like value
q/qdot/tau = accepted current state
```

Torque-specific names:

```txt
tau_ff_cmd = model-based feedforward torque command
tau_fb_cmd = host-side feedback torque command
tau_cmd    = final model-side torque command produced by host controller
tau_meas   = measured/reported torque, often current_meas * Kt
```

Gain-specific names:

```txt
command_kp, command_kd = hardware-interface impedance gain parameters
kp_fb, kd_fb           = host-side feedback gains
```

Struct field rule:

```txt
q_cmd     = free variable at command layer
cmd.q     = field inside RobotCommand
joint.q   = field inside a semantic joint-state struct
log.q_cmd = field inside RobotLogger or another mixed-layer record
```

Avoid adding new global suffixes for:

```txt
_prelimit
_raw
_limited
_sat
_packet
_motor
_driver_internal
```

Anti-patterns:

```cpp
q_cmd = solve_ik(x_des);  // Bad: solver output skipped the _sol layer.
```

Good:

```cpp
q_sol = solve_ik(x_des);
q_cmd = build_command_from_solution(q_sol, state);
```

Bad:

```cpp
tau_ff_cmd = tau_ff_cmd + tau_fb_cmd;  // Bad: feedforward name reused.
```

Good:

```cpp
tau_cmd = tau_ff_cmd + tau_fb_cmd;
```

Bad:

```cpp
state.tau_applied = driver_feedback.tau_meas;  // Bad: measured estimate is not applied ground truth.
```

Good:

```cpp
state.tau = torque_filter.update(driver_feedback.tau_meas);
```
