# Architecture

This repository is a whole-body-control stack built around inverse-dynamics
HQP. The documentation here is intentionally a map, not a full API reference.
When a detail changes, update the smallest durable document that future humans
and agents are likely to read.

For the detailed `control_architecture` wiring contract, see
`control_architecture/code_structure.md`.

## System Shape

Primary flow:

```text
RobotState
  -> RobotSystem
  -> ControlArchitecture
  -> FSM state updates task references
  -> IDProblemRegistry snapshots active tasks/contacts
  -> IDHQP solves
  -> IDSolution
  -> CommandAdapter / ROS hardware command
```

Configuration flow:

```text
YAML files
  -> ConfigLoader
  -> ConfigCompiler
  -> ConfigValidator
  -> RuntimeAssembler
  -> RuntimeConfig
  -> BindRegistry + StateMachineAssembler
  -> ControlArchitecture runtime
```

The hot path should contain model update, FSM update, problem snapshot, solve,
and output application. YAML parsing, object creation, factory lookup, and
schema validation belong outside the 1 kHz loop.

## Module Ownership

| Area | Owns | Should Not Own | Notes |
| --- | --- | --- | --- |
| `wbc_core` | Math primitives, tasks, contacts, trajectories, `RobotSystem`, `RobotState`, `IDProblem`, `IDProblemRegistry`, `IDHQP`, solver backends | YAML parsing, ROS controller code, robot-specific FSM states, high-level control architecture | This is the reusable control/math core. Keep it free of robot package dependencies. |
| `mppi_core` | ROS-free MPPI runtime objects, rollout/cost abstractions, object priors, contact and tactile context structs | ROS subscriptions/messages, hardware IO, IDHQP solving, robot-specific task assumptions, motor torque control | This layer proposes bounded delta-q joint-reference updates for impedance-style tracking. It may inform WBC through a future adapter, but v1 should not depend on `wbc_core`. |
| `control_architecture` | YAML loading/compilation/validation, runtime object graph assembly, FSM base types, `StateFactory`, `RobotControlProfile`, `ControlArchitecture` orchestration | Concrete robot states, ROS hardware IO, solver internals | This layer composes core primitives into a runtime. It should know `wbc_core`, not robot packages. |
| `wbc_ros/wbc_ros` | Generic `ros2_control` controller wrapper, ROS lifecycle, hardware state read/write, actuator command adaptation, `RobotControlProfile` hook | Robot-specific state-machine implementation, task math, YAML schema evolution | `WholeBodyController` should remain generic. Robot-specific behavior enters through a thin derived controller. |
| `wbc_ros/wbc_msgs` | ROS message contracts shared by WBC ROS packages | Controller logic or solver logic | Keep messages stable and boring. |
| `wbc_ros/joint_impedance_controller` | Low-level joint impedance command forwarding/tracking for hand-style position references | MPPI rollout logic, WBC task formulation, robot-specific high-level policy | Receives desired joint position/velocity plus impedance gains and feedforward effort. This is a v1 target for MPPI joint-reference output. |
| `controller/plato_mppi` | Plato/NariTouch ROS adapter from `mppi_core` commands to `wbc_msgs/ImpedanceCommands` | Generic MPPI algorithms, hardware interfaces, WBC task formulation | Subscribes to joint state and NariTouch tactile topics, runs the Jenga grasp MPPI policy, and publishes joint impedance commands. |
| `controller/optimo_controller` | Optimo-specific controller plugin adapter, Optimo FSM state classes, Optimo config examples | Forked generic WBC controller logic, core solver logic | The plugin class is only the ROS loader shim; it returns a `RobotControlProfile` configured with Optimo state registration. |
| `controller/draco_controller` | Robot-specific controller packages | Shared control architecture | Check status before relying on these; they may lag the current architecture. |
| `description/*` | Robot URDFs, meshes, and description package resources | Runtime controller behavior | Descriptions are data dependencies for `RobotSystem` and configs. |
| `robot_bringup/*` | Launch and integration wiring | Core control logic | Bringup should compose packages, not own algorithms. |
| `simulation/*` | Simulator integration and smoke environments | Core control logic | Use for runtime checks once the controller path is stable. |
| `hardware_interface/*` | Hardware-specific ROS interfaces | WBC formulation | Keep hardware IO isolated from solver/formulation code. |
| `tsid` | Vendored/reference TSID source and model assets | Active repo architecture decisions | `tsid/COLCON_IGNORE` keeps it out of normal colcon builds. Use as reference/baseline. |
| `wbc_core_legacy` | Historical packages kept for reference during migration | New architecture dependencies | Do not add new dependencies on legacy packages without an explicit decision. |

## Dependency Direction

Intended dependency direction:

```text
robot-specific controller package
  -> wbc_ros
  -> control_architecture
  -> wbc_core

mppi_core
  -> Pinocchio/Eigen
  -> joint impedance adapter/controller (integration layer, not owned by core)

controller/plato_mppi
  -> mppi_core + sdr_grasp_msgs + wbc_msgs + wbc_core::RobotSystem
  -> joint_impedance_controller command topic
```

Detailed runtime dependency:

```text
controller/optimo_controller
  -> RobotControlProfile registers Optimo concrete states
  -> wbc_ros::WholeBodyController owns ROS lifecycle/hardware IO
  -> control_architecture::ControlArchitecture owns runtime orchestration
  -> wbc_core owns RobotSystem, IDProblem, IDHQP, solvers, tasks, contacts
```

Allowed cross-boundary dependencies:

- `control_architecture` may depend on `wbc_core`.
- `wbc_ros` may depend on `control_architecture` and `wbc_core`.
- Robot controller packages may depend on `wbc_ros`,
  `control_architecture`, and `wbc_core`.
- Tests may include lower layers directly to validate contracts.

Disallowed dependencies:

- `wbc_core -> control_architecture`
- `wbc_core -> wbc_ros`
- `wbc_core -> controller/<robot>_controller`
- `control_architecture -> controller/<robot>_controller`
- `control_architecture -> wbc_ros`
- `IDProblemRegistry -> RuntimeConfig`
- `ConfigCompiler -> RobotSystem`
- `ConfigLoader -> RobotSystem`

## Runtime Contracts

### WBC YAML

Primary example:

- `controller/optimo_controller/config/wbc_example.yaml`

Important rules:

- `schema` currently targets `wbc_control_architecture/v1`.
- `controller.qddot_ref: false` means `qddot_ref := 0`, so
  `delta_qddot_sol == qddot_sol`.
- `solver.type` selects the inner QP backend enum used by the fixed HQP
  cascade policy. It is not a YAML switch between HQP and single-QP control.
- `task_pool` owns reusable task definitions.
- `contact_pool` owns reusable contact definitions.
- `state_machine` selects active tasks/contacts and may override task weight or
  level per state.
- YAML state `name` is the `StateFactory` key. Separate `type` or
  `implementation` fields are intentionally not part of the contract.

### FSM Registration

State classes should define a factory name with:

```cpp
STATE_NAME("initialize");
```

Robot-specific controllers expose state registration through a
`RobotControlProfile`:

```cpp
std::make_unique<wbc::RobotControlProfile>(&RegisterOptimoStates);
```

The generic ROS controller calls `CreateControlProfile()`. A robot-specific
controller derives from `wbc_ros::WholeBodyController` and returns its profile.
The FSM itself should not require ROS or pluginlib.

### ROS Controller

`wbc_ros::WholeBodyController` is the generic controller plugin. It owns:

- ROS lifecycle callbacks
- joint state readback
- joint command writeback
- runtime fault handling
- actuator command adaptation
- `ControlArchitecture` lifetime

The WBC command payload carries all standard actuator-facing channels:
`q`, `qdot`, `tau`, `kp`, and `kd`. The hardware/ROS layer decides which
channels to publish through `command_interfaces`; the core does not choose a
"torque only" versus "integrated command" mode.

Robot-specific plugins, such as `optimo_controller::OptimoController`, should be
thin loader shims that return a robot-specific `RobotControlProfile`.

### ID Problem And Solver

`IDProblemRegistry` stores non-owning references to already-created runtime
tasks/contacts and snapshots the active state into an `IDProblem`.

`IDHQP` solves a ready `IDProblem` and returns an `IDSolution` containing
acceleration, contact reaction, torque, and integrated command fields.

The intended solver style is HQP cascade. The YAML solver backend chooses the
inner QP implementation, for example `SOLVER_HQP_PROXQP` when available or an
eiquadprog baseline.

## Architecture Risks

- Documentation drift: `control_architecture/code_structure.md` is currently
  more complete than this repo-level file. Keep both aligned.
- Legacy drift: `wbc_core_legacy` remains in-tree. Do not accidentally revive
  old packages through new dependencies.
- Controller drift: robot-specific controller packages may not all match the
  current `wbc_ros + RobotControlProfile` architecture.
- Test drift: focused gtests exist for core/control architecture behavior, but
  ROS controller smoke coverage is still thin.
- RT drift: keep YAML parsing, runtime allocation, factory lookup, and noisy
  logging out of the control loop.

## When Changing Architecture

- Update this file when package ownership, dependency direction, public
  contracts, or runtime wiring changes.
- Update `control_architecture/code_structure.md` when config/runtime/FSM wiring
  internals change.
- Record durable tradeoffs in `docs/DECISIONS.md`.
- Add tests or checks for rules that should not drift.
