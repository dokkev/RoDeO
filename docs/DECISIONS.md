# Decisions

Record durable engineering decisions that future humans and agents should not
have to rediscover.

## Decision Log

| Date | Decision | Context | Consequences |
| --- | --- | --- | --- |
| 2026-05-10 | Run colcon from `/home/dk/workspace/rpc_ws`, not from `src/rpc_ros`. | The repository is nested inside a larger colcon workspace, and accidental source-local `build/`, `install/`, and `log/` directories caused confusion. | `AGENTS.md` and `docs/TESTING.md` must show workspace-root commands with `--base-paths src/rpc_ros`. |
| 2026-05-10 | Keep `wbc_core` as math/control core, not control architecture. | `wbc_core` was growing YAML, handler, state-machine, and assembly responsibilities. | YAML parsing, FSM wiring, and runtime assembly belong in `control_architecture`; robot behavior belongs in controller packages. |
| 2026-05-10 | Use `wbc::IDHQP`, `wbc::IDProblem`, and `wbc::IDSolution` naming for inverse-dynamics HQP. | The solver path is fundamentally inverse-dynamics QP/HQP, while WBC is the broader application concept. | Names should make the lower-level formulation explicit while allowing WBC-level wrappers above. |
| 2026-05-10 | Always use the HQP cascade policy for IDHQP; YAML selects only the inner QP backend. | TSID solver names can make cascade and backend look like competing concepts. | `solver.type` maps to `SolverHQP` backend names such as `SOLVER_HQP_PROXQP` or eiquadprog variants; users should not choose single-QP versus cascade in YAML. |
| 2026-05-10 | Put `qddot_ref` policy under `controller`, not `solver`. | `qddot_ref` controls the formulation reference, not the numerical backend. | When false, `qddot_ref := 0` and `delta_qddot_sol == qddot_sol`, making the problem a complete IDHQP. |
| 2026-05-10 | Use YAML state `name` as the factory key; reject separate `type` or `implementation` fields. | Name/type/implementation drift made state config harder to read. | State names must be chosen carefully and registered directly in `StateFactory`. |
| 2026-05-10 | Use `STATE_NAME("...")` for concrete state factory names. | C++ class names are not reliably available as stable, clean factory strings. | Each state class declares its intended YAML key explicitly while keeping registration boilerplate small. |
| 2026-05-10 | Register robot-specific control hooks through `RobotControlProfile`, not FSM pluginlib. | Pluginlib adds deployment friction for state machines, and ROS-free usage should remain possible. The hook may grow beyond FSM registration, for example robot-specific command or architecture configuration. | `wbc_ros::WholeBodyController` stays generic; robot controllers derive from it and return a profile in `CreateControlProfile()`. |
| 2026-05-10 | Keep `optimo_controller` as a thin robot-specific plugin adapter over `wbc_ros`. | Duplicating a full ROS controller per robot makes integration too hard for new users, but pluginlib still needs a concrete controller class to load. | `optimo_controller` should expose Optimo states/profile/config as the meaningful surface; the plugin class can stay hidden in the `.cpp`. |
| 2026-05-10 | Keep command output mode out of `wbc_core`; pass a full command payload and let hardware choose channels. | Some hardware needs only torque, while other hardware needs position, velocity, feedforward torque, `kp`, and `kd`. | `LowLevelCommand` carries `q`, `qdot`, `tau`, `kp`, and `kd`; `wbc_ros` maps configured `command_interfaces` to hardware interfaces. |
| 2026-05-10 | Keep `tsid` in-tree as reference/baseline with `tsid/COLCON_IGNORE`. | TSID solver/task conventions are useful while migrating, but should not dominate the active build. | Normal colcon builds skip `tsid`; copied or adapted code must use `wbc` namespace and repo-local contracts. |

## Decision Template

```text
Date:
Status:
Decision:
Context:
Options considered:
Consequences:
Follow-up:
```

## What Belongs Here

- Architecture direction
- Dependency choices
- Public contract changes
- Testing strategy changes
- Operational constraints
- Repeated tradeoffs that affect future work
