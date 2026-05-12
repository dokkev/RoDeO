# Testing

This repository lives inside a larger colcon workspace:

```text
repo path:   /home/dk/workspace/rpc_ws/src/rpc_ros
workspace:   /home/dk/workspace/rpc_ws
```

Always run colcon commands from the workspace root, never from
`src/rpc_ros`.

## Test Commands

Focused build for the current WBC stack:

```bash
cd /home/dk/workspace/rpc_ws
colcon build --base-paths src/rpc_ros \
  --packages-select wbc_core control_architecture wbc_ros optimo_controller
```

Focused gtests for core and control architecture:

```bash
cd /home/dk/workspace/rpc_ws
colcon test --base-paths src/rpc_ros \
  --packages-select wbc_core control_architecture \
  --ctest-args -R "test_" --output-on-failure
colcon test-result --verbose
```

Single package examples:

```bash
cd /home/dk/workspace/rpc_ws
colcon test --base-paths src/rpc_ros \
  --packages-select wbc_core \
  --ctest-args -R "test_id_hqp|test_wbc_solver_semantics|test_solver_baselines" \
  --output-on-failure

colcon test --base-paths src/rpc_ros \
  --packages-select control_architecture \
  --ctest-args -R "test_architecture" --output-on-failure

```

Full repository build, when broader drift is expected:

```bash
cd /home/dk/workspace/rpc_ws
colcon build --base-paths src/rpc_ros
```

Full repository test:

```bash
cd /home/dk/workspace/rpc_ws
colcon test --base-paths src/rpc_ros --ctest-args --output-on-failure
colcon test-result --verbose
```

## Current Test Surface

| Area | Tests | Notes |
| --- | --- | --- |
| `wbc_core` | `test_id_hqp`, `test_wbc_solver_semantics`, `test_solver_baselines` | Covers IDHQP behavior and solver semantics. |
| `control_architecture` | `test_architecture` | Covers config/runtime/FSM hierarchy wiring. |
| `wbc_ros` | No dedicated gtest currently | Needs ROS controller configure/activate/update smoke coverage. |
| `controller/optimo_controller` | No dedicated gtest currently | Should be validated through plugin loading and FSM profile registration smoke tests. |

## Test Strategy

- Unit tests should protect behavior and edge cases, not implementation trivia.
- Regression tests should be small and directly tied to the bug or risk.
- Integration smoke checks should verify YAML parsing, runtime assembly,
  state registration, controller plugin loading, configure/activate/update, and
  representative command output.
- Failure-path tests are expected for YAML errors, missing state names, invalid
  dimensions, non-finite robot state, solver failure, missing interfaces, and
  runtime faults.
- Hot-loop changes should prefer tests or smoke checks that catch allocation,
  stale command output, repeated logging, and state/command dimension drift.

## Coverage Expectations

| Area | Expected Coverage | Notes |
| --- | --- | --- |
| YAML schema and compilation | Positive and negative config tests | Unknown fields and legacy fields should fail early. |
| Runtime assembly | Config to `RuntimeConfig`, registry binding, state assembly | Keep object ownership and non-owning references explicit. |
| FSM hierarchy | State task/contact selection, level override, weight override, transition behavior | This is the main user-facing composition contract. |
| IDHQP | Solver success/failure, qddot/lambda/tau decode, qddot_ref on/off, torque limits | Use TSID models as baseline assets where useful. |
| MPPI core | Shape checks, deterministic sampling, rollout propagation, cost accumulation, receding-horizon shift | Keep ROS message conversion outside generic core. |
| ROS controller wrapper | Plugin load, lifecycle, interface mapping, safe fault command | Still a gap. Prefer a focused smoke test before broad launch tests. |

## Known Gaps

- Full lint/style validation may expose existing style debt. Do not confuse
  lint debt with gtest behavior regressions, but do not ignore it forever.
- There is no dedicated `wbc_ros::WholeBodyController` lifecycle smoke test yet.
- There is no automated guard that fails when `build/`, `install/`, or `log/`
  appears under `src/rpc_ros`.
- Simulation and hardware smoke commands are not yet documented here.

## Adding Tests

Before adding tests:

1. Identify the behavior contract.
2. Find the closest existing test style.
3. Add the narrowest test that fails for the intended regression.
4. Run the smallest relevant command and record any broader validation that was
   not exercised.
