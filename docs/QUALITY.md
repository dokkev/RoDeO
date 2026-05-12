# Quality

This file captures review standards and repo-specific quality preferences. Keep
it practical and enforce repeated rules mechanically when possible.

## Review Priorities

1. Correctness and bug risk
2. Maintainability and long-term change cost
3. Readability and local understandability
4. Unnecessary complexity or over-engineering
5. Beginner accessibility
6. API and module boundaries
7. Testability
8. Performance and resource usage
9. Style consistency

## Local Preferences

- Prefer simple, direct code until a real variation point exists.
- Do not add abstraction only because code is duplicated once.
- Do not split files unless the split improves local reasoning, ownership,
  dependency management, or testability.
- Entry points should show high-level flow, major dependencies, important side
  effects, and error handling.
- Helpers should improve clarity, reuse, testability, or separation of concerns.
- Comments should explain intent, tradeoffs, or non-obvious behavior.
- Keep robot-specific behavior in robot-specific packages.
- Keep `wbc_core` free of YAML, ROS controller, and FSM ownership.
- Keep YAML parsing, runtime object creation, and factory lookup out of the
  1 kHz control loop.

## Patterns To Preserve

- `wbc_core -> control_architecture -> wbc_ros -> robot-specific controller`
  dependency direction, read from right to left as "higher layers depend on
  lower layers".
- `wbc_ros::WholeBodyController` as the generic ROS wrapper.
- Robot-specific control hooks through `RobotControlProfile`, without requiring
  FSM pluginlib.
- State factory keys declared by `STATE_NAME("...")`.
- YAML state `name` as the only state implementation key.
- `ControlArchitecture` as orchestration: robot update, FSM update, IDProblem
  build, IDHQP solve, command output.
- `IDProblemRegistry` storing non-owning references to runtime objects owned by
  `RuntimeConfig`.
- Solver configuration where the HQP cascade policy is fixed and YAML selects
  the inner QP backend.

## Patterns To Avoid

- One-function classes or extra files unless they represent a real variation
  point or improve local reasoning.
- Catch-all "builder/assembler/manager" layers that only forward to one other
  function.
- Robot-specific state classes inside `control_architecture`.
- YAML parsing or schema validation inside control-loop code.
- Pluginlib for FSM behavior unless there is a concrete deployment reason.
- Silent stale command output after runtime faults.
- Repeated size checks, heap allocation, or verbose logging in the hot path.
- Creating `build/`, `install/`, or `log/` under `src/rpc_ros`.

## Known Quality Debt

| Area | Debt | Impact | Desired Direction |
| --- | --- | --- | --- |
| Docs | Repo-level docs were bootstrapped after architecture work | Future agents may trust placeholders or miss current decisions | Keep `docs/` synchronized with code and `control_architecture/code_structure.md`. |
| Lint/style | Full lint validation may fail on existing style debt | Full `colcon test` can be noisy even when focused gtests pass | Triage lint debt separately and make new changes cleaner than the surrounding code. |
| ROS smoke coverage | `wbc_ros` and `optimo_controller` lack focused lifecycle smoke tests | Controller plugin regressions may only appear at runtime | Add plugin load/configure/activate/update smoke tests. |
| Legacy packages | `wbc_core_legacy` remains in-tree | Accidental dependencies can revive old architecture | Keep legacy isolated; document any intentional reference use. |
| Runtime config | Some parsed fields may become obsolete as IDHQP evolves | Config surface can grow confusing | Audit unused fields before declaring the YAML schema stable. |
| Force/contact tasks | Force task wiring is still partly aspirational in sample YAML | Users may configure unsupported behavior | Keep unsupported sample sections commented or guard with clear validation. |

## Rules Worth Automating

Move repeated review comments here first. Promote them to lint, tests, or
scripts when they become stable.

- Fail or warn when `src/rpc_ros/build`, `src/rpc_ros/install`, or
  `src/rpc_ros/log` exists.
- Provide a small wrapper script for standard colcon build/test commands from
  `/home/dk/workspace/rpc_ws`.
- Add an include/dependency check for forbidden directions such as
  `wbc_core -> control_architecture` and `control_architecture -> wbc_ros`.
- Add YAML schema regression tests for unknown fields, removed fields, missing
  state names, and unknown task/contact references.
- Add ROS controller smoke tests for plugin loading and lifecycle transitions.
- Add focused hot-loop checks for unexpected allocation/logging where practical.
