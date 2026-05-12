# Repo Harness Hardening

Status: Active

## Scope

Turn the initial harness scaffold into a practical project-specific operating
surface for humans and agents.

This plan covers:

- keeping repo-level architecture docs aligned with the current WBC stack
- documenting real build/test/smoke commands
- recording durable decisions
- converting repeated mistakes into checks, scripts, or tests

## Non-Goals

- Do not redesign WBC architecture inside this plan.
- Do not migrate legacy packages unless a separate implementation plan exists.
- Do not make `AGENTS.md` a large manual; keep details in `docs/`.

## Acceptance Criteria

- `AGENTS.md` points to the durable docs and contains the workspace-root colcon
  rule.
- `docs/ARCHITECTURE.md` names package ownership, dependency direction, runtime
  contracts, and architecture risks.
- `docs/TESTING.md` contains executable colcon commands from the correct
  workspace root.
- `docs/QUALITY.md` records local preferences, patterns to preserve, patterns
  to avoid, known debt, and rules worth automating.
- `docs/DECISIONS.md` contains the major architecture decisions made during the
  IDHQP/control-architecture cleanup.
- At least one mechanical guard is added for a repeated workflow mistake.

## Validation

Use documentation review plus focused smoke commands:

```bash
cd /home/dk/workspace/rpc_ws
colcon build --base-paths src/rpc_ros \
  --packages-select wbc_core control_architecture wbc_ros optimo_controller
colcon test --base-paths src/rpc_ros \
  --packages-select wbc_core control_architecture \
  --ctest-args -R "test_" --output-on-failure
```

## Open Follow-Ups

- Add a guard script or test that flags `build/`, `install/`, or `log/` under
  `src/rpc_ros`.
- Add ROS controller plugin lifecycle smoke coverage for `wbc_ros` and
  `optimo_controller`.
- Decide whether stale package docs should be added for legacy packages or
  whether legacy directories should stay intentionally undocumented.
- Add a small dependency-direction check for forbidden include paths.
