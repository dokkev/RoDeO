# Codex Instructions

## Read Order

- Start here for workspace rules.
- Read `docs/ARCHITECTURE.md` before changing package boundaries, runtime
  wiring, public APIs, or dependency direction.
- Read `docs/QUALITY.md` before broad refactors or readability reviews.
- Read `docs/TESTING.md` before running builds, tests, or smoke checks.
- Read `docs/DECISIONS.md` before reversing an architecture choice.
- Use `docs/PLANS.md` for multi-session or risky work.

## Workspace Discipline

- This repository lives at `/home/dk/workspace/rpc_ws/src/rpc_ros`, but the
  colcon workspace root is `/home/dk/workspace/rpc_ws`.
- Always run `colcon build`, `colcon test`, and launch/smoke checks from
  `/home/dk/workspace/rpc_ws`, never from `src/rpc_ros`.
- When building or testing only this repository, use:

```bash
cd /home/dk/workspace/rpc_ws
colcon build --base-paths src/rpc_ros --packages-select <packages>
colcon test --base-paths src/rpc_ros --packages-select <packages>
```

- Before any colcon command, verify the working directory is
  `/home/dk/workspace/rpc_ws`.
- Do not create `build/`, `install/`, or `log/` under
  `/home/dk/workspace/rpc_ws/src/rpc_ros`.
- If those directories appear under `src/rpc_ros`, treat them as accidental
  build artifacts and remove them before continuing.
