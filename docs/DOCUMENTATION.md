# Documentation Guidelines

This document defines how to write maintainable comments and API documentation
for this repository.

The codebase follows Google C++ style through `.clang-format`. Documentation
should be compatible with Doxygen, but it should still read naturally in the
header/source file without requiring generated HTML.

## Goals

- Make public contracts clear at the declaration site.
- Document domain meaning, units, dimensions, ownership, and failure modes.
- Keep implementation comments close to the code they explain.
- Avoid comments that restate the obvious C++ syntax.
- Keep hot-path code readable without jumping through thin helper comments.

## Style Baseline

Use the repository `.clang-format` settings:

- `BasedOnStyle: Google`
- C++17
- 2-space indentation
- 80-column target
- attached braces
- no automatic comment reflow

Prefer clear names over explanatory comments. A comment should explain why a
concept exists, what contract it enforces, or how a domain convention maps to
code.

Good:

```cpp
/// Pinocchio generalized velocity, size nv().
math::Vector v;
```

Bad:

```cpp
/// Stores v.
math::Vector v;
```

## Doxygen Form

Use `///` for public API documentation. This matches Google-style line comments
and remains Doxygen-compatible.

Use `/** ... */` only when a long multi-paragraph block is genuinely clearer.

Preferred Doxygen commands:

```txt
\brief       One-sentence summary.
\param       Function parameter.
\return      Return value.
\throws      Exception type and reason.
\pre         Required condition before call.
\post        Guaranteed condition after call.
\invariant   Class or object invariant.
\note        Important clarification.
\warning     Dangerous or surprising behavior.
```

Use backticks in Markdown prose, but use Doxygen commands inside code comments.

## Files

Every new public header should start with a short file-level comment only when
the filename does not make its role obvious.

```cpp
/// \file robot-system.hpp
/// \brief Pinocchio-backed robot state and model helper.
```

Do not add long history, migration notes, or design essays to source files.
Put durable design context in `docs/ARCHITECTURE.md` or `docs/DECISIONS.md`.

## Classes And Structs

Document public classes and public data structs at the declaration.

For classes, document:

- Ownership
- Lifetime assumptions
- Thread/control-loop assumptions if relevant
- Important invariants

For data structs, document:

- Semantic meaning
- Size/dimension contract
- Units
- Frame convention
- Whether fields are optional

Follow `docs/NAMING.md` for field names. If the struct type already defines the
semantic layer, do not repeat the same suffix in each field name. If a struct
records multiple command components or layers, keep suffixes explicit.

Example:

```cpp
/// User-facing robot state.
///
/// This state stores semantic robot data supplied by the caller. It does not
/// store Pinocchio packed generalized vectors; use `GeneralizedState` for that.
struct RobotState {
  JointState joint;
  std::optional<BaseState> base;
};
```

## Functions

Document public functions when any of these are true:

- The function is part of a public API.
- Parameter meaning is not obvious from the name.
- Dimensions, units, frames, or ownership matter.
- The function can throw.
- The function has preconditions or postconditions.
- The function is part of the 1 kHz control path.

Public API example:

```cpp
/// \brief Updates the fixed-base robot state from joint state.
///
/// \param joint Joint position, velocity, and actuator torque feedback.
/// \throws std::invalid_argument if the robot is floating-base or if `joint`
///         violates the state dimension/finite-value contract.
/// \post `generalized_q()` equals `joint.q`.
/// \post `generalized_v()` equals `joint.qdot`.
void updateState(const JointState& joint);
```

Avoid documenting trivial accessors unless they carry domain meaning:

```cpp
/// \brief Returns actuator torque feedback, size na().
const math::Vector& tau_actuated() const;
```

Do not write:

```cpp
/// Gets the model.
const pinocchio::Model& model() const;
```

## Parameters

Use exact parameter names in `\param` entries.

Good:

```cpp
/// \param tau_actuated Actuator torque feedback, size na().
```

Bad:

```cpp
/// \param tau Torque.
```

When dimensions matter, include them directly:

```cpp
/// \param q Pinocchio generalized configuration, size nq().
/// \param v Pinocchio generalized velocity, size nv().
```

## Returns

Use `\return` for non-obvious values, references, and computed objects.

```cpp
/// \return Generalized actuation force, size nv(). For floating-base robots,
///         the first 6 entries are zero and the tail is actuator torque.
math::Vector generalized_actuation_force() const;
```

## Throws

Document every public API exception with `\throws`.

Use `std::invalid_argument` for invalid caller input such as wrong dimensions,
non-finite values, or an update path that does not match the root joint type.

```cpp
/// \throws std::invalid_argument if `generalized.q.size() != nq()`.
```

Do not rely on comments that only say "may throw"; state the condition.

## Units And Frames

Always document physical units and frames when values represent geometry,
motion, force, torque, time, or gains.

Examples:

```cpp
/// Base pose of the floating root in world frame.
pinocchio::SE3 pose_world_base;

/// Base twist of the floating root in world frame, used as Pinocchio
/// generalized velocity head<6>().
pinocchio::Motion twist_world_base;

/// State timestamp in seconds.
double time() const;
```

Use frame names in identifiers when possible:

```cpp
pose_world_base
twist_world_base
force_world
```

## Robot State Terminology

Use these terms consistently:

```txt
JointState
  joint.q     = joint configuration excluding floating base, size nq_joints()
  joint.qdot  = joint velocity excluding floating base, size nv_joints()
  joint.tau   = actuator torque feedback, size na()

BaseState
  pose_world_base   = floating base pose in world frame
  twist_world_base  = floating base twist in world frame

GeneralizedState
  q = Pinocchio generalized configuration, size nq()
  v = Pinocchio generalized velocity, size nv()
```

Do not call the joint state `q_actuated` unless the API truly excludes passive
or unactuated joints. Prefer `q_joints`, `qdot_joints`, `nq_joints()`, and
`nv_joints()` for the current `RobotSystem` contract.

Do not put actuator torque in `GeneralizedState`. Floating-base systems do not
have actuator torque on the 6D floating root.

## Inline Implementation Comments

Use implementation comments sparingly.

Add a comment when:

- A formula is non-obvious.
- A frame transformation is easy to misuse.
- A control-loop tradeoff is intentional.
- A branch exists for a domain reason not visible in the code.

Do not add comments for simple assignments, getters, or obvious loops.

Good:

```cpp
// Pinocchio free-flyer q stores quaternion as x, y, z, w.
generalized.q.segment<4>(3) << quat.x(), quat.y(), quat.z(), quat.w();
```

Bad:

```cpp
// Set q.
generalized.q = q;
```

## Hot Path Documentation

For 1 kHz control-path code, documentation should support local readability.

- Keep the main state transition visible in the function body.
- Avoid thin helper functions that only hide adjacent assignments.
- Document the contract at the public API, not by scattering comments through
  forwarding helpers.
- If a runtime check remains in the hot path, document why it is worth the
  cost.

## TODO And FIXME

Use TODOs only for concrete follow-up work.

Format:

```cpp
// TODO(owner-or-area): Specific action and reason.
```

Good:

```cpp
// TODO(wbc_core): Split nv_joints() from na() when passive joints are
// represented in RobotState.
```

Bad:

```cpp
// TODO: clean this.
```

## API Replacement

This repository is maintained by a small internal user group, so do not keep
compatibility shims for old internal APIs by default. When an API changes,
update callers and tests in the same change. Add a compatibility layer only
when there is an explicit external release boundary.

## Examples

Examples should compile conceptually and use current API names.

Good:

```cpp
robots::JointState joint;
joint.q = q_joints;
joint.qdot = qdot_joints;
joint.tau = tau_actuated;

robot.setTime(time_sec);
robot.updateState(joint);
```

Floating-base example:

```cpp
robots::BaseState base;
base.pose_world_base = pose_world_base;
base.twist_world_base = twist_world_base;

robot.setTime(time_sec);
robot.updateState(joint, base);
```

Generalized state escape hatch:

```cpp
robots::GeneralizedState generalized;
generalized.q = q;
generalized.v = v;

robot.updateState(generalized, tau_actuated);
```

## Review Checklist

Before merging public C++ API changes:

- Public classes and structs have Doxygen-compatible comments.
- Public functions document dimensions, units, frames, and throws where needed.
- Comments use current names from `docs/NAMING.md`.
- Examples use current APIs.
- The documentation does not restate obvious syntax.
- Generated Doxygen would preserve useful API information.
