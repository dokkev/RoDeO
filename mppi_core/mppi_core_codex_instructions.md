# mppi_core Codex Instructions

This document defines the development rules for the `mppi_core` package in the
existing Pinocchio/control workspace.

The current target use case is **PLATOMPPI**:

```text
finger-only MPPI
+ joint-position reference generation through bounded delta-q increments
+ low-level joint impedance tracking
+ NariTouch tactile feedback
+ wrist-independent disturbance rejection
+ one grasp policy before stage-conditioned policy expansion
```

The goal is not to build a generic whole-body MPC framework. The goal is to
build a small, readable, real-time-friendly MPPI core that lets the fingers
autonomously regulate contact quality while the wrist, object, and environment
may behave unpredictably.

---

## North Star

`mppi_core` is a **Pinocchio-based finger reference generator** for
impedance-controlled manipulation.

It proposes short-horizon finger joint references:

```text
q_finger_des
optional qdot_finger_ff derived from delta_q / dt
optional bounded stiffness / gain scale in later versions
```

It does **not** compute motor torques directly. A separate joint impedance
controller tracks the references:

```text
tau = Kp * (q_des - q) + Kd * (v_des - v)
```

The clean runtime interpretation is:

```text
NariTouch observer
  -> slip probability
  -> normal force
  -> contact centroid / contact patch features
  -> contact loss / force spike signals

mppi_core / PLATOMPPI
  -> samples short-horizon finger reference trajectories
  -> predicts finger reference motion and local tactile-risk proxies
  -> evaluates grasp tactile safety costs
  -> outputs q_finger_des and optional v_des

joint_impedance_controller
  -> tracks q_des, with optional bounded velocity feedforward
  -> provides physical compliance
  -> handles motor-level torque/current command
```

The first concrete use case is Jenga-like contact manipulation, but the generic
optimizer should still avoid hard-coding Jenga task logic inside
`MPPIOptimizer`.

Use:

```text
package:   mppi_core
namespace: mppi_core
example:   plato_mppi / jenga_tactile_mppi
```

---

## Initial Scope: Finger-Only MPPI

The initial PLATOMPPI controller controls **only finger joint references**.

Wrist/global hand pose is controlled externally and must not be part of the
MPPI action space in v1.

Do not add these to the v1 MPPI action:

```text
wrist pose
wrist velocity
floating base variables
whole-body variables
object pose
object velocity
```

The MPPI action should initially be a bounded finger joint-reference increment:

```text
u_k = delta_q_finger_ref_k
q_ref_{k+1} = integrate(q_ref_k, u_k)
v_des_{k+1} = u_k / dt   # optional feedforward, not the primary objective
```

For simple revolute-only finger models, vector addition may be acceptable:

```text
q_ref_{k+1} = q_ref_k + u_k
```

Use `pinocchio::integrate()` when the model contains non-Euclidean joints or
when consistent Pinocchio semantics are preferred.

The MPPI output is a finger joint-position reference for a low-level joint
impedance controller.

Preferred v1 output:

```text
q_finger_des
optional v_des
```

Do not implement torque MPPI in v1.

---

## Wrist-Independent Tactile Autonomy

The intended behavior is:

> The wrist may do whatever the higher-level system asks it to do. The fingers
> should autonomously maintain a safe and useful contact condition using tactile
> feedback.

In PLATOMPPI, wrist motion, object-environment interaction, extraction events,
peg-in-hole contact, wrist re-orientation, and support transitions should be
treated as **unmodeled external disturbances**.

Their effects are observed through NariTouch:

```text
normal force changes
normal force rate / force spikes
slip probability changes
contact centroid drift
contact area changes
contact loss
```

The controller's job is local tactile regulation:

```text
maintain enough grip to avoid slip
avoid excessive normal force
keep contact away from tactile patch boundaries
recover from contact loss
handle support-loss events
soften/release during placement when force spikes indicate jamming
```

Do not require wrist pose, wrist velocity, object pose, or object velocity as
mandatory inputs in v1.

A high-level stage or mode label is allowed and encouraged:

```text
Extraction
Capture
Placement
Release
```

This stage label expresses task intent. It is not the same as wrist motion
information.

---

## Disturbance-Rejection View

PLATOMPPI should not predict full object motion.

In the target tasks, the object may experience large unmodeled disturbances from:

```text
extraction
peg-in-hole / insertion
wrist re-orientation
support loss
environmental contact
neighboring object contact
```

Treat these effects as external disturbances whose consequences are observed
through NariTouch.

The controller objective is to keep local tactile/contact features inside a
stage-dependent safe envelope:

```text
normal force inside a safe window
slip probability below threshold
contact centroid away from sensor boundaries
contact area valid
force spikes handled safely
contact loss handled through reflex or recovery behavior
```

The rollout should predict:

```text
finger reference motion
finger kinematic consequences
local tactile-risk proxy evolution
```

The rollout should not predict:

```text
full object pose trajectory
full object velocity trajectory
full hand-object-environment contact dynamics
wrist trajectory
```

A good mental model:

```text
not:
  object trajectory MPC

but:
  disturbance-rejecting tactile contact governor
```

---

## NariTouch Assumption

For the initial PLATOMPPI implementation, NariTouch is the fixed tactile sensing
interface.

Generic tactile abstraction is not a v1 goal.

It is acceptable for `mppi_core` to include NariTouch-specific data structures
and utilities, including:

```text
NariTouch node geometry
normal-force aggregation
slip probability features
contact centroid computation
contact patch / contact area features
contact-valid flags
force-spike/contact-loss detection helpers
```

However, keep naming honest. If a type is NariTouch-specific, say so.

Prefer:

```cpp
NariTouchState
NariTouchNodeState
NariTouchLayout
ComputeNariTouchContactCentroid()
ComputeNariTouchNormalForce()
```

Avoid pretending sensor-specific code is generic:

```cpp
TactileSensorState  // avoid if it is really NariTouch-only
GenericContactState // avoid unless it is actually generic
```

A useful distinction:

```text
Allowed in mppi_core/contact:
  NariTouch sensing representation and feature extraction

Better in examples/adapters/stage policies:
  Jenga-specific transition logic
  extraction/placement task logic
  block-slot task assumptions
```

NariTouch is the contact-state observer. It is acceptable for the first
implementation to depend on it.

---

## Relationship to Existing Packages

### `wbc_core`

`mppi_core` should not depend on `wbc_core` in v1.

The initial dependency set should be:

```text
Pinocchio
Eigen
standard C++
ament_cmake
```

`wbc_core` owns:

```text
inverse dynamics
inverse kinematics
HQP formulations
IDProblem / IDSolution semantics
strict-priority whole-body feasibility
```

`mppi_core` must not:

```text
create or solve IDProblem
own HQP hierarchy semantics
duplicate IDHQP physics constraints
compute whole-body torques
introduce object dynamics into wbc_core::IDHQP
```

If WBC integration becomes useful later, add a thin adapter that translates MPPI
references into the WBC/control-architecture contract.

Do not put a `wbc_core` dependency into the generic MPPI core for v1.

### `control_architecture`

`control_architecture` may later assemble or configure `mppi_core`, but
`mppi_core` should not parse global robot behavior YAML directly.

Allowed:

```text
integration layer
  -> creates MPPI controller from compiled config structs
  -> passes measured finger state, current q_des, NariTouch state, and stage
     context into mppi_core every control cycle
```

Not allowed:

```text
mppi_core reads global behavior YAML
mppi_core constructs the full runtime architecture
mppi_core owns hardware interfaces
mppi_core publishes ROS messages directly
```

---

## Core Design Philosophy

### 1. Finger reference rollout first

The first version must use robot-only, finger-only reference rollout.

Default rollout state:

```text
x = [q_finger_ref, v_finger_ref, optional internal/reference state]
```

The measured robot state enters the observation/context:

```text
q_finger_measured
v_finger_measured
current q_finger_des
current v_finger_des
NariTouch features
stage/mode label
```

Rollout initial state should usually be the current commanded reference, not the
measured robot state:

```text
rollout initial q_ref = current commanded q_des
observation           = q_measured, v_measured, NariTouch
```

Keep tracking error visible:

```text
tracking_error = q_ref_current - q_measured
```

If tracking error is too large, cost terms should penalize aggressive new
references.

### 2. Object model as guardrail, not oracle

For Jenga-like manipulation, the controller should not assume that future object
pose can be accurately rolled out.

Known object geometry and inertial parameters may define costs, thresholds, and
safe-force windows:

```text
ObjectPrior
  mass
  dimensions
  approximate inertia
  friction prior
  shape/type metadata
```

The object prior is a parameter. It is not the default rollout state.

Use it as a guardrail:

```text
minimum grip force estimate
maximum safe normal force
stage-dependent force window
approximate contact feasibility
risk thresholds
```

Do not make object pose mandatory for v1.

### 3. MPPI proposes; impedance tracks

MPPI proposes finger joint references. Joint impedance control provides local
tracking and compliance.

For v1, MPPI may output:

```text
desired finger joint-reference increment
desired finger joint-position reference
optional bounded finger joint-velocity feedforward
```

Optional later outputs:

```text
bounded stiffness scale
bounded damping scale
reflex status / safety flag
```

Keep impedance gains fixed in the first implementation. Stage-conditioned gain
scaling may be added later, but it should remain explicit, bounded, and easy to
disable.

### 4. Keep hard real-time concerns visible

The initial version may be readable before being fully optimized. Still, avoid
patterns that obviously prevent future real-time use.

Prefer:

```text
preallocated buffers
dimension-cached Eigen objects
explicit resize/initialize phases
no allocations inside rollout hot loops after initialization
no filesystem access in the control loop
no YAML parsing in the control loop
deterministic update APIs
frame-name lookup only at initialization
```

Avoid:

```text
hidden global state
ownership ambiguity
dynamic allocation in rollout inner loops
dynamic dispatch-heavy cost evaluation in final hot paths
ROS publishers/subscribers inside mppi_core
sensor reading inside MPPIOptimizer
hardware ownership inside MPPIOptimizer
```

---

## Recommended Package Layout

Create or maintain a ROS 2 ament package:

```text
mppi_core/
  CMakeLists.txt
  package.xml
  README.md
  AGENTS.md

  include/mppi_core/
    grasp_types.hpp

    core/
      mppi_optimizer.hpp
      mppi_config.hpp
      action_sequence.hpp
      sampling_policy.hpp
      rollout_buffer.hpp
      rollout_result.hpp
      trajectory_buffer.hpp

    model/
      rollout.hpp
      pinocchio_rollout_model.hpp
      tactile_proxy_model.hpp

    costs/
      cost_term_base.hpp
      cost_stack.hpp
      action_smoothness_cost.hpp
      joint_limit_cost.hpp
      tracking_guard_cost.hpp
      normal_force_window_cost.hpp
      slip_risk_cost.hpp
      contact_centroid_cost.hpp
      contact_loss_risk_cost.hpp
      force_spike_cost.hpp

    contact/
      naritouch.hpp
      contact_belief.hpp

    stages/
      stage_id.hpp
      stage_policy.hpp
      stage_manager.hpp

    reflex/
      reflex_layer.hpp
      slip_reflex.hpp
      contact_loss_reflex.hpp
      force_spike_reflex.hpp

    interface/
      mppi_controller.hpp
      observation.hpp
      task_context.hpp
      command.hpp

    utils/
      random_utils.hpp
      timing.hpp
      eigen_checks.hpp

  src/
    core/
    model/
    costs/
    contact/
    stages/
    reflex/
    interface/
    utils/

  config/
    mppi.yaml
    stage_policy.yaml
    object_prior.yaml

  test/
    test_mppi_shapes.cpp
    test_sampling_policy.cpp
    test_cost_stack.cpp
    test_shift_nominal.cpp
    test_naritouch.cpp
    test_pinocchio_rollout.cpp

  examples/
    plato_mppi/
      README.md
      config/
```

Keep task examples separate from generic optimizer internals. Jenga-specific
stage transition logic should live in examples, adapters, or stage policies, not
inside `MPPIOptimizer`.

---

## Naming Rules

Use clear, explicit names.

Good:

```text
MPPIOptimizer
SamplingPolicy
ActionSequence
RolloutBuffer
PinocchioRolloutModel
ObjectPrior
NariTouchState
NariTouchLayout
ContactBelief
StagePolicy
CostStack
NormalForceWindowCost
SlipRiskCost
ContactCentroidCost
ForceSpikeCost
```

Bad:

```text
Manager
Handler
Processor
Thing
Controller2
MPPIStuff
GenericTactileStateThatIsActuallyNariTouch
```

Do not use names that imply object-state prediction unless the class really
performs object-state rollout.

Avoid:

```text
ObjectDynamicsRollout
ObjectStateMPC
FullObjectPredictor
```

Prefer:

```text
ObjectPrior
ContactBelief
TactileRiskProxy
NariTouchFeatures
```

---

## Namespace and Include Style

Use:

```cpp
namespace mppi_core {
}
```

Headers should be included as:

```cpp
#include "mppi_core/core/mppi_optimizer.hpp"
```

Follow the workspace `.clang-format`:

```text
C++17
Google-derived style
2-space indentation
80-column target
attached braces
pointer/reference alignment as configured
```

Use `EIGEN_MAKE_ALIGNED_OPERATOR_NEW` for classes/structs storing Eigen
fixed-size vectorizable members.

---

## Dependency Rules

Initial dependencies should be minimal:

```text
ament_cmake
Eigen3
pinocchio
```

Do not add a `wbc_core` dependency for v1.

Optional later dependencies:

```text
yaml-cpp      # config/adapters only, not control hot loop
rclcpp        # separate ROS wrapper if needed
proxsuite     # only if a future MPPI variant needs QP subproblems
```

Do not add CUDA, Torch, MuJoCo, or Python runtime dependencies to the initial
C++ core unless explicitly requested.

GPU acceleration can be a later backend, not the baseline architecture.

---

## API Boundary

The core API should look like a deterministic receding-horizon reference
generator:

```text
initialize(config, model)
update(observation, task_context) -> command/reference
reset()
shiftNominalTrajectory()
```

Recommended conceptual flow:

```text
Observation + TaskContext
  -> build NariTouch feature / contact belief
  -> sample finger action sequences around nominal sequence
  -> rollout finger reference model
  -> predict local tactile-risk proxy if enabled
  -> evaluate grasp cost
  -> compute MPPI weights
  -> update nominal action sequence
  -> return first q_des command and optional v_des
  -> shift horizon
```

Do not make `MPPIOptimizer` read sensors directly.

Do not make `MPPIOptimizer` publish ROS messages directly.

Do not make `MPPIOptimizer` own hardware interfaces.

Do not make wrist/object state mandatory for v1.

---

## Observation and Context

Recommended v1 observation:

```cpp
struct GraspObservation {
  Eigen::VectorXd q_measured;
  Eigen::VectorXd v_measured;

  Eigen::VectorXd q_ref_current;
  Eigen::VectorXd v_ref_current;

  NariTouchState tactile;

  bool has_gravity_context{false};
  Eigen::Vector3d gravity_in_sensor_frame{Eigen::Vector3d::Zero()};

  double time_s{0.0};
};
```

Do not add stage labels to v1 grasp. Stage policies can be added
later as a layer above this baseline policy.

Do not require:

```text
wrist pose
wrist velocity
object pose
object velocity
```

as mandatory fields in v1 observation/context.

---

## Rollout Rules

### Generic rollout model

`RolloutModelBase` should define:

```text
state_next = step(state, action, context, dt)
```

The first implementation should be Pinocchio-backed and finger-only.

### Pinocchio usage

The rollout model may use:

```text
forward kinematics
frame placements
frame velocities
Jacobians
simple joint-space integration
```

For v1, do not implement full rigid-body contact simulation.

The aim is not physics perfection. The aim is to let MPPI reason about how
candidate finger references move the finger geometry and tactile contact region
over a short horizon.

### Integration

Preferred v1 action semantics:

```text
action_k = delta_q_ref_cmd_k
q_ref_{k+1} = pinocchio::integrate(q_ref_k, action_k)
qdot_ff_{k+1} = action_k / dt   # optional feedforward only
```

This avoids making multi-finger velocity tracking the core control problem.
MPPI plans bounded position-reference increments; the impedance controller
handles local tracking and compliance.

### Tracked frames

Pinocchio rollout should optionally expose tracked finger/fingertip frame data
for cost evaluation:

```text
tracked frame placement
tracked frame linear velocity
tracked frame angular velocity
optional Jacobian
```

Frame names must be resolved to Pinocchio frame IDs during initialization, not
inside the rollout loop.

Rollout hot loops must not perform:

```text
filesystem access
YAML parsing
frame-name lookup
dynamic allocation after initialization
```

---

## Tactile-Risk Proxy Rollout

Because wrist/object/environment disturbances are unmodeled, do not pretend to
predict exact future tactile measurements.

Instead, v1 may use local directional risk proxies.

Examples:

```text
closing action
  -> predicted normal force tends to increase

opening action
  -> predicted normal force tends to decrease

high slip + opening action
  -> high slip risk

high normal force + closing action
  -> high overgrip / jam risk

contact centroid near boundary + action likely moving further outward
  -> high contact-loss risk

force spike + continued closing
  -> high jam/overconstraint risk
```

This model should be documented as a risk proxy, not as a physical object
dynamics model.

Potential type name:

```cpp
TactileRiskProxy
```

or:

```cpp
TactileProxyModel
```

Do not name it `ObjectDynamicsModel`.

---

## Sampling Rules

Use hot-started receding-horizon action sequences.

Recommended baseline:

```text
nominal action sequence U
sample noise around U
evaluate K rollouts
weight rollouts by exp-normalized cost
update U
return U[0]
shift U
```

Prefer smooth action sampling for contact tasks:

```text
num_horizon_steps: 20-50
num_knots: 4-8
sample knot actions
interpolate action sequence
```

This avoids high-frequency contact chatter.

Keep the sampling backend modular:

```text
GaussianSamplingPolicy
SplineSamplingPolicy
RiskSensitiveSamplingPolicy
```

### Optional synergy action space

For finger-only MPPI, a low-dimensional synergy action space may be preferable
to raw per-joint delta-q sampling.

Example:

```text
a = [
  common_grip_delta,
  differential_balance_delta,
  distal_curl_delta,
  release_or_soften_delta
]

delta_q_ref = B * a
```

If implemented, keep the mapping explicit, bounded, and testable.

Do not hide synergy matrices in cost functions.

---

## Cost Architecture

Use modular cost terms.

Generic costs:

```text
CostStack
  ActionSmoothnessCost
  JointLimitCost
  TrackingGuardCost
  ControlEffortCost
```

NariTouch/tactile costs:

```text
NormalForceWindowCost
SlipRiskCost
ContactCentroidCost
ContactLossRiskCost
ForceSpikeCost
TactileSafeSetCost
```

Stage/task costs:

```text
SupportLossReadinessCost
ReleaseReadinessCost
PlacementSofteningCost
```

Cost terms should be pure evaluators. They should not update global controller
state.

Stage-conditioned behavior should come from `StagePolicy` and `CostWeights`,
not from hidden if-statements scattered throughout the optimizer.

### Normal force window

For tactile manipulation, prefer a force window over simple force minimization:

```text
cost = relu(f_min - f_n)^2 + relu(f_n - f_max)^2
```

Stage semantics:

```text
Extraction:
  low minimum force
  strict maximum force
  avoid over-squeezing and jamming

Capture:
  higher minimum force
  moderate maximum force
  avoid slip/contact loss after support transition

Placement:
  low minimum force
  very strict maximum force
  allow compliance and self-alignment
```

### Tracking guard

Because the low-level impedance controller may lag behind references, keep
tracking error visible:

```text
tracking_error = q_ref_current - q_measured
```

If tracking error is large:

```text
penalize aggressive q_ref updates
penalize large delta_q_ref commands
prefer smoother recovery
```

---

## Stage-Conditioned Manipulation

The generic core may support stage-conditioned policy, but task-specific stage
definitions should live in examples, configs, adapters, or stage policy objects.

For the PLATOMPPI/Jenga example, intended stages are:

```text
Extraction
  local finger goal:
    hold just enough to avoid slip while avoiding over-squeeze and jam

Capture
  local finger goal:
    handle support-loss/contact-loss events and stabilize the grasp

Placement
  local finger goal:
    maintain a low-force compliant grasp while avoiding insertion jam

Release
  local finger goal:
    gradually release after support appears restored
```

The stage manager may use tactile context:

```text
slip probability
normal force rate
contact centroid shift
contact loss
force spike
support-loss probability
```

Do not hard-code these stages into `MPPIOptimizer`.

---

## Reflex Layer

For tactile manipulation, a reflex layer is allowed and encouraged.

The reflex layer is not MPPI. It is a fast command modifier for urgent tactile
events.

Examples:

```text
slip spike
  -> immediate bounded close bump

contact loss
  -> immediate bounded close/search bump

force spike
  -> immediate opening or stiffness reduction

contact centroid near boundary
  -> immediate recentering bias
```

Keep it separate:

```text
MPPI command/reference
  -> ReflexLayer
  -> final command/reference
```

Do not hide reflex behavior inside the optimizer update rule.

A useful rule of thumb:

```text
reflex = survival
MPPI = recovery and regulation
```

---

## Configuration Rules

Do not parse YAML inside hot paths.

Recommended config split:

```text
mppi.yaml
  horizon_steps
  dt
  num_rollouts
  temperature
  num_knots
  random_seed
  action_bounds
  noise_std
  optional synergy_matrix

stage_policy.yaml
  stage names
  normal force windows
  contact centroid safe region
  stiffness/gain bounds if enabled
  cost weights
  transition thresholds

object_prior.yaml
  mass
  dimensions
  inertia
  friction prior

naritouch.yaml
  node layout if not compiled
  force scaling
  slip feature thresholds
  contact centroid bounds
```

Config files should compile into plain structs before runtime control begins.

---

## Testing Requirements

Every meaningful change should add or update tests.

Minimum tests for v1:

```text
test_mppi_shapes
  checks rollout dimensions, action dimensions, horizon dimensions

test_sampling_policy
  checks deterministic seeded sampling and bound handling

test_cost_stack
  checks cost accumulation and stage-conditioned weights

test_shift_nominal
  checks receding-horizon shift behavior

test_naritouch
  checks total normal force, contact centroid, contact-valid flags

test_pinocchio_rollout
  checks basic q/v propagation and finite tracked frame data

test_tracking_guard_cost
  checks large tracking error produces higher cost
```

Tests should be runnable with:

```bash
colcon test --packages-select mppi_core
colcon test-result --verbose
```

For workspace-level sanity:

```bash
colcon build --packages-select mppi_core
```

If modifying a future integration adapter with `control_architecture`, also run:

```bash
colcon build --packages-select control_architecture
```

---

## Build System Rules

Use `ament_cmake`.

Follow the style of the workspace packages:

```text
define sources explicitly
build a shared library
install headers
export targets
export dependencies
add gtests under BUILD_TESTING
```

The package should export itself as a reusable CMake target:

```text
mppi_core
```

Do not rely on implicit include paths or uninstalled headers.

---

## Documentation Rules

Every major class should document:

```text
what it owns
what it does not own
whether it is safe for control-loop use
whether it allocates during update
what dimensions it expects
what assumptions it makes about NariTouch / finger-only control
```

Prefer comments that explain semantics, not obvious syntax.

Good:

```text
ObjectPrior is not a rollout state. It defines thresholds and cost parameters
for object-prior-aware tactile regulation.
```

Good:

```text
TactileProxyModel is a local directional risk model. It does not predict full
object dynamics.
```

Bad:

```text
This is the object prior.
```

---

## Non-Goals for v1

Do not implement these in the first pass unless explicitly requested:

```text
full object dynamics rollout
object pose estimation
wrist pose control
whole-body control
inverse-dynamics HQP integration
MuJoCo-based simulation backend
CUDA backend
Torch/JAX/Python dependency in C++ core
learned tactile transition model
differentiable dynamics
full ROS2 controller plugin
hardware interface
global task planner
perception pipeline
```

These can be future extensions once the finger-only tactile MPPI abstraction is
stable.

---

## Preferred First Milestone

Build the smallest useful core:

```text
mppi_core package compiles
MPPIConfig exists
ActionSequence exists
SamplingPolicy exists
MPPIOptimizer exists
RobotRolloutState exists as q_finger_ref / v_finger_ref
RolloutModelBase exists
CostStack exists
ActionSmoothnessCost exists
JointLimitCost exists
TrackingGuardCost exists
NariTouchState exists
unit tests compile and pass
```

Then add:

```text
PinocchioRolloutModel
NariTouch feature utilities
StagePolicy
NormalForceWindowCost
SlipRiskCost
ContactCentroidCost
ForceSpikeCost
ReflexLayer
```

Next concrete implementation direction:

```text
1. Fix MPPI action semantics to bounded finger delta_q_ref.
2. Keep rollout state centered on q_finger_ref and optional qdot_ff.
3. Implement PinocchioRolloutModel with integrate(q_ref, delta_q_ref).
4. Compute tracked fingertip frame pose/velocity from the q_ref rollout.
5. Add NariTouchState and feature extraction.
6. Start costs with joint limits, action smoothness, tracking guard,
   normal force window, slip risk, and contact centroid safety.
7. Output the first q_des command and optional qdot_ff.
```

Only after that, create the concrete `plato_mppi` / Jenga tactile example.

---

## Red Flags

Stop and reconsider if a change causes any of the following:

```text
mppi_core starts solving inverse-dynamics HQP internally
mppi_core computes motor torques directly
mppi_core controls wrist pose in v1
mppi_core requires object pose in v1
mppi_core requires wrist pose in v1
mppi_core directly publishes ROS messages
mppi_core parses YAML inside update()
MPPIOptimizer knows about Jenga stages directly
object pose becomes mandatory for generic rollout
cost terms mutate optimizer state
action dimensions are implicit or discovered from vector sizes with no checks
Eigen vectors resize silently inside rollout hot loops
NariTouch-specific code is named as if it were generic tactile code
package dependencies balloon before the core API is stable
```

---

## One-Sentence Identity

`mppi_core` is a Pinocchio-based finger MPPI library that generates short-horizon
joint reference trajectories for impedance-controlled fingers, using NariTouch
feedback to regulate local contact quality under large unmodeled wrist, object,
and environment disturbances.
