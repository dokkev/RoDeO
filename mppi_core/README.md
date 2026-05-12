# mppi_core

`mppi_core` is a ROS-free C++ package for reusable MPPI runtime objects.

The first optimizer surface is `mppi_core::MPPIOptimizer`: it owns a nominal
action sequence, samples noisy rollouts around it, evaluates a caller-provided
`RolloutModelBase` and `CostTermBase`, updates the nominal sequence with MPPI
weights, and returns the first action as a short-horizon command.

For the initial PLATOMPPI path, the action is a bounded finger `delta_q_ref`
increment. The rollout integrates that into the next `q_ref`; any `v_des`
feedforward is derived from `delta_q_ref / dt` and is not the primary tracking
objective.

The first implemented tactile surface is explicitly NariTouch-shaped. It keeps
only the signals MPPI needs early on:

- `slip_state`: 3D slip/shear state.
- `slip_velocity_state`: derived 3D slip velocity, normally computed by the
  ROS adapter from consecutive shear-displacement samples.
- `force_z`: tactile normal-axis force estimate.
- `contact_state`: contact mode enum with values `0`, `1`, and `2`.
- `nodes[8]`: per-hemisphere position, contact, center of pressure, and normal
  force.

For NariTouch `sdr_grasp_msgs/msg/Tactile`, a ROS adapter can map
`shear_displacement.x/y/theta` into `slip_state`, `force.z` into `force_z`, and
`contact_state` into `mppi_core::NariTouchContactState`. The message `units`
array maps to `NariTouchState::nodes`, where each NariTouch hemisphere
contributes `contact`, `cop`, and `normal_force`. The node `position_m` should
come from the sensor-local hemisphere layout; `NariTouchState` initializes the
NariTouch 4-by-2 hemisphere positions from `area_pos_ofs_x/y * 1e-3`. Rollout
models can use `NariTouchNodeSensorPosition` directly as local prediction data
in the tactile sensor frame. Raw pressure grids, sensor array metadata,
TF/world poses, and node indices should stay outside `mppi_core` until a cost
explicitly needs them.

NariTouch messages expose shear displacement rather than slip velocity. The
hardware-facing adapter should derive `slip_velocity_state` with a timestamped
finite difference and light filtering before passing the state into
`mppi_core`.

Object information enters MPPI as a lightweight prior rather than an object
state rollout. For a Jenga-sized block, use:

```cpp
const auto object = mppi_core::MakeJengaBlockObjectPrior();
```

The first Jenga policy is intentionally a contact-local tactile-risk governor,
not a full object dynamics simulator. The rollout state is a `GraspState`: it
carries the candidate joint reference, a predicted `NariTouchState`, and
predicted normal force, tangential load, slip risk, contact centroid, and
friction margin.

The predicted `NariTouchState` also carries a local contact patch. Each rollout
step estimates how many hemispheres should be active from normal force, picks
the nodes closest to the predicted centroid, and increases slip state when the
friction margin goes negative. The friction margin mirrors the linearized
friction-pyramid convention used by `wbc_core` contacts, but it is used as a
soft prediction feature rather than an HQP constraint.

```cpp
auto config = mppi_core::MakeDefaultJengaGraspConfig(joint_dim);

mppi_core::JengaGrasp policy;
policy.Initialize(joint_dim, config);

mppi_core::GraspObservation obs;
obs.q_measured = q_measured;
obs.v_measured = v_measured;
obs.q_ref_current = q_ref_current;
obs.v_ref_current = v_ref_current;
obs.tactile = naritouch_state;

const auto command = policy.Update(obs);
// command.q_des is safe to send downstream; command.delta_q_ref is the raw MPPI
// increment and command.v_des is delta_q_ref / dt.
```

The policy rolls out `delta_q_ref` candidates with
`DeltaQReferenceRolloutModel` and evaluates them with
`GraspStabilityCost`. Disturbances are represented as local tactile-risk
scenarios, such as normal-force drops/spikes, slip bumps, and contact-centroid
drift. This keeps the v1 controller focused on stabilizing contact quality
without pretending to predict full Jenga block motion.

The optimizer sampling budget and action bounds live in `config/mppi.yaml`:

```yaml
mppi:
  horizon_steps: 20
  dt: 0.01
  num_rollouts: 128
  temperature: 1.0
  random_seed: 1

  action:
    lower_bound: -0.004
    upper_bound: 0.004
    noise_std: 0.0015
```

Scalars under `action` expand to all controlled joints; joint-sized sequences
can be used when each finger joint needs a different bound or noise level.

The default grasp cost can be configured separately from YAML at initialization
time:

```yaml
grasp:
  normal_force_window:
    min_n: 0.5
    max_n: 2.5
    under_weight: 20.0
    over_weight: 40.0

  friction_margin:
    enabled: true
    weight: 20.0
    mu_nominal: 0.35
    required_force_weight: 10.0

  tactile_prediction:
    force_per_node_n: 0.4
    slip_decay: 0.9
    slip_margin_gain_per_n: 0.2
    centroid_drift_gain_m_per_n: 0.0005

  slip_risk:
    velocity_weight: 0.0

  action:
    closing_direction: [-1.0, -1.0, 1.0, 1.0]
```

The YAML loader is not part of the rollout hot loop. It updates
`MPPIConfig` and `GraspStabilityCostConfig` once during policy initialization.

```cpp
#include "mppi_core/config/grasp_config.hpp"
#include "mppi_core/config/mppi_config.hpp"

auto config = mppi_core::MakeDefaultJengaGraspConfig(joint_dim);
config.mppi = mppi_core::LoadMPPIConfigFromYamlFile(
    mppi_yaml_path, joint_dim, config.mppi);
config.grasp_stability_cost = mppi_core::LoadGraspConfigFromYamlFile(
    cost_yaml_path, joint_dim, config.grasp_stability_cost);

mppi_core::JengaGrasp policy;
policy.Initialize(joint_dim, config);
```
