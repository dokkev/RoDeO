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

The planner-facing tactile surface is `mppi_core::TactileState`. It is
hardware-agnostic and keeps only the physical/contact features MPPI needs:
contact presence, normal force, contact centroid and centroid velocity,
translational shear, rotational shear, scalar slip scores, support counts, and
optional geometry quality terms.

NariTouch remains a hardware-specific input representation in
`mppi_core::NariTouchState`. For `sdr_grasp_msgs/msg/Tactile`, the ROS adapter
maps `shear_displacement.x/y/theta` into NariTouch `slip_state`, `force.z` into
`force_z`, `contact_state` into `mppi_core::NariTouchContactState`, and message
`units` into `NariTouchState::units`. The adapter then calls
`ConvertNariTouchToTactileState(...)`, where NariTouch `slip_state.x/y` becomes
generic translational shear and `slip_state.z` becomes rotational shear. Raw
pressure grids, sensor array metadata, TF/world poses, and raw sensing node
indices should stay outside planner code until a cost explicitly needs them.

NariTouch messages expose shear displacement rather than slip velocity. The
hardware-facing adapter derives `slip_velocity_state` with a timestamped finite
difference and light filtering before converting to `TactileState`.

Object information enters MPPI as a lightweight prior rather than an object
state rollout. For a Jenga-sized block, use:

```cpp
const auto object = mppi_core::MakeJengaBlockObjectPrior();
```

The first Jenga policy is intentionally a contact-local tactile-risk governor,
not a full object dynamics simulator. The rollout state is a `GraspState`: it
carries the candidate joint reference, a predicted `TactileState`, and
predicted normal force, tangential load, slip risk, contact centroid, and
friction margin.

The predicted `TactileState` also carries a local contact support proxy. Each
rollout step estimates support count from normal force and increases
translational/rotational shear when the friction margin goes negative. The
friction margin mirrors the linearized friction-pyramid convention used by
`wbc_core` contacts, but it is used as a soft prediction feature rather than an
HQP constraint.

```cpp
auto config = mppi_core::MakeDefaultJengaGraspConfig(joint_dim);

mppi_core::JengaGrasp policy;
policy.Initialize(joint_dim, config);

mppi_core::GraspObservation obs;
obs.q_measured = q_measured;
obs.v_measured = v_measured;
obs.q_ref_current = q_ref_current;
obs.v_ref_current = v_ref_current;
obs.tau = measured_joint_torque;
obs.tactile = tactile_state;

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

For debugging, the optimizer can roll out any candidate `ActionSequence` and
return a `RolloutTrace`. This is the main way to inspect whether a proposed
control input makes sense: `states[k + 1]` contains the predicted joint
reference, tactile contact patch, slip risk, force proxy, and friction margin
after `actions[k]`.

```cpp
mppi_core::ActionSequence candidate(joint_dim, horizon_steps);
candidate.setAction(0, delta_q_ref);

const auto trace = policy.PredictRollout(obs, candidate);
const auto& predicted = trace.states[1];
```

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
    under_weight: 5.0
    over_weight: 30.0

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
See [`docs/grasp_tuning.md`](docs/grasp_tuning.md) for the current hardware
tuning knobs and symptom-based adjustment guide.

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
