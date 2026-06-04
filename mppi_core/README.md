# mppi_core

`mppi_core` is a ROS-free C++ package for reusable MPPI runtime objects. The
current controller shape is contact-local: enter MPPI after tactile contact is
active, roll out small joint-reference increments, predict tactile state from
torque-residual contact forces, and score force/slip/centroid/contact-patch
quality.

The main runtime pieces are:

- `mppi_core::MPPIOptimizer`
- `mppi_core::DeltaQReferenceRolloutModel`
- `mppi_core::GraspStabilityCost`
- `mppi_core::TactileState`
- `mppi_core::ContactForceProjection`
- `mppi_core::ContactForceRollout`
- `mppi_core::ContactForceCorrection`
- `mppi_core::RobotCommand`

For a click-through view of how these pieces exchange data, open
[`docs/mppi_flow.html`](docs/mppi_flow.html) in a browser.

`TactileState` is hardware-agnostic. It carries contact presence, normal force,
contact centroid, shear, slip scores, support counts, and contact points. The
NARI adapter remains hardware-specific input plumbing: `NariTouchState` is
converted to `TactileState` before planner code sees it.

In measured/current states, `tau` is measured joint torque. In future rollout
states, `tau` is a predicted commanded-torque proxy from the rollout impedance
model. That means the first force-aware rollout step is anchored by measured
torque, while later steps are force hypotheses based on predicted torque.

By default, tactile rollout is force-aware required. If torque-residual contact
force projection is unavailable or invalid, the rollout state is invalid and MPPI
assigns the invalid-rollout penalty. The kinematic contact-patch rollout can be
enabled only as an explicit ablation/debug fallback.

`MPPIOptimizer` is intentionally rollout-model agnostic, so it does not inspect
`TactileRolloutPolicy` or perform the no-contact gate itself. A force-aware
required caller must check tactile contact before calling `Update(...)`; the
optimizer only provides a safety backstop by returning a hold command when all
sampled rollouts are invalid.

```cpp
mppi_core::MPPIConfig mppi_config;
mppi_core::DeltaQReferenceRolloutConfig rollout_config;
rollout_config.tactile_rollout_policy =
    mppi_core::TactileRolloutPolicy::kForceAwareRequired;

auto model = std::make_shared<mppi_core::DeltaQReferenceRolloutModel>(
    joint_dim, rollout_config);
auto cost = std::make_shared<mppi_core::GraspStabilityCost>(
    mppi_core::GraspStabilityCostConfig{});

mppi_core::MPPIOptimizer optimizer;
optimizer.Initialize(mppi_config, model, cost);

mppi_core::GraspObservation obs;
obs.q_measured = q_measured;
obs.v_measured = v_measured;
obs.q_ref_current = q_ref_current;
obs.v_ref_current = v_ref_current;
obs.tau = measured_joint_torque;
obs.tactile = tactile_state;
obs.contact_kinematics = &contact_kinematics;
obs.contact_force_projection_config = &projection_config;
obs.contact_force_rollout_config = &force_rollout_config;

const auto command = optimizer.Update(obs);
```

`RobotCommand` is the generic final command packet. It is not the MPPI action.
MPPI may sample `delta_q_ref`, but the downstream controller receives a hybrid
impedance command:

```text
tau_cmd = tau_ff + kp * (q_des - q) + kd * (qdot_des - qdot)
```

For pure position/velocity impedance behavior, keep `tau_ff = 0`. For mostly
torque-feedforward behavior, keep `kp` and `kd` small or zero. Hybrid behavior
uses both nonzero feedforward torque and nonzero gains.

Sampling budget and action bounds live in `config/mppi.yaml`. Contact-local cost
and force-aware rollout settings live in `config/grasp.yaml`:

```yaml
grasp:
  tactile_prediction:
    rollout_policy: force_aware_required

  contact_force_rollout:
    enable_force_projection_update: true
    force_lowpass_alpha: 0.5
    max_predicted_normal_force_n: 20.0
    rollout_torque_stiffness_nm_per_rad: 1.0
    rollout_torque_damping_nms_per_rad: 0.01

  normal_force_window:
    min_n: 0.5
    max_n: 2.5
    under_weight: 20.0
    over_weight: 40.0

  slip_risk:
    threshold: 0.25
    weight: 4.0
```

For debugging, `MPPIOptimizer::PredictRollout(...)` returns a `RolloutTrace`.
`states[k + 1]` contains the predicted joint reference, predicted torque proxy,
and force-aware tactile prediction after `actions[k]`.
