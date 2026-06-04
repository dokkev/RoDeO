# Force-Aware Tactile MPPI Tuning

This guide covers the current contact-gated tactile/torque MPPI path. MPPI is
intended to run after tactile contact exists. Approach or guarded-motion behavior
belongs to the caller.

`MPPIOptimizer` does not own this contact gate because it is a generic optimizer
over a `RolloutModelBase`. In `force_aware_required` mode, the caller/controller
must check tactile contact before calling `Update(...)`. If the caller misses the
gate, force-aware rollout failure marks samples invalid and the optimizer returns
a hold command when all rollouts are invalid.

## Files

- `mppi_core/config/mppi.yaml`: horizon, sampling, temperature, and action bounds.
- `mppi_core/config/grasp.yaml`: force-aware rollout settings and contact-local
  cost weights.
- ROS adapter config: joint list, activation gate, command publishing, and
  hardware adapter parameters.

## Rollout Policy

```yaml
tactile_prediction:
  rollout_policy: force_aware_required
```

`force_aware_required` is the runtime default. Every rollout step must project
contact force from torque residuals and update tactile state through
`ContactForceRollout`. If projection fails, the rollout is invalid.

For ablation/debug only:

```yaml
tactile_prediction:
  rollout_policy: force_then_kinematic_fallback
```

This tries force-aware rollout first, then allows the kinematic contact-patch
rollout. The simple heuristic tactile proxy is not part of the runtime core.
Keep this mode for debug/ablation only; it is not the default controller path.

## Force Rollout

```yaml
contact_force_rollout:
  enable_force_projection_update: true
  force_lowpass_alpha: 0.5
  max_predicted_normal_force_n: 20.0
  shear_force_gain_m_per_n_s: 0.0001
  rotational_shear_gain_rad_per_nm_s: 0.01
  friction_violation_confidence_decay: 0.2
  negative_normal_confidence_decay: 0.5
  min_stable_support_count: 4
  min_contact_confidence: 0.001
  shear_ref_m: 0.001
  rotational_shear_ref_rad: 0.02
  rollout_torque_stiffness_nm_per_rad: 1.0
  rollout_torque_damping_nms_per_rad: 0.01
```

`rollout_torque_stiffness_nm_per_rad` and
`rollout_torque_damping_nms_per_rad` define the predicted future torque proxy
from sampled `delta_q_ref` actions. Current observed states still use measured
joint torque.

## Cost Weights

```yaml
normal_force_window:
  min_n: 0.5
  max_n: 2.5
  under_weight: 20.0
  over_weight: 40.0

slip_risk:
  threshold: 0.25
  weight: 4.0
  velocity_weight: 0.0

contact_centroid:
  enabled: true
  boundary_weight: 20.0
  x_min: -0.008
  x_max: 0.008
  y_min: -0.008
  y_max: 0.008

contact_patch:
  enabled: true
  target_node_count: 6
  weight: 2.0
```

Normal force costs keep predicted force in a usable window. Slip costs penalize
large shear-derived slip risk. Centroid and patch costs keep contact local and
well supported.

## Sampling

```yaml
mppi:
  horizon_steps: 20
  dt: 0.01
  num_rollouts: 128
  temperature: 1.0
  action:
    lower_bound: -0.002
    upper_bound: 0.002
    noise_std: 0.002
```

Actions are bounded joint reference increments. Larger `num_rollouts` and longer
`horizon_steps` improve coverage at higher CPU cost. Smaller `temperature` makes
the update prefer the lowest-cost samples more aggressively.
