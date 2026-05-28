# Grasp MPPI Tuning Guide

This document lists the knobs that are currently tunable for the Jenga grasp
MPPI path.

The current controller is a grasp-maintain controller. It does not plan a full
object pose trajectory. It samples small joint reference increments, predicts a
local NariTouch contact state, evaluates grasp quality, and publishes joint
impedance references.

## Files

- `mppi_core/config/mppi.yaml`
  Sampling budget, horizon, temperature, and `delta_q_ref` action bounds.
- `mppi_core/config/grasp.yaml`
  Grasp cost weights and tactile prediction gains.
- `controller/plato_mppi/config/jenga_grasp_mppi.yaml`
  ROS adapter settings, controlled joints, activation gate, command gains, and
  optional MPPI parameter overrides.

The MPPI node loads these files at startup. Relaunch `plato_mppi` after editing
them. Controller-manager YAML changes require reloading the hardware/control
launch as well.

## Current Control Semantics

The action is a bounded joint reference increment:

```text
u_k = delta_q_ref_k
q_ref_next = q_ref + delta_q_ref
v_des = delta_q_ref / dt
```

For the current Plato setup, MPPI controls:

```yaml
joint_names: [joint3, joint4, joint5, joint6]
```

`joint1` and `joint2` are held by `fixed_joint_positions` in the ROS adapter.

The current closing sign convention is:

```yaml
action:
  closing_direction: [-1.0, -1.0, 1.0, 1.0]
```

This means negative `joint3/joint4` delta and positive `joint5/joint6` delta are
treated as closing.

## MPPI Sampling Knobs

These live in `mppi_core/config/mppi.yaml`.

```yaml
mppi:
  horizon_steps: 20
  dt: 0.01
  num_rollouts: 128
  temperature: 1.0
  random_seed: 1

  action:
    lower_bound: -0.002
    upper_bound: 0.002
    noise_std: 0.002
```

| Field | Unit | Effect |
| --- | --- | --- |
| `horizon_steps` | steps | Number of future steps per rollout. Larger sees farther but costs more CPU. |
| `dt` | seconds | MPPI rollout step. Also maps `delta_q_ref` to `v_des`. |
| `num_rollouts` | samples | Number of candidate action sequences. Larger is smoother/more reliable but slower. |
| `temperature` | cost units | Higher averages more rollouts. Lower makes MPPI more winner-take-most. |
| `action.lower_bound` | rad/step | Minimum `delta_q_ref` per controlled joint. Scalar expands to all joints. |
| `action.upper_bound` | rad/step | Maximum `delta_q_ref` per controlled joint. |
| `action.noise_std` | rad/step | Gaussian exploration around the nominal action sequence. Samples are clipped to bounds. |

With `dt = 0.01`, these approximate feedforward velocities:

```text
0.001 rad/step -> 0.1 rad/s
0.002 rad/step -> 0.2 rad/s
0.004 rad/step -> 0.4 rad/s
```

The first rollout is the current nominal sequence without noise. The other
rollouts add Gaussian noise and then clamp each action to the configured bounds.

## Grasp Cost Weights

These live in `mppi_core/config/grasp.yaml`.

### Normal Force Window

```yaml
normal_force_window:
  min_n: 0.5
  max_n: 2.5
  under_weight: 20.0
  over_weight: 40.0
  use_object_weight_lower_bound: true
  weight_safety_factor: 1.5
```

| Field | Effect |
| --- | --- |
| `min_n` | Minimum desired normal force before object-weight lower bound. |
| `max_n` | Maximum desired normal force. Increase only if over-grip is acceptable. |
| `under_weight` | Penalty for too little force. Higher makes the policy close harder. |
| `over_weight` | Penalty for too much force. Higher makes the policy avoid squeezing. |
| `use_object_weight_lower_bound` | If true, raises `min_n` using object mass, friction, and contact count. |
| `weight_safety_factor` | Multiplier on object-weight lower bound. Higher is safer against slip but can over-grip. |

For the default Jenga prior, the object lower bound can dominate `min_n`.
Approximate lower bound:

```text
force_min = safety_factor * mass * g / (mu * supporting_contact_count)
```

### Friction Margin

```yaml
friction_margin:
  enabled: true
  weight: 20.0
  mu_nominal: 0.35
  mu_min: 0.2
  required_force_weight: 10.0
```

| Field | Effect |
| --- | --- |
| `enabled` | Enables soft friction margin penalty. This is not a hard constraint. |
| `weight` | Penalizes negative margin: `mu * normal_force - tangential_load`. |
| `mu_nominal` | Estimated friction coefficient used for margin prediction. |
| `mu_min` | Lower clamp used when estimating required normal force. |
| `required_force_weight` | Penalizes cases where required force exceeds `max_n`. |

Increase `weight` when slip risk should dominate. Decrease it if the controller
becomes too conservative and only squeezes.

### Tangential Load Proxy

```yaml
tangential_load_proxy:
  gravity_weight: 1.0
  motion_weight: 0.0
  slip_weight: 3.0
  force_spike_weight: 5.0
```

| Field | Effect |
| --- | --- |
| `gravity_weight` | Scales object weight contribution to tangential load. |
| `motion_weight` | Scales action-speed contribution. Currently a simple `||action|| / dt` proxy. |
| `slip_weight` | Converts slip risk into tangential load. Higher reacts more to slip. |
| `force_spike_weight` | Reserved for disturbance scenarios/force-spike cost. |

For the first hardware passes, keep `motion_weight` low or zero unless fast
reference changes are clearly causing slip.

### Normal Force Proxy

```yaml
normal_force_proxy:
  closing_force_gain: 1.0
  opening_force_gain: 1.0
  max_force_n: 5.0
```

| Field | Effect |
| --- | --- |
| `closing_force_gain` | Predicted force increase per closing radian. Higher makes closing look more effective. |
| `opening_force_gain` | Predicted force decrease per opening radian. |
| `max_force_n` | Clamp on predicted normal force. |

These are model gains, not cost weights. If MPPI keeps closing but predicted
force barely changes, increase `closing_force_gain`. If it is too confident that
tiny moves fix force, decrease it.

### Tactile Prediction

```yaml
tactile_prediction:
  force_per_node_n: 0.4
  slip_decay: 0.9
  slip_margin_gain_per_n: 0.2
  centroid_drift_gain_m_per_n: 0.0005
  slip_velocity_decay: 0.85
  slip_velocity_margin_gain_per_nps: 0.2
  action_slip_damping_gain_per_rad: 0.0
  max_slip_velocity: 100.0
  centroid_velocity_decay: 0.9
  centroid_velocity_slip_gain: 0.001
  max_centroid_velocity_mps: 0.05
```

| Field | Effect |
| --- | --- |
| `force_per_node_n` | Converts predicted normal force into predicted active node count. |
| `slip_decay` | Decay on predicted slip risk across rollout steps. |
| `slip_margin_gain_per_n` | Adds slip risk when friction margin is negative. |
| `centroid_drift_gain_m_per_n` | Moves predicted centroid along slip direction when margin is bad. |
| `slip_velocity_decay` | Decay on predicted slip velocity. |
| `slip_velocity_margin_gain_per_nps` | Adds slip velocity from margin deficit. |
| `action_slip_damping_gain_per_rad` | Allows closing action to damp predicted slip velocity. Currently safe at zero. |
| `max_slip_velocity` | Clamp on predicted slip velocity. |
| `centroid_velocity_decay` | Decay on predicted centroid velocity. |
| `centroid_velocity_slip_gain` | Converts slip velocity into centroid velocity. |
| `max_centroid_velocity_mps` | Clamp on predicted centroid velocity. |

These are rollout model gains. Tune them after the basic force and action
bounds feel reasonable.

### Slip Risk

```yaml
slip_risk:
  threshold: 0.25
  weight: 4.0
  velocity_weight: 0.0
```

| Field | Effect |
| --- | --- |
| `threshold` | No direct slip penalty below this risk. |
| `weight` | Penalty above threshold. |
| `velocity_weight` | Adds slip velocity magnitude into slip risk. |

Increase `weight` to react harder to slip. Raise `threshold` if the sensor is
noisy and MPPI twitches from harmless slip estimates.

### Contact Centroid

```yaml
contact_centroid:
  enabled: true
  boundary_weight: 20.0
  x_min: -0.008
  x_max: 0.008
  y_min: -0.008
  y_max: 0.008
```

| Field | Effect |
| --- | --- |
| `boundary_weight` | Penalizes predicted centroid outside the safe rectangle. |
| `x_min`, `x_max`, `y_min`, `y_max` | Safe centroid bounds in sensor-local meters. |

Use this when contact drifts toward the edge of the NariTouch surface.

### Contact Patch

```yaml
contact_patch:
  enabled: true
  target_node_count: 6
  weight: 2.0
```

| Field | Effect |
| --- | --- |
| `target_node_count` | Desired number of active NariTouch hemispheres. |
| `weight` | Penalizes having fewer predicted contact nodes. |

Important: the current ROS node selects one active tactile sensor for the MPPI
observation. Activation can require both `tactile_0` and `tactile_1`, but this
cost is per selected NariTouch sensor, not summed over both sensors.

### Tracking Guard

```yaml
tracking_guard:
  weight: 5.0
  action_scale_weight: 20.0
```

| Field | Effect |
| --- | --- |
| `weight` | Penalizes current reference/measured tracking error. |
| `action_scale_weight` | Penalizes large new actions more when tracking error is large. |

Increase these when the hand lags, oscillates, or command references run away
from measured joint states.

### Smoothness And Joint Limits

```yaml
action_smoothness:
  weight: 1.0

joint_limit:
  weight: 10.0
```

| Field | Effect |
| --- | --- |
| `action_smoothness.weight` | Penalizes large `delta_q_ref`. Higher makes motion quieter. |
| `joint_limit.weight` | Penalizes rollout states outside configured joint bounds. |

In the ROS adapter, hard clamping uses:

```yaml
q_lower_bound: [...]
q_upper_bound: [...]
```

Those bounds are also copied into the MPPI joint-limit cost.

## ROS Adapter Knobs

These live in `controller/plato_mppi/config/jenga_grasp_mppi.yaml`.

| Field | Effect |
| --- | --- |
| `control_rate_hz` | ROS command publish rate. |
| `activation_requires_all_tactile_enough_contact` | If true, MPPI starts only after all tactile topics are fresh and above threshold. |
| `activation_contact_state_threshold` | Current startup gate. `2` means enough contacts. |
| `publish_hold_without_tactile` | Publishes the held reference while waiting for tactile data. |
| `slip_velocity_filter_alpha` | Higher follows raw finite-difference slip velocity faster. Lower smooths more. |
| `slip_velocity_max_norm` | Clamp for derived slip velocity. |
| `centroid_velocity_filter_alpha` | Same idea for contact centroid velocity. |
| `centroid_velocity_max_norm_mps` | Clamp for centroid velocity. |
| `stiffness`, `damping` | Joint impedance command gains sent with every MPPI command. |

The adapter can override MPPI fields without editing `mppi_core/config/mppi.yaml`:

```yaml
mppi:
  action_lower_bound: -0.0015
  action_upper_bound: 0.0015
  action_noise_std: 0.0008
```

The cost YAML is currently loaded from `cost_yaml_path`; use a separate copied
grasp YAML if you want per-experiment cost tuning.

## Symptom-Based Tuning

### Fingers Move Too Much

Try in this order:

1. Lower `mppi.action.lower_bound` and `mppi.action.upper_bound` magnitude.
2. Lower `mppi.action.noise_std`.
3. Increase `action_smoothness.weight`.
4. Increase `tracking_guard.action_scale_weight`.
5. Lower impedance `stiffness` if the commanded reference is fine but hardware
   motion is too sharp.

Example conservative action setting:

```yaml
mppi:
  action_lower_bound: -0.001
  action_upper_bound: 0.001
  action_noise_std: 0.0006
```

### Fingers Move Like Nothing Is Happening

Try in this order:

1. Increase action bounds to `0.002` or `0.003` rad/step.
2. Increase `mppi.action.noise_std`, but keep it at or below the bound for
   early hardware tests.
3. Decrease `action_smoothness.weight`.
4. If contact force prediction is too weak, increase
   `normal_force_proxy.closing_force_gain`.

### Slipping Or Contact Feels Weak

Try in this order:

1. Increase `normal_force_window.under_weight`.
2. Increase `friction_margin.weight`.
3. Increase `tangential_load_proxy.slip_weight`.
4. Increase `slip_risk.weight`.
5. If Jenga weight support is too weak, increase
   `normal_force_window.weight_safety_factor`.

Watch `normal_force_window.max_n`: if the minimum required force is too close to
`max_n`, MPPI will have no gentle solution and will tend to squeeze.

### Over-Gripping

Try in this order:

1. Decrease `normal_force_window.min_n`.
2. Decrease `normal_force_window.weight_safety_factor`.
3. Increase `normal_force_window.over_weight`.
4. Decrease `friction_margin.weight` if margin cost dominates everything.
5. Lower `normal_force_window.max_n` only if the force sensor is calibrated and
   the value is physically safe.

### Contact Patch Is Too Small

Try:

```yaml
contact_patch:
  target_node_count: 6
  weight: 4.0

tactile_prediction:
  force_per_node_n: 0.3
```

Lower `force_per_node_n` makes the same predicted force activate more nodes in
the rollout. Higher `contact_patch.weight` makes MPPI prefer that outcome.

### Contact Drifts To Sensor Edge

Try:

```yaml
contact_centroid:
  boundary_weight: 40.0
```

If this causes twitching, reduce `centroid_velocity_slip_gain` or increase
`centroid_velocity_decay` smoothness by lowering it slightly.

### Slip Velocity Is Noisy

Try:

```yaml
slip_velocity_filter_alpha: 0.1
slip_velocity_max_norm: 20.0

slip_risk:
  velocity_weight: 0.0
```

Then re-enable velocity contribution only after the finite difference looks
stable.

## Suggested First Hardware Sweep

Start with a quiet action setting:

```yaml
mppi:
  action_lower_bound: -0.001
  action_upper_bound: 0.001
  action_noise_std: 0.0006
  temperature: 1.0
```

If the hand is stable but too weak, increase force/slip costs before increasing
action bounds. If the optimizer is choosing sensible `delta_q_ref` but the hand
still moves too hard, tune impedance `stiffness` and `damping` in
`jenga_grasp_mppi.yaml`.

For a more responsive but still moderate setting:

```yaml
mppi:
  action_lower_bound: -0.002
  action_upper_bound: 0.002
  action_noise_std: 0.001
```

Avoid making `noise_std` much larger than the action bound during early hardware
tests; most samples will clip to the bounds and the policy can look jerky.
