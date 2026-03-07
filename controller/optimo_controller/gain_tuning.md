# Optimo WBC Gain Tuning Results

## Architecture Overview

The Optimo 7-DOF manipulator WBC uses a cascaded control structure:

```
            WBC (WBIC)                          Joint PID
   ┌──────────────────────────┐    ┌──────────────────────────────┐
   │ Task kp/kd → op_cmd      │    │ Outer: qdot_ref = kp_pos*e_q │
   │ kp_ik     → IK (q_cmd)   │    │        + kd_pos*e_qdot       │
   │ weight    → QP priority   │    │ Inner: tau_fb = kp_vel *     │
   │ Output: tau_ff (FF torque)│    │        (qdot_ref - qdot)     │
   └──────────────────────────┘    └──────────────────────────────┘
                │                               │
                └───── tau = tau_ff + tau_fb ────┘
```

**Gain parameters:**

| Parameter | Location | Role |
|-----------|----------|------|
| `kp` / `kd` | task_list.yaml | Operational-space PD (used in WBIC torque computation) |
| `kp_ik` | task_list.yaml | IK position gain (scales position error in WBIC FindConfiguration) |
| `weight` | task_list.yaml / state_machine.yaml | QP task priority weighting |
| `kp_pos` / `kd_pos` | joint_pid_gains.yaml | Joint-level PD (cascade outer loop) |
| `kp_vel` | joint_pid_gains.yaml | Cascade inner loop gain (velocity → torque) |

## MuJoCo Joint Dynamics

Added to `optimo_description/mjcf/optimo.xml` to simulate realistic model error:

| Joint | Damping [Nm*s/rad] | Coulomb Friction [Nm] | Stiffness [Nm/rad] | Spring Ref [rad] |
|-------|-------------------|-----------------------|--------------------|--------------------|
| J1 | 2.0 | 1.0 | 48.0 | 0 |
| J2 | 2.0 | 1.2 | 47.0 | pi |
| J3 | 1.5 | 1.0 | 25.0 | 0 |
| J4 | 1.5 | 1.0 | 20.0 | -pi/2 |
| J5 | 0.5 | 2.0 | 24.0 | 0 |
| J6 | 0.5 | 2.0 | 24.0 | -pi/2 |
| J7 | 0.5 | 2.0 | 23.0 | 0 |

**Source:** `optimo_description/config/joint_dynamics.yaml`

- **Damping/friction:** Directly from hardware specs (gear/bearing losses, dry friction)
- **Stiffness:** ~5% of real SEA stiffness values (450-960 Nm/rad). Full SEA stiffness is NOT modeled because MuJoCo `stiffness` creates a spring to a fixed equilibrium, whereas real SEA has the spring between motor and link. The 5% values simulate small unmodeled compliance effects.
- **springref:** Set to home keyframe positions

These forces are NOT in the URDF/pinocchio model, so they create model error that the WBC feedforward cannot cancel. Joint PD feedback is required.

---

## Phase 1: Ideal Dynamics (No Friction/Springs)

### Joint Position Task Gains (Initialize/Home State)

Sweep over kp/kd for joint position tracking to home configuration:

| kp | kd | Steady-State Error | Notes |
|----|----|--------------------|-------|
| 50 | 14 | < 0.001 rad | Slow convergence |
| 100 | 20 | < 0.001 rad | Good balance |
| 200 | 28 | < 0.001 rad | Fast convergence |

**Selected:** `kp=100, kd=20` (critically damped, 2*sqrt(kp))

### EE Position/Orientation Task Gains (Cartesian Teleop)

Multi-waypoint tracking test: 8 waypoints forming a 3D path (XZ square + Y depth probe), constant-velocity teleop at 0.05 m/s, 1s hold at each waypoint.

| Config | Transit RMS [mm] | Worst Transit [mm] | Hold Error [mm] | Ori Error [deg] |
|--------|-------------------|--------------------:|----------------:|----------------:|
| kp=800/kd=57 | 0.16 | 0.66 | 0.07 | 0.00 |
| kp=1600/kd=80 | 0.10 | 0.47 | 0.05 | 0.00 |
| kp=3200/kd=113 | 0.06 | 0.34 | 0.05 | 0.00 |
| kp=6400/kd=160 | 0.05 | 0.24 | 0.07 | 0.00 |

**Key findings (ideal dynamics):**
- All gains stable up to kp=6400
- Sub-millimeter tracking across the board
- `kp_ik` has **no effect** with inertia compensation enabled (tested 1.0, 5.0, 10.0, 20.0 — identical results)
- `jpos_kp` has minimal effect on cartesian tracking (jpos weight is 0.01 in cartesian state)

---

## Phase 2: Realistic Dynamics (Damping + Friction + Stiffness)

### Without Joint PID (Baseline)

| Config | Transit RMS [mm] | Worst Transit [mm] | Hold Error [mm] | Ori Error [deg] |
|--------|-------------------|--------------------:|----------------:|----------------:|
| kp=1600/kd=80 | 5.59 | 10.80 | 10.29 | 4.26 |
| kp=3200/kd=113 | 3.84 | 7.44 | 6.86 | 3.42 |

Model error causes 5-11mm tracking error and ~4-6 deg orientation drift.

### With Joint PD Feedback

Cascade PID: `tau_fb = kp_vel * (kp_pos*(q_des-q) + kd_pos*(qdot_des-qdot) - qdot)`

#### kp_vel = 1 (direct PD torque)

| PD kp_pos | PD kd_pos | EE kp | Transit RMS [mm] | Worst Transit [mm] | Hold Error [mm] | Ori Error [deg] |
|-----------|-----------|-------|-------------------|--------------------:|----------------:|----------------:|
| 10 | 2 | 1600 | 4.61 | 9.16 | 8.60 | 3.41 |
| 50 | 10 | 1600 | 2.79 | 5.88 | 5.37 | 1.84 |
| 100 | 20 | 1600 | 2.01 | 4.28 | 3.82 | 1.16 |
| 200 | 28 | 1600 | 1.45 | 3.00 | 2.58 | 0.67 |
| 500 | 45 | 1600 | 6.31 | 15.10 | 9.95 | 3.09 |
| 100 | 20 | 3200 | 1.58 | 3.30 | 3.04 | 1.07 |
| **200** | **28** | **3200** | **1.07** | **2.33** | **2.07** | **0.63** |
| 500 | 45 | 3200 | 3.99 | 9.90 | 4.07 | 3.94 |

#### kp_vel = 5 (amplified velocity correction)

| PD kp_pos | PD kd_pos | EE kp | Transit RMS [mm] | Worst Transit [mm] | Hold Error [mm] | Ori Error [deg] |
|-----------|-----------|-------|-------------------|--------------------:|----------------:|----------------:|
| 50 | 10 | 1600 | 11.35 | 22.79 | 19.93 | 4.52 |
| 100 | 20 | 1600 | 13.04 | 30.77 | 29.16 | 3.73 |
| 200 | 28 | 1600 | 13.07 | 31.00 | 29.43 | 2.78 |

**kp_vel > 1 consistently worse** — amplifies velocity correction, causes overshoot/oscillation.

### Optimal Configuration (Realistic Dynamics)

```yaml
# task_list.yaml
jpos_task:   kp=100, kd=20, weight=1.0
ee_pos_task: kp=3200, kd=113, weight=100.0
ee_ori_task: kp=3200, kd=113, weight=100.0

# joint_pid_gains.yaml
kp_pos: 200
kd_pos: 28
kp_vel: 1
```

**Performance:**
| Metric | Value |
|--------|------:|
| Avg transit RMS | 1.07 mm |
| Worst transit error | 2.33 mm |
| Avg hold error | 0.82 mm |
| Worst hold error | 2.07 mm |
| Worst orientation error | 0.63 deg |

---

## Observations

1. **PD kp_pos sweet spot: 200.** Below 200 → too little correction. Above 500 → oscillation/instability from fighting WBC feedforward.

2. **kp_vel = 1 always wins.** Higher kp_vel amplifies the velocity loop, causing overshoot. Keep at 1 for PD-position mode.

3. **EE kp=3200 slightly better than 1600** with PD enabled (1.07 vs 1.45 mm transit RMS).

4. **kp_ik irrelevant** with inertia compensation. The torque-based control (WBIC MakeTorque phase) dominates over the IK phase (FindConfiguration).

5. **Steady-state error at non-home positions** is structural with PD-only control + spring model. Spring force creates a constant bias that PD can't fully cancel (finite gain, no integrator). For home-target configs, the spring *helps* convergence (error < 0.002 rad).

6. **Damping (2.0 Nm*s/rad) + friction (1-2 Nm) are the dominant disturbances** in practice. The 5% stiffness adds position-dependent bias but is secondary.

## Test Infrastructure

- **Test file:** `optimo_controller/test/test_gain_tuning.cpp`
- **Test name:** `TrajectoryTracking.MultiWaypointCartesianTeleop`
- **Waypoints:** 8-point 3D path (XZ square, Y depth probe, return to home)
- **Protocol:** Constant-velocity teleop (0.05 m/s) → 1s hold at each waypoint
- **Metrics:** Transit RMS/max, hold error (distance from target after 1s), orientation error

---

## Phase 3: Adaptive Feedforward Compensators

### Architecture

Two independent modules added to `ControlArchitecture`, executed between WBIC solve and PID feedback:

```
 WBIC Output (tau_ff)
       │
       ├── + Friction Comp: tau += f_c*sign(qdot) + f_v*qdot
       │                    (adaptive: f_c, f_v estimated online)
       │
       ├── - Momentum Obs:  tau -= K_o * (M*qdot - p_hat)
       │                    (disturbance rejection via momentum residual)
       │
       └── tau_ff (augmented)
              │
              + PID feedback (tau_fb)
              │
              = Final tau (clamped)
```

### 1. Adaptive Friction Compensator

**Model:** `tau_fric = f_c * sign(qdot) + f_v * qdot`

**Adaptation law** (gradient descent on velocity error):
- `f_c += gamma_c * e_v * sign(qdot) * dt`
- `f_v += gamma_v * e_v * qdot * dt`
- where `e_v = qdot_des - qdot`

**Safety:** Estimates clamped to `[0, max_f_c]` and `[0, max_f_v]`.

**Source:** `wbc_util/include/wbc_util/adaptive_friction_compensator.hpp`

### 2. Momentum Observer (De Luca 2003)

**Model:** Estimates disturbance torque from momentum residual:
- `p = M(q) * qdot` (generalized momentum)
- `tau_dist = K_o * (p - p_hat)` (residual)
- `p_hat += (tau_cmd + C*qdot - g + tau_dist) * dt`

**Key:** K_o controls observer bandwidth. Higher K_o → faster tracking but more noise.

**Source:** `wbc_util/include/wbc_util/momentum_observer.hpp`

### YAML Configuration

```yaml
controller:
  friction_compensator:
    enabled: true
    gamma_c: 5.0      # Coulomb friction learning rate
    gamma_v: 2.0      # Viscous friction learning rate
    max_f_c: 5.0      # Safety limit [Nm]
    max_f_v: 3.0      # Safety limit [Nm*s/rad]

  momentum_observer:
    enabled: true
    K_o: 50.0          # Observer bandwidth gain
    max_tau_dist: 20.0  # Disturbance estimate clamp [Nm]
```

---

## Phase 4: Domain Randomization Results

### Protocol

- **Randomization:** Joint damping, friction, stiffness each scaled by uniform [0.5, 1.5] per joint
- **Test:** 8-waypoint Cartesian teleop (same as Phase 2)
- **Configurations:** PID-only baseline, PID+friction, PID+observer, PID+both
- **Seeds:** 5 random configurations

### Compensator Comparison (5 seeds, ±50% randomization)

| Config | Avg Transit RMS [mm] | Avg Worst Transit [mm] | Avg Hold Err [mm] | Avg Worst Hold [mm] | Ori [deg] | Stable |
|--------|--------------------:|----------------------:|------------------:|-------------------:|----------:|--------|
| PID only | 1.00 | 2.08 | 0.78 | 1.85 | 0.58 | 5/5 |
| PID + Friction | 1.00 | 2.06 | 0.78 | 1.85 | 0.58 | 5/5 |
| **PID + MomObs** | **0.08** | **0.32** | **0.03** | **0.06** | **0.00** | **5/5** |
| PID + Both | 0.08 | 0.32 | 0.03 | 0.06 | 0.00 | 5/5 |

**Improvement with Momentum Observer:** 12.5x transit RMS, 30x hold error reduction.

### Friction Compensator Gain Sweep (±70% randomization, seed=99)

| gamma_c | gamma_v | Transit RMS [mm] | Worst Transit [mm] | Hold [mm] | Ori [deg] |
|--------:|--------:|-----------------:|-------------------:|----------:|----------:|
| 0 (baseline) | 0 | 1.24 | 2.76 | 2.59 | 4.31 |
| 1 | 0.5 | 1.25 | 2.76 | 2.59 | 4.31 |
| 5 | 2 | 1.25 | 2.74 | 2.58 | 4.31 |
| 10 | 5 | 1.25 | 2.72 | 2.58 | 4.31 |
| 20 | 10 | 1.26 | 2.70 | 2.59 | 4.31 |
| 50 | 20 | 1.27 | 2.71 | 2.59 | 4.31 |
| 5+obs | 2 | 0.10 | 0.31 | 0.24 | 4.35 |

**Finding:** Friction compensator alone is ineffective in short tests (~20s). Adaptation is too slow for the gradient descent law to converge. Momentum observer dominates.

### Momentum Observer Gain Sweep (±70% randomization, seed=77)

| K_o | Transit RMS [mm] | Worst Transit [mm] | Hold Err [mm] | Ori [deg] |
|----:|-----------------:|-------------------:|--------------:|----------:|
| 0 (baseline) | 1.33 | 2.99 | 2.75 | 0.85 |
| 10 | 0.16 | 0.32 | 0.07 | 0.00 |
| 30 | 0.10 | 0.32 | 0.05 | 0.00 |
| **50** | **0.08** | **0.32** | **0.07** | **0.00** |
| 100 | 0.07 | 0.32 | 0.07 | 0.00 |
| 200 | 0.56 | 1.39 | 0.74 | 4.36 |
| 50+fric | 0.08 | 0.32 | 0.07 | 0.00 |

**Optimal K_o: 30-100.** Below 30 → slow convergence. Above 200 → noise amplification causes degradation.

---

## Recommended Production Configuration

```yaml
controller:
  enable_gravity_compensation: true
  enable_coriolis_compensation: true
  enable_inertia_compensation: true

  joint_pid:
    enabled: true
    gains_yaml: "joint_pid_gains.yaml"  # kp_pos=200, kd_pos=28, kp_vel=1

  momentum_observer:
    enabled: true
    K_o: 50.0
    max_tau_dist: 20.0

# task_list.yaml gains:
# jpos:  kp=100, kd=20
# ee_pos: kp=3200, kd=113
# ee_ori: kp=3200, kd=113
```

**Performance with ±50% domain randomization:**
| Metric | PID Only | PID + MomObs | Improvement |
|--------|--------:|------------:|------------:|
| Transit RMS | 1.00 mm | 0.08 mm | 12.5x |
| Worst transit | 2.08 mm | 0.32 mm | 6.5x |
| Hold error | 1.85 mm | 0.06 mm | 30.8x |
| Orientation | 0.58° | 0.00° | ∞ |

## Test Infrastructure (Phase 3-4)

- **Compensator classes:** `wbc_util/include/wbc_util/adaptive_friction_compensator.hpp`, `momentum_observer.hpp`
- **Integration:** `wbc_architecture/src/control_architecture.cpp` → SolverUpdate() between WBIC and PID
- **Test file:** `optimo_controller/test/test_domain_randomization.cpp`
- **Tests:** `CompensatorComparison` (5 seeds), `FrictionCompGainSweep`, `MomentumObserverGainSweep`
