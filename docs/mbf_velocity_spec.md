# MBF Velocity Task — Observation, Action, DR & Reward Spec

Source of truth: `src/tasks/velocity/velocity_env_cfg.py` +
`src/tasks/velocity/config/mbf/env_cfgs.py` +
`src/assets/robots/mbf/mbf_constants.py`.

Tasks: **`Mbf-Rough`**, **`Mbf-Flat`**.

---

## Simulation / control timing

| Parameter | Value |
|---|---|
| Physics `dt` | 0.005 s (200 Hz) |
| Decimation | 4 |
| Control `dt` | 0.02 s (50 Hz) |
| Episode length | 20 s (training) |
| Action type | Joint position targets (`JointPositionActionCfg`) |
| Action scale | `0.25 * effort_limit / stiffness` = **0.125** rad per joint |
| Action offset | default joint pose (`use_default_offset=True`) |
| Actuators | Delayed PD position (`DelayedActuator` over `BuiltinPositionActuator`) |
| Nominal PD | Kp=20, Kd=0.5 |
| Nominal armature | 0.003 |
| Delay buffer | 0–4 physics steps (0–20 ms), randomized each reset |

---

## Actions

| Term | Type | Targets | Notes |
|---|---|---|---|
| `joint_pos` | `JointPositionActionCfg` | all 12 joints (`.*`) | Relative to default pose; clipped by runner if configured |

Commanded joints (order follows MJCF):  
`fl/fr/rl/rr` × `{shoulder, hip, knee}`.

---

## Observations — stacking / history

Group-level history (not per-term):

| Group | `history_length` | `flatten_history_dim` | `concatenate_terms` | Corruption |
|---|---|---|---|---|
| **actor** (`Mbf-Rough`) | **5** | **True** (flatten time into feature dim) | True (`dim=-1`) | On (train) |
| **actor** (`Mbf-Flat`) | **1** | True | True | On (train) |
| **critic** (both) | **1** | True | True | Off |

Stack type: **group-level temporal history**, flattened to
`(num_envs, obs_dim * history_length)`.  
Not a separate time axis; not per-term history.

Play mode disables actor corruption.

---

## Actor observation (onboard / deployable)

Order is concatenation order. Dims are per frame.

| # | Term | Dim | Noise (train) | Notes |
|---|---|---|---|---|
| 1 | `base_ang_vel` | 3 | U[-0.2, 0.2] | IMU gyro |
| 2 | `projected_gravity` | 3 | U[-0.05, 0.05] | from base orientation |
| 3 | `command` | 3 | — | twist `[vx, vy, ωz]` |
| 4 | `phase` | 2 | — | `[sin, cos]` of 0.6 s gait clock; zero when standing |
| 5 | `joint_pos` | 12 | U[-0.01, 0.01] | relative to default |
| 6 | `joint_vel` | 12 | U[-1.5, 1.5] | relative |
| 7 | `actions` | 12 | — | previous action |

**Per-frame dim = 47.**

| Task | History | Actor input size |
|---|---|---|
| `Mbf-Rough` | 5 | **235** |
| `Mbf-Flat` | 1 | **47** |

**Not in actor:** `base_lin_vel`, `height_scan` (blind locomotion for sim2real).

---

## Critic observation (privileged, train-only)

Same base terms as actor (corruption off), then extras:

| Term | Dim | Notes |
|---|---|---|
| `base_lin_vel` | 3 | IMU / root lin vel (privileged) |
| `height_scan` | grid | **Currently disabled** (deliberate blind-locomotion choice — the `terrain_scan` `RayCastSensor` is removed from the scene entirely in `mbf_rough_env_cfg()`, not just from observations). When enabled: raycast `1.6×1.0` m @ 0.1 m → 17×11 = **187**; scaled by `1/max_distance`. Rough only. |
| `foot_height` | 4 | site z of `fl/fr/rl/rr_foot` |
| `foot_air_time` | 4 | from `feet_ground_contact` |
| `foot_contact` | 4 | contact flags |
| `foot_contact_forces` | 12 | 4×3 net force |

Critic history = 1. Flat removes `height_scan`.

---

## Domain randomization

### Startup (once per env)

| Event | What | Operation | Range |
|---|---|---|---|
| `foot_friction` | foot geom friction | abs | **[0.2, 1.7]** (shared across 4 feet) |
| `encoder_bias` | joint encoder bias | — | **±0.015 rad** |
| `base_com` | chassis COM offset | add | **±0.03 m** on x/y/z |
| `base_mass` | chassis mass | scale | **[0.8, 1.25]** |
| `joint_armature` | all DOFs | abs | **[0.002, 0.012]** |
| `joint_damping` | viscous damping | abs | **[0.05, 0.30]** |
| `joint_friction` | Coulomb `frictionloss` | abs | **[0.02, 0.30]** |
| `pd_gains` | actuator Kp / Kd (all 3 DelayedActuator groups) | scale | Kp **[0.8, 1.2]**, Kd **[0.7, 1.3]** |
| `imu_mount_offset` | chassis orientation (simulates IMU mounting error) | compose w/ default | roll/pitch/yaw **±0.05 rad** |
| `effort_limits` | actuator effort limit (all 3 DelayedActuator groups) | scale | **[0.7, 1.15]** |
| `joint_default_pos` | qpos0 (physical zero-offset, distinct from `encoder_bias`'s reading-only bias) | add | **±0.03 rad** |
| `link_inertia` | mass + inertia, jointly (all 16 leg links) | `pseudo_inertia` alpha | **[-0.1, 0.1]** (≈ ±20% density) |

### Reset (each episode)

| Event | What | Range |
|---|---|---|
| `reset_base` | root xy / yaw | xy ±0.5 m, yaw ±π |
| `reset_robot_joints` | joint pos / vel noise | pos **±0.1 rad**, vel **±0.5 rad/s** |
| `actuator_delay` | synced command lag | **0–4** physics steps (0–20 ms) |

### Interval

| Event | Interval | Range |
|---|---|---|
| `push_robot` | every **4–6 s** | lin vel ±0.6/0.6/0.4; ang ±0.6/0.6/0.8 |
| `wind_gust` | sustained wrench on chassis, 0.1–0.3s hold, 3–6s cooldown | force **±15 N**, torque **±2 N·m** |

### Observation noise

Actor uniform noise (see table above); enabled when `enable_corruption=True`.

### Play-mode changes

- No `push_robot`
- No actor noise
- Optional `randomize_terrain` on reset (rough play)

---

## Rewards

Weights differ for rough vs flat where noted.

| Term | Weight (Rough) | Weight (Flat) | Sign / role | Key params |
|---|---|---|---|---|
| `track_linear_velocity` | **1.25** | 1.0 | + track `vx,vy` (+ `vz` penalty) | `std=0.4` rough / `√0.25` flat |
| `track_angular_velocity` | 1.0 | 1.0 | + track `ωz` | `std=√0.5` |
| `foot_air_time` | **1.0** | 0.5 | + single-stance air/contact time | threshold 0.35 s |
| `foot_gait` | **0.75** | 0.5 | + trot phase match | period 0.6; offset `[0, 0.5, 0.5, 0]` |
| `pose` | 0.35 | 0.5 | + default pose (speed-dependent stds) | see std tables below |
| `body_orientation_l2` | **−1.5** | −1.0 | − tilt (projected gravity xy) | body=`chassis` |
| `foot_clearance` | **−2.0** | −1.0 | − swing height error × foot speed | target **0.12** / 0.10 m |
| `stand_still` | −1.0 | −1.0 | − joint drift when cmd≈0 | threshold 0.1 |
| `foot_swing_height` | **−0.75** | −0.4 | − peak swing ≠ target at landing | target 0.12 / 0.10 m |
| `foot_slip` | −0.4 | −0.25 | − xy foot vel in contact | |
| `action_rate_l2` | −0.03 | −0.05 | − Δaction | |
| `body_ang_vel` | −0.05 | −0.05 | − base roll/pitch rate | |
| `angular_momentum` | −0.025 | −0.025 | − root angmom | |
| `soft_landing` | −2e-3 | −1e-3 | − impact force at first contact | |
| `joint_acc_l2` | −2.5e-7 | −2.5e-7 | − joint accel | |
| `joint_pos_limits` | −10.0 | −10.0 | − soft limit violation | |
| `is_terminated` | −200.0 | −200.0 | − episode failure | sparse |

### Pose stds (rad)

| Regime | Shoulder | Hip | Knee |
|---|---|---|---|
| Standing | 0.05 | 0.10 | 0.15 |
| Walking | 0.20 | 0.45 | 0.60 |
| Running | 0.25 | 0.55 | 0.70 |

Walking threshold 0.1, running threshold 1.5 (cmd speed).

**Priority on rough:** velocity tracking + foot clearance / air time / gait, then stay upright; posture is softer so legs can adapt.

**Note (`foot_clearance`, `foot_swing_height`):** both measure foot height in
**world-frame Z** against a fixed `target_height`, which assumes flat ground
at z=0. On a stair or slope the local ground under a foot is not at z=0, so
on rough terrain these measure height above the world origin plane rather
than clearance above actual ground. A terrain-relative rewrite was trialled
and reverted (2026-08-29) — see CLAUDE.md before revisiting.

---

## Terminations

| Term | Condition |
|---|---|
| `time_out` | episode length |
| `fell_over` | base tilt > 70° |
| `illegal_contact` | non-foot/non-knee geom vs terrain, force > 10 N |

---

## Terrains

### `Mbf-Flat`

Plane only. No terrain curriculum. No height scan.

### `Mbf-Rough`

Generator from `ROUGH_TERRAINS_CFG` with slopes removed; MBF-scaled features:

| Sub-terrain | Proportion | Notes |
|---|---|---|
| `flat` | 0.2 | |
| `pyramid_stairs` | 0.2 | step height 0–0.08 m, width 0.25 m |
| `pyramid_stairs_inv` | 0.2 | step height 0–0.06 m |
| `random_rough` | 0.1 | noise 0.01–0.06 m |
| `wave_terrain` | 0.1 | amp 0–0.10 m, 3 waves |

Removed: `hf_pyramid_slope`, `hf_pyramid_slope_inv`.  
Curriculum: `terrain_levels` + `command_vel` stages.

---

## Visualization

```bash
python scripts/visualize_robot.py                         # crouched MJCF
python scripts/visualize_robot.py --mode zero_joints
python scripts/visualize_robot.py --mode zero             # live env, zero actions
python scripts/visualize_robot.py --mode random --task Mbf-Rough

python scripts/visualize_terrain.py                       # MBF rough grid
python scripts/visualize_terrain.py --preset default
python scripts/visualize_terrain.py --rows 4 --cols 4
```
