# Actuator system identification

Identifies the MBF joint actuator model (`armature`, `damping`, `frictionloss`,
command lag) from real-hardware bench logs, fitting **against the actual MuJoCo
plant** so the numbers can go straight into the MJCF.

Migrated here from two places that are no longer the source of truth:

| was | now |
|---|---|
| `mbf_ws/src/scripts/sinewave_analyze.py` | [`plot_log.py`](plot_log.py) — unchanged, plots a raw log |
| `LeggedGym-Ex/legged_gym/sim2real/identify_actuator.py` + `leg_inertia.py` | [`identify_actuator.py`](identify_actuator.py) — rewritten, see below |
| _(nothing — the standup log was unanalysed)_ | [`check_standup.py`](check_standup.py) — new |
| `LeggedGym-Ex/legged_gym/scripts/mbf_standup_replay.py` (Isaac Gym) | [`replay_standup.py`](replay_standup.py) — MuJoCo port |
| `~/Downloads/sysid_data/` | [`data/sysid/`](../../data/sysid/) |

## Data

`data/sysid/` holds bench logs from the real robot, all at 200 Hz:

- `{shoulder,hip,knee}_sine_test_log.csv` — 10 s single-joint sine tracking on
  the `fl` leg, 1.5 Hz drive, amplitude 0.30/0.40/0.40 rad, Kp=20 Kd=0.5.
  Real tracking error is 27/31/32 mrad RMS.
- `standup_mit_log.csv` — 5 s whole-body standup at 100 Hz, all 12 joints,
  Kp=20 Kd=0.5 (logged per-sample, confirming the sine tests' assumed gains).
  Not usable for damping/friction/armature — the legs are loaded against the
  ground, so a replay measures contact modelling, not the actuator. It *is* the
  only data covering torque delivery, all 12 actuators, and load; see
  [`check_standup.py`](check_standup.py) and Results below.

Two caveats about these logs, both discovered the hard way:

1. **The torque channels are unusable in the sine logs, but fine in the
   standup log.** In the sine logs `tau` is identically zero and `tauest`
   correlates with the commanded torque at only 0.04–0.15 — genuinely garbage,
   so torque there must be reconstructed from the PD law. In the standup log
   `tauest` is a real measurement: it recovers the commanded torque with
   correlation up to 0.99 at a scale factor of 7.8 (median), i.e. it is reported
   **motor-side** and joint torque is `tauest * 8` for the GIM6010-8's 8:1
   reduction. It is not "mis-scaled by 100x" as the old notes claimed.
2. **The two datasets do not even share a joint convention with each other,
   let alone with the MJCF.** In the sine logs, shoulder `q` spans
   [-1.71, -1.11] rad against a model range of [-1.047, 1.047], and hip also
   falls partly outside its range. In the standup log, shoulder and hip match
   the MJCF well (final hip 0.70–1.06 vs the model's 0.70 standing angle) but
   the **knees carry a ~-2.7 rad zero offset** (final +1.04…+1.34 vs the
   model's -1.40). It is an offset, not a sign flip. Any script feeding logged
   angles into the model must map conventions first.
   `identify_actuator.py` sidesteps this by fitting gravity as
   `A·sin(q) + B·cos(q)`, which is offset- and orientation-invariant.

## Why the old fit was replaced

The LeggedGym-Ex version fit a hand-rolled 1-DOF ODE
(`I·q̈ = τ − b·qd − c·tanh(qd/0.05)`) and then recovered armature by subtracting
a structural inertia from `leg_inertia.iy_about_hip()`. Three problems:

- `iy_about_hip` is derived for the **hip** axis but was applied unchanged to
  all three joints — hence the identical `I_struct = 0.0089` printed in every
  section of `sim2real.md`. The real composite inertias differ by 8x.
- It uses URDF link masses (hip 0.6 / knee 0.15 / foot 0.025 kg) that `mbf_v2`
  has since changed to 0.7 / 0.1 / 0.01 kg.
- `c·tanh(qd/0.05)` is not MuJoCo's `frictionloss`, which is a solver-side
  stiction constraint, and the ODE was integrated outside MuJoCo, so neither the
  friction semantics nor the discretisation matched the sim the values were
  destined for. (This turned out *not* to bias the answer much — see Results —
  but that was luck, not design.)

## Method

Each bench log drives one joint while the rest are held, which is exactly a
1-DOF system. The identifier:

1. Reads `I_struct` — the moving subtree's composite inertia about the joint
   axis — out of the **compiled mjlab model's mass matrix**, at a configurable
   bench pose. No hardcoded inertia.
2. Builds a one-hinge MuJoCo model carrying that inertia, with the env's own
   solver settings (Euler, 10/20 iterations), so `armature` and `frictionloss`
   have their real MuJoCo semantics.
3. Replays the recorded command through it, reproducing the GIM6010-8 firmware
   law — PD recomputed every physics substep from live state, `qdes` held for
   the control tick, **including the velocity feedforward**
   `Kd·(qd_des − qd)`. Omitting the feedforward biases damping.
4. Minimises `q_sim − q_real` over `armature`, `damping`, `frictionloss` and the
   two gravity coefficients, with command lag grid-searched separately.

`--mode pooled` additionally enforces that all three joints share one armature,
since they use the same GIM6010-8 while their structural inertias differ 8x.
This was intended to make the fit over-determined enough to resolve armature.
**It does not work** — see Results. Damping and frictionloss are identified
cleanly either way.

## Usage

```bash
# recommended: one shared armature, per-joint damping/friction/gravity
uv run python scripts/sysid/identify_actuator.py --mode pooled

# per-joint armature, to see the spread the pooled fit resolves
uv run python scripts/sysid/identify_actuator.py --mode per-joint

# score the parameters currently in the repo, no fitting
uv run python scripts/sysid/identify_actuator.py --mode eval-current

# whole-body standup audit: torque delivery, conventions, effort headroom
uv run python scripts/sysid/check_standup.py

# replay the real standup through the mjlab model, overlay sim vs real + plot
uv run python scripts/sysid/replay_standup.py
uv run python scripts/sysid/replay_standup.py --viewer        # watch it live
uv run python scripts/sysid/replay_standup.py --no-knee-flip  # show the flip matters

# how much does the unknown bench pose move I_struct?
uv run python scripts/sysid/identify_actuator.py --mode pooled --pose-sweep

# plot a raw log
uv run python scripts/sysid/plot_log.py data/sysid/hip_sine_test_log.csv --save
```

Runtime is a few minutes per fit (each residual evaluation replays 2000 control
ticks x `--substeps` through MuJoCo).

## Results

All numbers below are replay RMSE against the real logs, fit and scored at the
env's own 5 ms `sim.timestep`.

| parameter set | shoulder | hip | knee | mean |
|---|---|---|---|---|
| values already in the repo | 7.7 | 8.8 | 9.6 | 8.7 mrad |
| identified | 6.2 | 7.3 | 7.9 | **7.2 mrad** |

Identified values, now in `mbf_v2/mbf.xml`:

| joint | damping | frictionloss | lag |
|---|---|---|---|
| shoulder | 0.135 (was 0.14) | 0.228 (was 0.22) | 0 |
| hip | 0.175 (was 0.18) | 0.092 (was 0.08) | 0 |
| knee | 0.176 (was 0.18) | 0.146 (was 0.15) | 0 |

### The headline: the ported values were already right

Damping and frictionloss land within a few percent of what was already in the
repo. The remaining 8.7 -> 7.2 mrad improvement is almost entirely the fitted
gravity term, which the `eval-current` baseline holds at zero — not a damping
correction. **There was no meaningful actuator gap to close here.** Both are
also stable to within 1% across every bench pose tested, so they are genuinely
identified rather than fit artefacts.

### Armature is *not* identifiable from this data

This is the substantive finding, and it invalidates how the original
`sim2real.md` number was derived. Every configuration below produces **exactly
the same 7.2 mrad**:

| condition | I_struct (shoulder) | fitted armature | RMSE |
|---|---|---|---|
| bench pose (0, 0) | 0.00939 | 0.0024 | 7.2 mrad |
| bench pose (-0.8, -1.6) | 0.00557 | 0.0041 | 7.2 mrad |
| bench pose (-1.7, -2.5) | 0.00503 | 0.0047 | 7.2 mrad |
| pose (-0.8, -1.6), 1 ms substeps | 0.00557 | 0.0060 | 7.2 mrad |

Armature moves 2.5x with **zero** change in fit quality. The fit pins the
*total* effective inertia; the split between rotor inertia and the unlogged
bench pose's structural inertia is unconstrained. Pooling across three joints
was expected to break this via their 8x differing structural inertias and does
not, because the unknown pose re-introduces exactly the freedom that pooling
removed.

So `armature` is left at its previous **0.003**. That value was derived
incorrectly, but there is no evidence in this data to move it, and moving a
parameter on a flat likelihood is worse than leaving it. The `joint_armature`
DR range `[0.002, 0.012]` in `env_cfgs.py` already spans the identified band,
which is where this uncertainty belongs.

### Damping is timestep-specific

Refitting at 1 ms substeps wants ~25% less damping (0.099/0.137/0.139) for the
same data, because MuJoCo's implicit damping term `(M + h·D)` under-damps as `h`
grows and needs a larger `D` to compensate. The values in the MJCF are
calibrated for the env's 5 ms step. **If `sim.timestep` ever changes, refit** —
and never mix values fit at different substeps.

Incidentally this explains why the original hand-ported damping was so close:
it was fit against a 5 ms discretisation too.

### What the standup log adds

`check_standup.py` answers three questions the single-joint sweeps cannot, none
of which need a contact model.

**Torque delivery — the actuator model is validated.** On the loaded joints
(knees, peak 3.5–4.6 Nm) the current loop tracks its setpoint at correlation
0.91–0.99, and `tauest * 8` reproduces the commanded PD torque in both scale and
rms. The drive applies what the PD law asks for, so mjlab's
`BuiltinPositionActuatorCfg` — an ideal PD torque source — is a faithful model.
**There is no actuator gap beyond the PD law itself.** This is the strongest
single statement the hardware data supports, and the sine logs could not make it
(their torque channels are dead).

**Effort headroom.** A real standup peaks at 4.63 Nm, 46% of
`EFFORT_LIMIT = 10 Nm`, with bus voltage sagging only 23.55 → 23.10 V. Nothing
saturates, so the effort limit is still unvalidated from above — but 10 Nm is at
least not obviously too low.

**Conventions — two real deployment hazards.**

- All four knees in the standup log carry a **~-2.7 rad zero offset** relative
  to the MJCF (standing at +1.04…+1.34 where the model stands at -1.40).
  Shoulders and hips agree to within 0.36 rad. The sine logs use a third,
  different convention again. This is an offset, **not** a sign flip — see the
  replay section, where getting that wrong is what the base-height check
  catches.
- Reported current/torque sign disagrees with the commanded PD torque on **7 of
  12 joints**. The robot does stand up, so the position loop is fine — this is a
  per-drive *reporting* polarity, not a control error. Consequence: `iq`/`tauest`
  **signs** cannot be compared across joints without per-joint calibration;
  magnitudes are fine.

Neither of these is a simulation parameter, but both would silently break a
sim-trained policy on hardware, so they are worth pinning down in the deployment
layer before the next hardware run.

### Standup replay (`replay_standup.py`)

MuJoCo port of the old Isaac Gym `mbf_standup_replay.py`. It drives the **env's
own actuators** — the spec comes from `Entity(get_mbf_robot_cfg())`, so gains,
effort limit, armature, damping, frictionloss and collisions are exactly what
training uses — with the recorded `qdes`, and overlays sim vs real per joint.
Output: `data/sysid/standup_mjlab_replay.png`.

The old script's `SIGN`/`OFFSET` table does **not** carry over: it targets the
LeggedGym URDF, whose rest pose (hip 1.57, knee -2.5) is not `mbf_v2`'s
(hip 0.70, knee -1.40). For `mbf_v2` the mapping is simply `knee -> -knee`, no
offsets, which `check_standup.py` derives independently.

| joint group | sim-vs-real RMSE | hardware's own tracking error |
|---|---|---|
| hips | 16–36 mrad | 17–30 mrad |
| knees | 71–140 mrad | 60–88 mrad |
| shoulders | 37–106 mrad | 16–25 mrad |

**Hips track the real standup about as well as the hardware tracks its own
command** — this is the load-bearing degree of freedom, and it confirms the
body model, actuator and conventions together.

**Knees have the right trajectory shape but a ~100 mrad steady-state offset**:
sim settles slightly past the command, real settles slightly short of it. That
is a static-load discrepancy (model total mass is 7.42 kg — worth checking
against the real robot) and not something this data resolves.

**The parameter sweep confirms this log cannot identify plant parameters.**
A 4x4 grid over armature 0.002-0.02 (10x) and damping 0.0-0.8 moves the mean
RMSE only from 0.0460 to 0.0467 rad -- a **1.6% spread**. A standup is
quasi-static and contact-dominated, so it barely responds to either. The script
detects this and refuses to let you read the reported `BEST` as meaningful,
which the Isaac Gym original would have done happily. This is now a measurement
rather than an argument.

**Shoulder disagreement is expected and not meaningful.** These logs have no
IMU, so the base pose is unmeasured; the sim's base is settled onto the ground
rather than matched. Lateral stance and yaw are therefore free to differ, and
the shoulders are nearly unloaded with a total range of only ±0.1 rad.

Two things this replay is *not*: it is not an actuator identification (loaded
legs mean contact dominates the residual), and it is not a base-trajectory
comparison (there is no logged base to compare to).

## Known limits of this data

- **One drive frequency (1.5 Hz), one amplitude, one leg.** With a single tone
  the inertial term is collinear with gravity, which is why armature comes out
  unidentifiable. A frequency sweep (chirp, ~0.5-8 Hz) is the single
  highest-value new experiment: inertia dominates at high frequency and would
  pin armature directly.
- **The bench pose of the unlogged downstream joints is unknown**, and it moves
  `I_struct` by 71% (shoulder) and 58% (hip). Only the knee is invariant. This
  is the second half of the armature problem and is trivially fixable: log all
  12 joint positions next time even when driving one.
- **The sine logs are free-swinging, no load** (torque peaks ~0.7 Nm against a
  10 Nm limit). The standup log reaches 4.6 Nm and confirms torque delivery
  there, but still nothing saturates, so gear efficiency near the limit and the
  saturation path are unidentified in both.
- **The standup log cannot identify plant parameters.** Loaded legs mean a
  replay measures the contact model. Identifying per-actuator damping/friction
  spread across all 12 joints would need each joint swept individually, off the
  ground.
- **No IMU**, so observation noise on the base state has no grounding.
