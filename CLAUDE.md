# mbf_rl_lab — project notes for Claude

RL training/deployment scaffolding for the **MBF** quadruped on top of
[`mjlab`](https://github.com/mujocolab/mjlab) `1.2.0` (MuJoCo + NVIDIA Warp
GPU physics, i.e. `mujoco-warp`, **not** IsaacLab/PhysX). Task pipeline is
ported from `unitree_rl_mjlab`. See [README.md](README.md) for setup/usage
and [docs/mbf_velocity_spec.md](docs/mbf_velocity_spec.md) for the full
observation/action/reward/DR spec (kept in sync with
`src/tasks/velocity/config/mbf/env_cfgs.py`).

## Hardware

Dev machine GPU: **NVIDIA RTX 4050 Laptop, 6 GB VRAM** (`nvidia-smi`). This is
small relative to typical mjlab/IsaacLab dev hardware (24 GB+ workstation
cards) and is the direct cause of the OOM issue below — always frame memory
budgeting against this 6 GB ceiling, not against reference configs written
for bigger cards.

## Registered tasks

- `Mbf-Flat` — plane terrain. No mesh/heightfield geoms, no raycast sensor.
  Cheap: 4096 envs ≈ **2.5 GB** VRAM.
- `Mbf-Rough` — generated terrain (`ROUGH_TERRAINS_CFG`, stairs/wave/random
  sub-terrains as **MESH** geoms) + a `RayCastSensor` height scanner for the
  critic. Much more expensive — see below.

## Resolved: OOM on `Mbf-Rough` at high `num_envs` (investigated + fixed 2026-08-28)

**Current state: fixed and verified.** `ccd_iterations=50`, the height
scanner left disabled, and a set of new DR events (see below) together train
at `num_envs=4096` on the 6 GB GPU with 1.68 GB free (`4.21/6.05 GB`) —
confirmed with a real end-to-end `scripts/train.py Mbf-Rough
--env.scene.num-envs 4096` run (3 iterations, exit 0, ~17k steps/s, no
NaN/OOM). The investigation below is kept for context on *why*.

**This is a `mujoco-warp` buffer-sizing issue, not evidence that mjlab is
"heavier" than IsaacLab in general.** `Mbf-Flat` at 4096 envs is in fact
lighter than a typical IsaacLab/PhysX footprint. The blowup is specific to
`mujoco-warp`'s convex narrowphase (GJK/EPA) collision code path, which only
activates for **MESH**-type collision pairs — i.e. rough terrain only.

### Root cause #1 (primary, quantitatively confirmed)

`cfg.sim.mujoco.ccd_iterations = 500` is set for `Mbf-Rough`
(`env_cfgs.py`, copied from mjlab's own `go1`/`g1` rough-terrain reference
configs — this value is **not** MBF-specific and was written against bigger
GPUs). In `mujoco_warp/_src/collision_convex.py::convex_narrowphase`, EPA
scratch buffers are allocated per **CUDA graph capture** with shape
`(naccdmax, 6 + MJ_MAX_EPAFACES * ccd_iterations)`, where
`naccdmax = num_envs * nconmax` (mjlab sets `nconmax=35` for rough, unchanged
from the base cfg).

At `num_envs=4096`: `naccdmax = 4096*35 = 143360`. The `epa_face` buffer alone
is `143360 * (6 + 5*500) * 4 bytes = 1,437,040,640 bytes` — this exact number
was reproduced as a hard allocation failure
(`RuntimeError: Failed to allocate 1437040640 bytes on device 'cuda:0'`)
inside `Simulation.create_graph()`. There are 5 more buffers of similar shape
(`epa_vert`, `epa_vert_index`, `epa_pr`, `epa_norm2`, `epa_horizon`) allocated
in the same call — together several GB, before any RL/training memory.

Memory scales **linearly in `ccd_iterations`** and **linearly in
`num_envs * nconmax`**. Dropping `ccd_iterations` from 500 → 50 (mjlab's own
default) was verified empirically to fix it:

| Config | num_envs | Result |
|---|---|---|
| Rough, ccd=500 (current) | 4096 | **OOM** — fails allocating 1.44 GB EPA buffer |
| Rough, ccd=50 | 4096 | OK — 3.66 GB / 6.05 GB used |
| Rough, ccd=50 | 2048 | OK — 1.74 GB |
| Rough, ccd=50 | 1024 | OK — 0.91 GB |

### Root cause #2 (secondary, real, easy to miss)

Independently of `ccd_iterations`, re-enabling the `terrain_scan`
`RayCastSensor` (height scan) at high `num_envs` on rough (mesh) terrain adds
real memory (raycast-vs-mesh broadphase), enough to tip an already
near-full 6 GB budget over the edge. Verified even **with `ccd_iterations`
already reduced to 50**:

| num_envs | height_scan enabled | Result |
|---|---|---|
| 3072 | yes | OK — 4.04 GB |
| 3584 | yes | OK — 4.65 GB |
| 4096 | yes | **CUDA error 700: illegal memory access** (`wp_cuda_graph_begin_capture`) |
| 4096 | no | OK — 3.66 GB |

**This is the likely link to the "NaN faults":** when the exhausted
allocation happens *inside CUDA graph capture*, Warp/CUDA doesn't always
raise a clean Python exception — it can surface as a raw
`illegal memory access` CUDA fault (error 700), which poisons the CUDA
context. Once that happens, subsequent kernel reads of the corrupted
memory pool can plausibly come back as garbage (NaN/Inf) rather than a clean
crash on every occurrence — consistent with intermittent NaNs rather than a
deterministic failure. This was reproduced directly, not inferred from
memory: it is **not** that the height-scanner logic itself is buggy (it
works fine ≤3584 envs); it's the last straw on an already marginal VRAM
budget.

`env_cfgs.py` originally stripped `terrain_scan`/`height_scan` entirely from
`Mbf-Rough` as an experiment to chase root cause #2 — that alone would have
dodged #2 but **not** #1 (`ccd_iterations=500` alone already OOMs at 4096
envs with no height scan at all). **Applied fix:** both are now addressed —
`ccd_iterations=50` (see below) and the height scanner stays disabled (a
deliberate blind-locomotion choice, not required for the OOM fix alone).

### `ccd_iterations=50`: fidelity check before applying

Before dropping `ccd_iterations`, ran a controlled 300-step / 512-env A/B
test (same seed, same random action sequence, only `ccd_iterations` varied:
500/150/100/50) measuring `mujoco_warp` `Contact.dist` (negative =
penetration):

- **Mean penetration depth was identical to 3 decimal places** (-0.000436 to
  -0.000438 m) across every tested value, and again after the final DR/reward
  changes below (-0.000438 m) — no bulk fidelity loss.
- Worst-case single-contact penetration depth was **not** a usable signal:
  repeat runs at the *same* `ccd_iterations` gave different worst-case values
  (e.g. ccd=500 gave both -0.030 m and -0.069 m across three reruns) — this
  metric is dominated by GPU-parallel run-to-run non-determinism (contact
  compaction order), not by `ccd_iterations`. Don't trust a single-run
  worst-case comparison for this without repeat-run verification.
- No NaN/Inf and no "CCD overflow" warnings at any tested value, including a
  real end-to-end training smoke test (`scripts/train.py Mbf-Rough
  --agent.max-iterations 3 --env.scene.num-envs 4096`, exit 0).

`ccd_iterations` only affects **MESH/HFIELD** collision pairs (GJK/EPA path);
primitive-primitive contacts (capsule/box/sphere) use closed-form solutions
and are unaffected regardless of this setting
(`mujoco_warp/_src/collision_convex.py`:
`epa_iterations = 16 if nboxbox == ncollision else m.opt.ccd_iterations`).

### New domain-randomization events (added on top of the existing sim2real DR)

All added in `_add_sim2real_domain_rand()`, `env_cfgs.py`:

| Event | What | Why |
|---|---|---|
| `imu_mount_offset` | `dr.body_quat` on chassis, ±0.05 rad roll/pitch/yaw | Simulates IMU mounting misalignment — `imu_ang_vel`/`projected_gravity` were previously read as perfectly aligned. |
| `effort_limits` | Actuator effort-limit scale [0.7, 1.15] | Payload/battery-sag/motor-wear robustness. |
| `joint_default_pos` | qpos0 offset ±0.03 rad | Joint calibration/zero-offset error (distinct from `encoder_bias`, which only biases the *reading*, not the physical setpoint). |
| `link_inertia` | `pseudo_inertia` (alpha_range ±0.1) on all 16 leg links | Physically-consistent (mass+inertia together) per-link density variation, manufacturing tolerance. |
| `wind_gust` | `apply_body_impulse` on chassis, ±15 N / ±2 N·m, 0.1-0.3s sustained, 3-6s cooldown | Sustained external wrench (wind/bump), complementary to `push_robot`'s instantaneous velocity kick. |

Explicitly **not** added (user's call): soft joint-limit randomization
(`dr.joint_limits`), terrain-geom friction randomization, and
`reset_root_state_from_flat_patches`.

**Gotcha found while adding these — `effort_limits`:** mjlab's own
`dr.effort_limits` raises `TypeError` on any MBF actuator because MBF wraps
every actuator in `DelayedActuatorCfg` (for sim2real command latency) and
`dr.effort_limits` — unlike `dr.pd_gains`, which already does this — never
unwraps `DelayedActuator.base_actuator` before its `isinstance` check. Fixed
locally in `src/tasks/velocity/mdp/dr_extras.py::effort_limits_delayed`
(same logic, one extra unwrap line) rather than patching the installed
package.

**Gotcha found while adding these — `link_inertia` / `pseudo_inertia`:**
`dr.pseudo_inertia` diagonalizes the perturbed inertia tensor via
`torch.linalg.eigh`, batched over `num_envs * num_bodies_selected`. On this
GPU/driver/torch/cuSOLVER combination, that batched `eigh` has a **~0.27 MB
workspace cost per 3x3 matrix** (measured directly, not estimated) — a batch
of 16384 alone needs ~4.5 GB. mjlab's own internal chunk limit
(`_MAX_EIGH_BATCH=16384` in `mjlab/envs/mdp/dr/body.py`) is *itself* one
full OOM-sized chunk on a 6 GB card: at `num_envs=4096 × 16 bodies =
65536`, this reproduced as a **`torch.OutOfMemoryError: Tried to allocate
4.14 GiB`** inside `_decompose_pseudo_inertia_J`. Fixed locally in
`dr_extras.py::pseudo_inertia_chunked`, which calls mjlab's `dr.pseudo_inertia`
in sub-batches of 128 envs at a time (128×16=2048 per call, empirically
~0.57 GB peak, safe) — this only runs once at `mode="startup"`, so the extra
Python-level loop has no training-time cost. If you ever see an unexplained
multi-GB `torch.OutOfMemoryError` (not a `mujoco_warp`/Warp allocation
error) from an event at env construction, suspect a batched
`torch.linalg.eigh`/`svd`/similar LAPACK call before suspecting the
simulator.

### REVERTED (2026-08-29): terrain-relative rewrite of `feet_clearance` / `feet_swing_height`

Both terms were rewritten mid-session to measure foot height relative to
local ground instead of world-Z, then **reverted at the user's request** —
`rewards.py` is back to its pre-session state (`feet_clearance` = world-Z
function, `feet_swing_height` = original peak/target class, neither defining
`reset`). Do not re-apply without being asked.

The diagnostic measurements taken during that work are still valid
properties of the *current* (reverted) code, and are worth knowing:

- `feet_clearance`/`feet_swing_height` measure world-frame Z against a fixed
  `target_height`, which implicitly assumes flat ground at z=0. Once the
  robot climbs a stair/slope the local ground is no longer at z=0, so on
  rough terrain these measure "height above the world origin plane", not
  clearance above actual ground. Neither term reads `height_scan`; both use
  raw `site_pos_w[...,2]`.
- In `feet_swing_height`, `data.found == 0` (used to accumulate the peak) and
  `current_air_time > 0` (what `compute_first_contact()` derives from)
  **disagree on ~53% of (env, foot, step) samples**, and ~76% of air phases
  last a single control step (contact chatter). Measured over 110k landings,
  62% had `peak_heights == 0` at the scoring instant, giving `error = -1`
  (cost 1.0 per foot) as a floor rather than a measurement.
- Neither class defines `reset()`, and `RewardManager._prepare_terms` only
  registers class terms for the reset pass if they define it — so
  `peak_heights` is not cleared between episodes.

`phase` (`observations.py`) was checked and has no height dependency at all
(pure time-based gait clock) — unaffected either way.


## Resolved: "critic observation group contains NaN" crash during real training (2026-08-29)

Real `scripts/train.py Mbf-Rough --env.scene.num-envs 4096` runs (current
`venv`, py3.12) hit `rsl_rl`'s `check_nan()` failing on the **critic**
observation group, twice in a row, both within the first ~100 iterations.

**Reproduction attempts:** ran 650+ iterations across two separate runs
(150 + 500) at the identical config (`num_envs=4096`, `ccd_iterations=50`,
`seed=42` — verified byte-identical to the crashing runs' saved
`params/env.yaml`) with `--enable-nan-guard` armed (which watches
`qpos`/`qvel`/`qacc`/`sensordata` after **every physics substep**, far finer
grained than a per-control-step check) — zero reproductions, zero dumps.
Conclusion: this is a genuine but rare, GPU-execution-order-dependent event
(consistent with the non-determinism already established in the
`ccd_iterations` investigation above — same seed, different runs, different
numerics), not a deterministic bug tied to a specific iteration. Small-sample
luck, not evidence the bug doesn't exist.

**Root cause of why it crashes (not just when):** found by reading mjlab's
manager source, not by reproducing the trigger.

- `ManagerBasedRlEnv.step()` computes `termination_manager.compute()` and
  `reward_manager.compute()` on the **same post-physics, pre-reset** data.
  If that data has already diverged to NaN, existing terminations
  (`bad_orientation`: `torch.acos(x).abs() > limit_angle`, etc.) use
  ordinary comparisons — which are IEEE-754 **always False** against NaN.
  A diverged env can therefore silently fail *every* termination check, is
  never reset, and its NaN state rides through to the next observation.
- `RewardManager.compute()` (mjlab's own code) already guards against
  exactly this: `value = torch.nan_to_num(value, nan=0.0, posinf=0.0,
  neginf=0.0)` per term, before summing — this is *why* the crash was
  reported for the observation group and not rewards.
- `ObservationManager` has the identical mechanism available
  (`ObservationGroupCfg.nan_policy: Literal["disabled","warn","sanitize",
  "error"]`, plus `nan_check_per_term: bool = True` which — when enabled —
  sanitizes *before* a term enters its delay/history circular buffer, so a
  single NaN frame can't contaminate several subsequent frames of history)
  — but it **defaults to `"disabled"`**, and this project's `actor`/`critic`
  `ObservationGroupCfg`s never set it. This is the actual, verified gap:
  mjlab ships the fix, it's just off by default and unused here.

**Applied fix (both, defense-in-depth):**

1. `nan_policy="warn"` on both `actor` and `critic` `ObservationGroupCfg`
   (`velocity_env_cfg.py`) — matches the reward manager's existing
   sanitize-before-crash behavior, plus logs which term/envs were hit
   instead of silently hiding it.
2. New termination `local_mdp.numerical_divergence`
   (`src/tasks/velocity/mdp/terminations.py`), registered in
   `mbf_rough_env_cfg()` (applies to `Mbf-Flat` too, same as
   `illegal_contact`). Checks `root_link_vel_w`/`joint_vel` for
   `isnan`/`isinf` (which, unlike ordinary comparisons, correctly detect
   NaN) plus absurd-but-still-finite magnitude thresholds, and forces a
   reset — fixing this at the *source* env the same step it diverges,
   rather than only sanitizing the symptom every step after. Shows up as
   `Episode_Termination/numerical_divergence` in training logs (0.0 in
   ~1150+ iterations tested so far, as expected for a rare event).

Verified with a real 500-iteration `--env.scene.num-envs 4096` run after
applying both — trains cleanly, `numerical_divergence` termination properly
registered and logged (stayed at 0.0 the whole run — the rare event didn't
recur). Combined with the pre-fix reproduction attempts, this is 1150+
total clean iterations across three separate `num_envs=4096` runs, and the
live trigger was never caught in any of them (no nan_guard dump, no
`numerical_divergence` firing, no "Sanitizing to 0" log). Since the actual
trigger wasn't caught live, this fix is verified *mechanistically* (the gap
it closes is real and confirmed by reading the manager source, not
inferred) rather than by reproducing a before/after crash. Further blind
reproduction attempts have low expected value at this point — if it recurs
in real training, `--enable-nan-guard` plus the new
`Episode_Termination/numerical_divergence` metric will now both give a
direct signal instead of a hard crash.

## Performance: `njmax=1500` was ~11x oversized, cost real compute not just memory (2026-08-29)

Rough terrain trains at ~40-60% of flat's throughput at the same `num_envs`
(flat: ~2.3s/iter, ~42k steps/sec; rough: ~5.5s/iter, ~17.7k steps/sec). Most
of that gap is inherent (MESH/HFIELD collision pairs need mujoco-warp's
iterative GJK/EPA path; flat's plane geom uses closed-form primitive
collision — see the `ccd_iterations` section above for the mechanism). But
one piece of it wasn't inherent: `njmax=1500`, inherited unchanged from the
shared base config.

**Confirmed by reading `mujoco_warp/_src/solver.py`:** the Newton solver's
GPU kernels launch with `dim=(d.nworld, d.njmax)` — the launched thread grid
size is fixed at the *allocated* `njmax`, not the actual per-step active
constraint count (`d.nefc`). An oversized `njmax` is therefore wasted GPU
work on every single step, not just wasted memory headroom — unlike
`nconmax`/`ccd_iterations`, which are primarily memory concerns.

**Measured actual usage:** a 1000-step/4096-env stress test at the hardest
terrain level with mixed violent/idle action magnitudes (checking
`d.nefc`/`d.nacon` directly) found `max_nefc_per_world=138` — under 10% of
the allocated 1500, and no NaN.

**Fix:** `cfg.sim.njmax = 400` in `mbf_rough_env_cfg()` (~2.9x margin over
the measured peak; values below ~300 gave no further measured speedup, so
there was no reason to cut it closer). Verified with a real
`scripts/train.py` run (not just env construction):

| `njmax` | Steps/sec | vs. baseline |
|---|---|---|
| 1500 (previous) | ~17,700 | — |
| 400 (applied) | ~19,700-19,800 | **+~12%** |
| 300 | ~19,600 | +~11% |
| 150 | ~20,050 | +~13% (margin too thin — only ~1.1x the measured peak) |

`nconmax=35` was checked too (`max_nacon_total=14,443` peak across all 4096
worlds combined, vs. `naconmax=143,360` allocated) but left unchanged — it's
already the smallest value mjlab's own reference configs use, and reducing
it further gave no measurable additional speedup in a quick check
(`njmax=300+nconmax=20` was within noise of `njmax=300` alone).

## Performance: leg links used full URDF mesh collision — swapped for primitives (2026-08-29)

Checked `mbf_v2/mbf.xml`: `chassis`, `*_shoulder_link`, `*_hip_link`,
`*_knee_link` all had `class="collision"` geoms using `mesh="..."` (full
URDF-derived triangle meshes) — only the foot used a primitive (sphere).
Cross-referenced against `mujoco_warp/_src/collision_driver.py`'s
`MJ_COLLISION_TABLE`: any pair involving `MESH` is classified `CONVEX`
(the same iterative GJK/EPA path as `ccd_iterations` above) regardless of
what the *other* geom is — including `HFIELD, MESH` and `BOX, MESH`. So
every hip/knee/shoulder-vs-terrain candidate pair was forced onto the
expensive path purely because of the robot's own collision geometry, not
because the contact itself was inherently complex. `SPHERE,BOX`,
`CAPSULE,BOX`, `SPHERE,SPHERE`, `CAPSULE,CAPSULE` etc. are all `PRIMITIVE`
(closed-form) in the same table.

**Fix:** replaced the shoulder/hip/knee mesh collision geoms with capsule
(hip, knee) and box (shoulder) primitives, sized from the actual mesh
geometry rather than guessed:

- Loaded each unique mesh (`shoulder.stl`, `lefthip.stl`/`rightthip.stl`,
  `leftknee.stl`/`rightknee.stl`) with `trimesh`, computed axis-aligned
  bounding-box extents and centroid in the mesh's own local frame.
- `shoulder`: extents ~(0.061, 0.095, 0.080) — no dominant long axis → box,
  half-extents = extents/2.
- `hip`: extents ~(0.080, 0.06-0.08, **0.175**) — ~2.2-2.9x elongated along
  Z → capsule along local Z, radius = area-equivalent circle from the two
  perpendicular extents (`sqrt(rx*ry)`), half-length = `z_extent/2 - radius`
  (so the capsule's total cap-to-cap length matches the mesh's Z-span
  exactly, not the Z-span plus extra from the end caps).
- `knee`: extents ~(0.042, 0.018, **0.177**) — ~4-10x elongated along Z,
  same capsule method — an excellent capsule fit.
- The hip/knee collision geoms had no `pos`/`quat` offset (mesh-local frame
  == body-local frame), so their computed centroid/size could be used
  directly. The shoulder geom (and the mirrored right-side hip, which reuses
  `rightthip.stl` under a rotation) **does** carry a `quat` in the MJCF for
  left/right mirroring — reused that exact same `quat` on the new primitive
  and rotated the mesh-local centroid by it (`R(quat) @ centroid`) to get
  the correct `pos`, so the primitive lands in the identical body-frame
  location/orientation the mesh did. Verified by recompiling the model and
  reading back each new geom's resolved `type`/`size`/`pos`.
- Visually verified with an offscreen `mujoco.Renderer` render (collision
  geoms only, and overlaid semi-transparent with the visual mesh) rather
  than assuming the math was right — legs show a clean thick-hip →
  thin-knee taper into the foot sphere, well contained within the original
  mesh silhouette, no floating or detached primitives.

**Result** (`scripts/train.py Mbf-Rough --env.scene.num-envs 4096`, after
the `njmax=400` fix above): **~19,700 → ~24,300-24,400 steps/sec (+~24%
more)**. Combined with the `njmax` fix, rough-terrain throughput is now
~17,700 → ~24,400 steps/sec, **+~38%** total, with no terrain-fidelity
change and a geometrically-justified (not guessed) collision approximation
on the robot's own links.

One-time cost to be aware of: the first run after this change took several
minutes longer than usual with no output — this is `mujoco_warp` JIT-
compiling kernels for collision-pair types (box-box, capsule-box, etc.)
that never existed in the model before (only mesh-involving pairs existed
previously). This is a one-time cost cached to `~/.cache/warp/`; subsequent
runs start fast immediately. Don't mistake this for a hang.

## REVERTED (2026-08-29): `feet_swing_height` rewrite

`feet_swing_height` was rewritten mid-session (liftoff-latched reference,
`min_air_time` validity gate, `swing_scored_frac` metric) and then
**reverted at the user's request**. `rewards.py` is back to its pre-session
state. Do not re-apply without being asked.

**The audit findings that motivated it remain true of the current code** and
are worth knowing before touching this term again. Audited by reading
mjlab's `RewardManager._step_reward` buffer (raw*weight, populated once per
`env.step()`) — **do not** re-call a stateful term's `__call__` to measure
it, that double-invokes its side effects and corrupts its own tracking state
(this produced a bogus "term is dead, 0.0" reading mid-investigation).

- `joint_acc_l2` and `foot_clearance` are **fine** and only *look* big.
  `joint_acc_l2` raw is ~2.7e5 (sum of 12 squared accels, ~92 rad/s² each)
  times a deliberately tiny -2.5e-7 weight = 0.067 weighted, rank ~8. That
  large-raw × tiny-weight shape is the standard formulation, not a bug.
- `feet_swing_height` measured raw 3.96 — ~10x the next term and 7x the
  primary tracking objective. Over 110k landings: 44% had **negative** swing
  height, 62% had `peak_heights == 0`, and 79% of the term's cost came from
  those landings. Contributing causes: `data.found == 0` and
  `current_air_time > 0` disagree on 52.8% of samples, and ~76% of air phases
  last ≤1 control step (chatter; measured air-time median 0.01 s vs an
  intended ~0.26 s trot swing).


### Gotcha: `Metrics/twist/error_vel_*` is an accumulator, not a mean

`velocity_command.py::_update_metrics` does `metrics["error_vel_xy"] +=
|err| / max_command_step` every step, where `max_command_step =
resampling_time_range[1] / step_dt` = 400. Over a full 1000-step episode that
is `2.5 * mean|err|`, so a logged 0.85 means **0.34 m/s** actual mean error.
Reading it as a mean makes tracking look ~2.5x worse than it is (it
reconciles exactly with `track_linear_velocity = exp(-err²/std²)`).

### Watch item: terrain curriculum starts by collapsing

`terrain_levels` falls 3.49 → ~0 over the first ~400 iterations, then
recovers slowly. This is expected, not a bug: `terrain_levels_vel` promotes
only if net displacement from the env origin exceeds `size[0]/2 = 4.0 m` in a
20 s episode. An idealized *perfect* command-tracker (pure kinematics, no
physics) clears 4 m in ~60% of episodes at stage-0 command ranges, so the
threshold *is* achievable — the early collapse reflects policy quality, not a
mis-set threshold. Expect it to unlock as tracking error falls.

## Multiple robot generations

`src/assets/robots/mbf/` (v1 MJCF) and `src/assets/robots/mbf_v2/` (new,
currently referenced by `mbf_constants.py` — URDF-derived kinematics/mesh
collision, LeggedGym-Ex-identified actuator params) both exist. `git status`
shows `mbf_constants.py` currently points at `mbf_v2`; treat `mbf_v2` as the
active model unless told otherwise.

## Logs / experiment tracking

- `logs/rsl_rl/<experiment_name>/<timestamp>/` — checkpoints (`model_N.pt`),
  `policy.onnx`, tensorboard events, `params/{env,agent}.yaml` (the actual
  resolved config for that run, including `num_envs` — check this first when
  debugging a specific past run instead of assuming CLI defaults).
- `wandb/offline-run-*/` — offline W&B runs; `files/mjlab_mbf.diff` snapshots
  the git diff at run time, `files/model_*.pt` symlink back into `logs/`.
  Two consecutive rough runs on 2026-08-25 (`69zxoza5`, `66hr0tvz`) died
  after ~800 and ~300 iterations respectively with no local
  `logs/rsl_rl/mbf_velocity_rough/` directory surviving — consistent with a
  hard process kill (OOM/CUDA fault) rather than a graceful stop.
