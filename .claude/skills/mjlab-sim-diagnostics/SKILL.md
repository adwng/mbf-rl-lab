---
name: mjlab-sim-diagnostics
description: Diagnose CUDA OOM, illegal-memory-access, and NaN faults in this repo's mjlab (mujoco-warp) envs with real measurements instead of guessing. Use whenever training crashes, hangs, or the user asks "why is this running out of memory / producing NaNs".
---

# mjlab / mujoco-warp simulation diagnostics

This repo runs on `mjlab` 1.2.0, which is MuJoCo + **NVIDIA Warp** GPU
physics (`mujoco-warp`), not IsaacLab/PhysX. Its memory model is different
enough from PhysX that intuitions from IsaacLab configs (env counts, terrain
complexity) do not transfer directly. Don't reason from memory about "mjlab
should be lighter" — measure.

## First: check the hardware, not just the config

```bash
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
```

This repo's dev GPU is a 6 GB laptop card (see [CLAUDE.md](../../../CLAUDE.md)).
Reference configs copied from mjlab's own examples (e.g. `go1`/`g1` in
`.venv/lib/python*/site-packages/mjlab/tasks/velocity/config/`) are often
tuned for 24 GB+ workstation cards. A value like `ccd_iterations=500` that's
totally safe there can OOM here. Always compare against *this machine's*
budget, not the upstream example's.

## Second: reproduce with a minimal env-construction probe, not a full training run

Constructing `ManagerBasedRlEnv` + one `reset()` triggers essentially all of
the GPU memory mjlab/mujoco-warp will ever allocate for the simulation itself
(model + data buffers, contact/constraint arrays, sensor buffers, CUDA graph
capture). You do **not** need to run RSL-RL/PPO to reproduce a sim-level OOM
— that only adds rollout-buffer/optimizer memory on top, and makes crashes
slower and noisier to iterate on.

Minimal probe pattern (adapt paths/imports as needed — this exact script was
used to produce the numbers in CLAUDE.md):

```python
import os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
import torch
import mjlab.tasks          # noqa: registers built-in tasks
import src.tasks            # noqa: registers Mbf-Rough / Mbf-Flat
from mjlab.tasks.registry import load_env_cfg
from mjlab.envs import ManagerBasedRlEnv

cfg = load_env_cfg(sys.argv[1])          # e.g. "Mbf-Rough"
cfg.scene.num_envs = int(sys.argv[2])
# Optionally override knobs under test, e.g.:
# cfg.sim.mujoco.ccd_iterations = 50
# cfg.sim.nconmax = 35

free0, total = torch.cuda.mem_get_info()
try:
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    env.reset()
    torch.cuda.synchronize()
    free1, _ = torch.cuda.mem_get_info()
    print(f"OK used={(free0-free1)/1e9:.3f}GB total={total/1e9:.3f}GB")
except torch.cuda.OutOfMemoryError as e:
    print(f"OOM: {e}")
except Exception as e:
    import traceback; traceback.print_exc()
```

Key point: use `torch.cuda.mem_get_info()` (driver-level free/total), **not**
`torch.cuda.memory_allocated()`/`memory_reserved()`. Warp has its own CUDA
allocator, separate from PyTorch's caching allocator — PyTorch's memory
stats will report near-zero even when Warp has allocated gigabytes on the
same device. `mem_get_info()` is allocator-agnostic and reflects real usage.

Binary-search `num_envs` (e.g. 4096 → 2048 → 1024 → ...) and toggle one
config knob at a time to isolate which setting is responsible before
touching code.

## Reading the failure mode

Two distinct failure signatures show up for what is fundamentally the same
problem (VRAM exhaustion) — don't assume they're different bugs:

- **`RuntimeError: Failed to allocate <N> bytes on device 'cuda:0'`** — a
  clean Warp allocation failure outside CUDA graph capture. The traceback
  will point at the exact `wp.empty(...)`/`wp.zeros(...)` call and its
  shape; walk up the call stack in
  `.venv/lib/python*/site-packages/mujoco_warp/_src/` to see what the shape
  is a function of (usually `nworld`, `nconmax`/`naconmax`/`naccdmax`, or an
  iteration count like `ccd_iterations`). Compute the byte count by hand
  (`shape product × dtype size`) and confirm it matches the error — this
  turns "probably memory" into a verified root cause.
- **`RuntimeError: Warp CUDA error 700: an illegal memory access was
  encountered (in function wp_cuda_graph_begin_capture, ...)`** (or
  `wp_free_device_async`, or similar) — an allocation that was exhausted
  *during* CUDA graph capture. Warp/CUDA doesn't always surface this as a
  clean Python exception; it can come back as a raw illegal-access fault
  instead, and the fault poisons the CUDA context (subsequent calls,
  including cleanup/free calls, keep erroring). Treat this the same as an
  OOM — the fix is the same (reduce memory pressure) — but be aware that in
  a live training process this failure mode is exactly the kind of thing
  that can leave corrupted memory behind and manifest as NaN/Inf in later
  steps rather than a clean crash every time, which is worth mentioning if
  the user is chasing intermittent NaNs.

## Cross-checking suspected NaN sources empirically, not by inspection alone

If a specific term (sensor, reward, observation) is suspected of causing
NaNs, don't just remove it and hope — isolate it:

1. Reproduce the suspected term's memory/behavior in the minimal probe above
   with everything else held fixed (e.g. re-add a sensor that's currently
   disabled, or force a specific terrain/observation configuration) and
   confirm it changes the failure mode or threshold.
2. Use mjlab's built-in `NanGuard`
   (`mjlab/utils/nan_guard.py`, enabled via `--enable-nan-guard` in
   `scripts/train.py`, dumps to `/tmp/mjlab/nan_dumps`) to capture the actual
   pre-NaN simulation state on the next real run, instead of only reasoning
   from memory pressure. Check whether that directory has any dumps before
   assuming no one has captured one yet.
3. Prefer a config that fixes *all* identified contributing causes at once
   over removing the one feature you originally suspected — as seen in this
   repo, an OOM/CUDA-fault can have more than one independent contributor
   (e.g. both `ccd_iterations` and a raycast sensor pushing the same 6 GB
   budget over the edge); fixing only the suspected one can leave the
   problem partially unsolved.

## GPU memory scaling knobs specific to this stack

- `cfg.sim.nconmax` / `cfg.sim.njmax` — per-world contact/constraint buffer
  caps; total allocation is roughly `num_envs × nconmax` /
  `num_envs × njmax`.
- `cfg.sim.mujoco.ccd_iterations` — **only matters when MESH/HFIELD convex
  collision pairs exist** (i.e. rough/mesh terrain, not a flat plane). Scales
  several GJK/EPA scratch buffers linearly; each buffer's row count is
  `num_envs × nconmax` (via `naccdmax`), so cost is
  `O(num_envs × nconmax × ccd_iterations)`. This is the single most
  expensive knob for rough-terrain tasks in this repo.
- `cfg.sim.contact_sensor_maxmatch` — per-contact-sensor match buffer size;
  large values (e.g. 500, needed for rough terrain's more numerous contact
  candidates) cost more per env than the flat-terrain default (64).
- A `RayCastSensor` (height scanner) against mesh terrain adds real (if
  smaller) per-env memory for raycast-vs-mesh broadphase; cheap at low
  `num_envs`, but can be the deciding factor when the rest of the budget is
  already near the ceiling.

## "Observation group ... contains NaN" from `rsl_rl`: check for a dormant fix before writing one

If a real training run crashes with `rsl_rl`'s `check_nan()` raising on one
observation group (not rewards), don't assume you need to build NaN handling
from scratch — check whether mjlab's own manager already has an unused knob
for it first:

- `RewardManager.compute()` already sanitizes every term with
  `torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)` before summing
  — this is *why* a physics divergence crashes on observations, not rewards:
  rewards are already protected.
- `ObservationGroupCfg` has the equivalent mechanism
  (`nan_policy: Literal["disabled","warn","sanitize","error"] =
  "disabled"`, `nan_check_per_term: bool = True`) — but it's off by default.
  Setting `nan_policy="warn"` (or `"sanitize"`) on the affected group is
  usually the actual fix, not a symptom worked around.
- Separately, remember that ordinary comparisons (`x > threshold`,
  `torch.acos(x).abs() > limit`) are IEEE-754 **always False** against NaN —
  so a diverged env can silently fail every existing termination check and
  never get reset, letting the corrupted state ride through indefinitely.
  If you want the env to actually recover (not just have its observation
  papered over every step), add a termination that explicitly checks
  `torch.isnan(x) | torch.isinf(x)` on velocity/state fields (plus an
  absurd-but-finite magnitude threshold to catch it just before full
  divergence) — see `local_mdp.numerical_divergence` in
  `src/tasks/velocity/mdp/terminations.py` for a template.
- Before spending a lot of compute trying to reproduce a rare crash: check
  whether repeated runs at the *identical* config (verified via the crashing
  run's saved `params/env.yaml`, including seed) reliably reproduce it. On
  this GPU/mujoco-warp stack, physics execution has already been shown to be
  non-deterministic run-to-run even at a fixed seed (see the `ccd_iterations`
  A/B test above) — a clean run at the same config is not proof the bug is
  fixed, only that you didn't hit it this time. Prefer fixing the
  *mechanism* (read the manager source to find the actual gap) over trying
  to force a live reproduction.

## A non-simulator memory trap: batched `torch.linalg.eigh` in `dr` events

Not every multi-GB OOM at env construction is `mujoco_warp`/Warp. Some `dr`
functions (e.g. `dr.pseudo_inertia`, which diagonalizes a perturbed inertia
tensor per body per env) call batched LAPACK ops like `torch.linalg.eigh`
over a `(num_envs * num_bodies_selected, 3, 3)` batch. On at least one
GPU/driver/torch/cuSOLVER combination this had a **~0.27 MB workspace cost
per 3x3 matrix** (measured directly with a standalone
`torch.linalg.eigh(torch.randn(B, 3, 3, device="cuda:0"))` probe, bisecting
`B`) — a batch of 16384 alone needed ~4.5 GB. mjlab's own internal chunk
limit for this (`_MAX_EIGH_BATCH` in `mjlab/envs/mdp/dr/body.py`) may itself
still be one OOM-sized chunk on a small GPU.

**Tell:** the traceback shows `torch.OutOfMemoryError` (not a Warp
`RuntimeError: Failed to allocate...`), pointing into `torch.linalg.eigh`/
`svd`/similar, from inside an event function during `event_manager.apply()`
at env construction. **Fix pattern:** wrap the offending `dr.*` call in a
small loop that splits `env_ids` into sub-batches (e.g. 128 envs at a time)
and calls the real function per sub-batch — cheap since this only runs once
at `mode="startup"`. Verify the safe sub-batch size with the standalone
`torch.linalg.eigh` probe above before picking a chunk size, rather than
guessing.
