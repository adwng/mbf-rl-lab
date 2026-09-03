---
name: mjlab-train-play
description: Reference for training, playing back, and inspecting MBF policies in this repo (scripts/train.py, play.py, list_envs.py). Use whenever the user wants to launch, resume, or evaluate a training run, or asks what flags/tasks are available.
---

# Training / playing MBF policies

Two registered tasks (`src/tasks/velocity/config/mbf/__init__.py`):
`Mbf-Flat`, `Mbf-Rough`. List all registered tasks:

```bash
python scripts/list_envs.py --keyword Mbf
```

## Train

```bash
python scripts/train.py Mbf-Flat
python scripts/train.py Mbf-Rough
```

Common flags (forwarded to `tyro`, so any dataclass field on
`TrainConfig`/`ManagerBasedRlEnvCfg`/`RslRlBaseRunnerCfg` is overridable from
the CLI, e.g. `--env.scene.num-envs`, `--env.sim.mujoco.ccd-iterations`):

- `--gpu-ids 0` / `--gpu-ids all` / specific ids
- `--env.scene.num-envs <N>` — override env count (default comes from the
  task's registered cfg; see `env_cfgs.py` — currently `num_envs=1` in the
  factory default, always overridden by whatever was actually passed at
  launch, so **check `logs/rsl_rl/<run>/params/env.yaml` for what a past run
  actually used**, don't assume from the source)
- `--video` / `--video-interval N` — periodic MP4s in the run dir
- `--enable-nan-guard` — dumps a diagnostic snapshot to
  `/tmp/mjlab/nan_dumps` on first NaN/Inf detection (see
  `mjlab-sim-diagnostics` skill)
- `--agent.resume --agent.load-run "<run-dir-name>" --agent.load-checkpoint "model_5000.pt"`

If you're chasing an OOM/NaN issue, prefer the minimal env-construction
probe in the `mjlab-sim-diagnostics` skill over repeated full `train.py`
runs — it's much faster to iterate on and isolates the simulation layer from
RSL-RL/PPO overhead.

## Play / visualize

```bash
python scripts/play.py Mbf-Flat --checkpoint-file logs/rsl_rl/mbf_velocity_flat/<run>/model_5000.pt
python scripts/play.py Mbf-Flat --agent zero      # no checkpoint needed
python scripts/play.py Mbf-Flat --agent random
```

## Where things land

- `logs/rsl_rl/<experiment_name>/<timestamp>/` — `model_<iter>.pt`,
  `policy.onnx` (embeds obs/action order, default pose, action scale,
  decimation, sim dt for hardware deployment), tensorboard events,
  `params/{env,agent}.yaml` (the **resolved** config actually used — this is
  the ground truth for "what did that run use", not the source files, since
  source defaults get overridden by CLI flags).
- `git/mjlab_mbf.diff` inside a run dir — the exact code diff at launch time,
  useful for correlating a specific run's behavior with uncommitted
  experiments.
- `wandb/offline-run-*/` — offline W&B logging (sync later with
  `wandb sync` if online reporting is wanted); `files/model_*.pt` are
  symlinks back into `logs/rsl_rl/...`, `files/mjlab_mbf.diff` mirrors the
  git diff.

## Foot swing height tuning

`src/tasks/velocity/config/mbf/env_cfgs.py`:
`cfg.rewards["foot_clearance"].params["target_height"]`. See
`src/tasks/velocity/mdp/rewards.py` for all available reward terms.

## Full obs/action/reward/DR spec

[docs/mbf_velocity_spec.md](../../../docs/mbf_velocity_spec.md) is the
maintained source of truth for exact observation ordering, dims, reward
weights, and domain-randomization ranges — check it before hand-deriving any
of that from the config files.
