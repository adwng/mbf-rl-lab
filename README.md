# mbf_rl_lab

Reinforcement-learning training and deployment scaffolding for the **MBF**
quadruped, structured the same way as
[`unitree_rl_mjlab`](https://github.com/unitreerobotics/unitree_rl_mjlab) and
built on top of [`mjlab`](https://github.com/mujocolab/mjlab) `1.2.0`.

The goal is to reuse the velocity-tracking task pipeline from `unitree_rl_mjlab`
(rewards, observations, terminations, curriculum, runner), with MBF as the only
robot.

## Structure

```
mbf_rl_lab/
├── setup.py
├── scripts/
│   ├── train.py
│   ├── play.py
│   └── list_envs.py
└── src/
    ├── __init__.py
    ├── assets/
    │   └── robots/
    │       ├── __init__.py             # exports MBF
    │       └── mbf/
    │           ├── mbf_constants.py    # actuators, init state, collisions
    │           └── xmls/
    │               ├── mbf.xml         # MJCF
    │               └── assets/         # STL meshes
    └── tasks/
        ├── __init__.py
        └── velocity/
            ├── velocity_env_cfg.py     # base velocity env factory
            ├── mdp/                    # rewards / observations / etc
            ├── rl/                     # VelocityOnPolicyRunner (auto-ONNX)
            └── config/
                └── mbf/
                    ├── env_cfgs.py     # mbf_flat / mbf_rough
                    ├── rl_cfg.py       # PPO hyperparams
                    └── __init__.py     # task registration
```

## Registered tasks

- `Mbf-Flat`  – plane terrain, 5k iterations, history length 1 (single-frame
  proprioception is enough on flat ground).
- `Mbf-Rough` – generated rough terrain with curriculum, 10k iterations,
  actor history length 5 (blind locomotion: no `height_scan`, infers terrain
  from a 5-frame proprioception window).

Both use an asymmetric actor / critic:

- **Actor** observes only sensors available on the real robot:
  `imu_ang_vel`, `projected_gravity`, command, `phase`, `joint_pos`,
  `joint_vel`, last action. No `base_lin_vel`, no `height_scan`.
- **Critic** additionally gets privileged signals during training:
  `base_lin_vel`, `height_scan` (rough only), foot heights / air time /
  contact / contact forces.

## Setup

This project depends on `mjlab==1.2.0` from PyPI (NOT the upstream `main`
branch — there's no need to clone the mjlab repo).

```bash
cd mbf_rl_lab
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

Or with `uv`:

```bash
cd mbf_rl_lab
uv venv
uv pip install -e .
```

The MJCF assets ship with the package (under `src/assets/robots/mbf/xmls/`),
so no further data download is required.

## Setup on Google Colab

Colab gives you a free T4 (16 GB) GPU which is enough for `Mbf-Flat` and is
usable (slow, but works) for `Mbf-Rough`. The trick is that `mjlab` 1.2.0 is
built against `mujoco-warp` 3.5.0 + a recent PyTorch CUDA, both of which need
a fresh runtime.

### 1. Pick a GPU runtime

`Runtime` → `Change runtime type` → **T4 GPU** (or A100 / L4 if you have Pro).

Verify in a cell:

```python
!nvidia-smi
```

### 2. Clone and install

You can either push this project to a Git repo first, or upload the folder to
Drive. The git approach is cleanest:

```python
# In a Colab cell:
!git clone https://github.com/<your-user>/mbf_rl_lab.git
%cd mbf_rl_lab

# Use uv for fast, reproducible installs:
!pip install -q uv
!uv pip install --system -e .
```

If you uploaded the folder to Drive instead:

```python
from google.colab import drive
drive.mount("/content/drive")
%cd /content/drive/MyDrive/mbf_rl_lab   # adjust path
!pip install -q uv
!uv pip install --system -e .
```

`--system` tells `uv` to install into the existing Colab Python instead of
creating a venv (Colab's runtime already _is_ the env).

### 3. Headless rendering / EGL

`mjlab` uses MuJoCo's EGL backend for headless rendering. Colab containers
have the right libraries pre-installed, you just need to set the env var:

```python
import os
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
```

(`scripts/train.py` already sets `MUJOCO_GL=egl` internally before launching,
but you'll want it set in your notebook environment too if you import the
modules directly.)

### 4. (Optional) Persist logs to Drive

Training logs go to `./logs/rsl_rl/...` by default. Mount Drive and symlink so
checkpoints survive runtime restarts:

```python
from google.colab import drive
drive.mount("/content/drive")
!mkdir -p /content/drive/MyDrive/mbf_logs
!ln -sfn /content/drive/MyDrive/mbf_logs /content/mbf_rl_lab/logs
```

### 5. (Optional) Weights & Biases

`mjlab`'s runner can stream metrics to wandb:

```python
import wandb
wandb.login()  # paste API key
```

Then add `--agent.logger wandb --agent.wandb-project mbf_rl_lab` to the train
command.

### 6. Train from a notebook cell

Background the long-running command with `&` and tail the log so the cell
stays responsive:

```python
!nohup python scripts/train.py Mbf-Flat \
    --agent.max-iterations 5000 \
    > train_flat.log 2>&1 &
!sleep 5 && tail -n 50 train_flat.log
```

Then in another cell, watch progress live:

```python
!tail -f train_flat.log
```

(Interrupt the cell to stop tailing without killing training.)

For `Mbf-Rough` on a T4, expect ~1.0–1.5x slowdown vs. the flat task because
of the extra ray-cast + contact work. You may want to drop the env count if
you OOM:

```python
!python scripts/train.py Mbf-Rough --env.scene.num-envs 2048
```

### 7. Tensorboard inside the notebook

```python
%load_ext tensorboard
%tensorboard --logdir logs/rsl_rl
```

### Caveats

- Colab free runtimes are time-limited (~12 h, less if idle). Save to Drive
  (step 4) so a disconnect doesn't lose checkpoints.
- The MuJoCo viewer (`scripts/play.py --viewer native`) needs a display, so
  use `--viewer viser` and forward the printed URL through Colab's port
  preview if you want interactive playback.
- `scripts/train.py --video` works on Colab (renders MP4 via EGL); the videos
  end up in `logs/rsl_rl/<run>/videos/train/`.

## Train

```bash
# Flat terrain
python scripts/train.py Mbf-Flat

# Rough terrain
python scripts/train.py Mbf-Rough
```

Useful flags (forwarded to `tyro`):

```bash
# Use 1 GPU (default), or "all", or specific ids:
python scripts/train.py Mbf-Flat --gpu-ids 0
python scripts/train.py Mbf-Flat --gpu-ids all

# Periodic videos in the run directory:
python scripts/train.py Mbf-Flat --video --video-interval 2000

# Enable NaN guard (writes diagnostic dump on any NaN/Inf):
python scripts/train.py Mbf-Flat --enable-nan-guard

# Resume:
python scripts/train.py Mbf-Flat --agent.resume \
    --agent.load-run "<run-dir-name>" --agent.load-checkpoint "model_5000.pt"
```

Logs go to `logs/rsl_rl/<experiment_name>/<timestamp>/`. Each `save` writes
`model_<iter>.pt` and exports `policy.onnx` with embedded metadata
(observation order, action order, default joint pos, action scale, decimation,
sim dt) for hardware deployment.

## Play / visualize

```bash
# Trained policy from a checkpoint file:
python scripts/play.py Mbf-Flat --checkpoint-file logs/rsl_rl/mbf_velocity_flat/<run>/model_5000.pt

# Random / zero policy (no checkpoint needed):
python scripts/play.py Mbf-Flat --agent zero
python scripts/play.py Mbf-Flat --agent random
```

## List registered tasks

```bash
python scripts/list_envs.py --keyword Mbf
```

## Foot height tuning

If you want higher foot swings, edit
`src/tasks/velocity/config/mbf/env_cfgs.py`:

```python
cfg.rewards["foot_clearance"].params["target_height"] = 0.10  # was 0.08
```

You can also raise the weight (more negative) or add `foot_swing_height`
penalising landings far from the target. See
`src/tasks/velocity/mdp/rewards.py` for available reward terms.
