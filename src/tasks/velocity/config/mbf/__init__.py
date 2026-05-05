from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  mbf_flat_env_cfg,
  mbf_rough_env_cfg,
)
from .rl_cfg import (
  mbf_flat_ppo_runner_cfg,
  mbf_rough_ppo_runner_cfg,
)

register_mjlab_task(
  task_id="Mbf-Rough",
  env_cfg=mbf_rough_env_cfg(),
  play_env_cfg=mbf_rough_env_cfg(play=True),
  rl_cfg=mbf_rough_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mbf-Flat",
  env_cfg=mbf_flat_env_cfg(),
  play_env_cfg=mbf_flat_env_cfg(play=True),
  rl_cfg=mbf_flat_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
