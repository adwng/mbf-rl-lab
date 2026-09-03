from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def numerical_divergence(
  env: ManagerBasedRlEnv,
  max_lin_vel: float = 50.0,
  max_ang_vel: float = 100.0,
  max_joint_vel: float = 200.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Terminate envs whose root/joint state has diverged (NaN/Inf or absurd
  but still-finite magnitude).

  A rare physics blow-up (e.g. an unresolved deep mesh-edge penetration
  producing a huge contact impulse) can push qpos/qvel to NaN. The existing
  terminations (`bad_orientation`, etc.) use ordinary comparisons like
  `torch.acos(x).abs() > limit_angle`, which are IEEE-754 *always False*
  against a NaN input — so a diverged env silently fails every termination
  check, never gets reset, and its NaN state rides through to the next
  observation, surfacing downstream as
  "observation group ... contains NaN values" in the RL library rather than
  at its actual source. This explicitly checks `isnan`/`isinf` (which,
  unlike ordinary comparisons, correctly detect NaN) plus an absurd-velocity
  threshold (to catch the state just before it fully diverges to NaN), and
  forces a reset before that happens.
  """
  asset: Entity = env.scene[asset_cfg.name]
  lin_vel = asset.data.root_link_vel_w[:, 0:3]
  ang_vel = asset.data.root_link_vel_w[:, 3:6]
  joint_vel = asset.data.joint_vel

  bad = torch.isnan(lin_vel).any(dim=-1) | torch.isinf(lin_vel).any(dim=-1)
  bad |= torch.isnan(ang_vel).any(dim=-1) | torch.isinf(ang_vel).any(dim=-1)
  bad |= torch.isnan(joint_vel).any(dim=-1) | torch.isinf(joint_vel).any(dim=-1)
  bad |= torch.norm(lin_vel, dim=-1) > max_lin_vel
  bad |= torch.norm(ang_vel, dim=-1) > max_ang_vel
  bad |= torch.norm(joint_vel, dim=-1) > max_joint_vel
  return bad


def illegal_contact(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    # force_history: [B, N, H, 3]
    force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
    return (force_mag > force_threshold).any(dim=-1).any(dim=-1)  # [B]
  assert data.found is not None
  return torch.any(data.found, dim=-1)