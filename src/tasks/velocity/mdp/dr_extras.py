"""Domain-randomization glue for actuator wrappers that mjlab's own `dr`
module doesn't recognize.

MBF wraps every actuator in `DelayedActuatorCfg` (see
`src/assets/robots/mbf/mbf_constants.py::_delayed`) to simulate sim2real
command latency. mjlab's own `dr.pd_gains` already handles this by unwrapping
`DelayedActuator.base_actuator` before dispatching on actuator type — but
`dr.effort_limits` is missing that same one-line unwrap (compare
`mjlab/envs/mdp/dr/actuator.py::pd_gains` vs `::effort_limits` in the
installed package), so it raises `TypeError` for any MBF actuator. This
module re-implements `effort_limits` with that unwrap added; everything else
(ctrl-id resolution, sampling, forcerange write) is identical to mjlab's
version.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch

from mjlab.actuator import BuiltinPositionActuator, XmlPositionActuator
from mjlab.actuator.delayed_actuator import DelayedActuator
from mjlab.entity import Entity
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.dr._core import _DEFAULT_ASSET_CFG
from mjlab.envs.mdp.dr._types import resolve_distribution
from mjlab.managers.event_manager import requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

# torch.linalg.eigh's batched CUDA path (used by dr.pseudo_inertia to
# diagonalize the perturbed inertia tensor) has a per-matrix workspace cost
# far larger than the matrix itself on this stack (measured ~0.27 MB per 3x3
# matrix on an RTX 4050 6GB — a batch of 16384 alone needs ~4.5 GB). mjlab's
# own internal chunk size (_MAX_EIGH_BATCH=16384 in
# mjlab/envs/mdp/dr/body.py) is still one full OOM-sized chunk on a 6 GB
# card. This constant caps how many envs we hand to a *single*
# dr.pseudo_inertia call; num_envs / _PSEUDO_INERTIA_ENV_CHUNK sequential
# calls are made instead, each well under mjlab's own chunk size, so the
# transient eigh workspace never grows with num_envs. Only matters at
# mode="startup" (runs once at env construction), so the extra Python-level
# loop has no training-time cost.
_PSEUDO_INERTIA_ENV_CHUNK = 128


@requires_model_fields("actuator_forcerange")
def effort_limits_delayed(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  effort_limit_range: tuple[float, float],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  distribution: Literal["uniform", "log_uniform"] = "uniform",
  operation: Literal["scale", "abs"] = "scale",
) -> None:
  """Like `mjlab.envs.mdp.dr.effort_limits`, but unwraps `DelayedActuator`."""
  asset: Entity = env.scene[asset_cfg.name]

  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  else:
    env_ids = env_ids.to(env.device, dtype=torch.int)

  if isinstance(asset_cfg.actuator_ids, list):
    actuators = [asset.actuators[i] for i in asset_cfg.actuator_ids]
  else:
    actuators = asset.actuators[asset_cfg.actuator_ids]
  if not isinstance(actuators, list):
    actuators = [actuators]

  actuators = [
    a.base_actuator if isinstance(a, DelayedActuator) else a for a in actuators
  ]

  for actuator in actuators:
    if not isinstance(actuator, (BuiltinPositionActuator, XmlPositionActuator)):
      raise TypeError(
        "effort_limits_delayed only supports BuiltinPositionActuator or "
        f"XmlPositionActuator (optionally wrapped in DelayedActuator), got "
        f"{type(actuator).__name__}"
      )

    ctrl_ids = actuator.global_ctrl_ids
    num_actuators = len(ctrl_ids)

    dist = resolve_distribution(distribution)
    effort_samples = dist.sample(
      torch.tensor(effort_limit_range[0], device=env.device),
      torch.tensor(effort_limit_range[1], device=env.device),
      (len(env_ids), num_actuators),
      env.device,
    )

    if operation == "scale":
      default_forcerange = env.sim.get_default_field("actuator_forcerange")
      env.sim.model.actuator_forcerange[env_ids[:, None], ctrl_ids, 0] = (
        default_forcerange[ctrl_ids, 0] * effort_samples
      )
      env.sim.model.actuator_forcerange[env_ids[:, None], ctrl_ids, 1] = (
        default_forcerange[ctrl_ids, 1] * effort_samples
      )
    elif operation == "abs":
      env.sim.model.actuator_forcerange[env_ids[:, None], ctrl_ids, 0] = (
        -effort_samples
      )
      env.sim.model.actuator_forcerange[env_ids[:, None], ctrl_ids, 1] = (
        effort_samples
      )


def pseudo_inertia_chunked(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  **kwargs,
) -> None:
  """Like `mjlab.envs.mdp.dr.pseudo_inertia`, but calls it in small env
  sub-batches to avoid the `torch.linalg.eigh` workspace blowup — see
  `_PSEUDO_INERTIA_ENV_CHUNK` above for why.
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  for chunk in env_ids.split(_PSEUDO_INERTIA_ENV_CHUNK):
    dr.pseudo_inertia(env, chunk, **kwargs)
