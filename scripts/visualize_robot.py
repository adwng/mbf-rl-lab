"""Visualize the MBF robot (MJCF pose, zero actions, or random actions).

Usage:
  # Static MJCF at the default crouched pose (native MuJoCo viewer):
  python scripts/visualize_robot.py

  # Same, with all joints at zero:
  python scripts/visualize_robot.py --mode zero_joints

  # Hold default pose via zero actions in the flat training env:
  python scripts/visualize_robot.py --mode zero

  # Random actions in the rough training env:
  python scripts/visualize_robot.py --mode random --task Mbf-Rough

  # Viser (browser) when no display is available:
  python scripts/visualize_robot.py --mode random --viewer viser
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path
from typing import Literal


def _launch_mjcf(pose: Literal["default", "zero_joints"]) -> None:
  """Open the compiled MJCF with either the crouched init pose or all joints 0."""
  import mujoco
  import mujoco.viewer

  sys.path.insert(0, ".")
  from mjlab.entity.entity import Entity
  from src.assets.robots.mbf.mbf_constants import get_mbf_robot_cfg

  robot = Entity(get_mbf_robot_cfg())
  model = robot.spec.compile()
  data = mujoco.MjData(model)

  data.qpos[0:3] = [0.0, 0.0, 0.206]
  data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]

  def _qadr(name: str) -> int:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    return int(model.jnt_qposadr[jid])

  for q in ("fl", "fr", "rl", "rr"):
    data.qpos[_qadr(f"{q}_shoulder_joint")] = 0.0
    if pose == "default":
      data.qpos[_qadr(f"{q}_hip_joint")] = 0.7
      data.qpos[_qadr(f"{q}_knee_joint")] = -1.4
    else:
      data.qpos[_qadr(f"{q}_hip_joint")] = 0.0
      data.qpos[_qadr(f"{q}_knee_joint")] = 0.0

  mujoco.mj_forward(model, data)
  print(f"[INFO] MJCF pose={pose}  nq={model.nq} ngeom={model.ngeom}")
  for q in ("fl", "fr", "rl", "rr"):
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{q}_foot")
    print(f"       {q}_foot z={data.site_xpos[sid, 2]:.4f}")
  mujoco.viewer.launch(model, data)


def _load_play_module():
  play_path = Path(__file__).resolve().parent / "play.py"
  spec = importlib.util.spec_from_file_location("mbf_play", play_path)
  assert spec is not None and spec.loader is not None
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod


def _launch_env(
  mode: Literal["zero", "random"],
  task: str,
  viewer: Literal["auto", "native", "viser"],
  num_envs: int,
) -> None:
  """Delegate to the play pipeline with a dummy zero/random policy."""
  sys.path.insert(0, ".")
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401

  play = _load_play_module()
  play.run_play(
    task,
    play.PlayConfig(
      agent=mode,
      num_envs=num_envs,
      viewer=viewer,
      no_terminations=True,
    ),
  )


def main() -> None:
  parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
  )
  parser.add_argument(
    "--mode",
    choices=("mjcf", "default", "zero_joints", "zero", "random"),
    default="mjcf",
    help=(
      "'mjcf'/'default': static crouched MJCF; "
      "'zero_joints': static MJCF with all joints at 0; "
      "'zero'/'random': live env with dummy policy."
    ),
  )
  parser.add_argument(
    "--task",
    default="Mbf-Flat",
    help="Task id for zero/random modes (default: Mbf-Flat).",
  )
  parser.add_argument(
    "--viewer",
    choices=("auto", "native", "viser"),
    default="auto",
    help="Viewer backend for zero/random modes.",
  )
  parser.add_argument("--num-envs", type=int, default=1)
  args = parser.parse_args()

  if args.mode in ("mjcf", "default"):
    _launch_mjcf("default")
  elif args.mode == "zero_joints":
    _launch_mjcf("zero_joints")
  else:
    if args.viewer == "auto":
      has_display = bool(
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
      )
      viewer: Literal["auto", "native", "viser"] = (
        "native" if has_display else "viser"
      )
    else:
      viewer = args.viewer
    _launch_env(args.mode, args.task, viewer, args.num_envs)


if __name__ == "__main__":
  main()
