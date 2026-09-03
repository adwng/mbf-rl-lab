"""Open the MuJoCo viewer on a generated terrain.

Usage:
  # MBF-tuned rough terrain (what `Mbf-Rough` uses):
  python scripts/visualize_terrain.py

  # Show the upstream mjlab default ROUGH_TERRAINS_CFG (with pyramid stairs):
  python scripts/visualize_terrain.py --preset default

  # Show the upstream ALL_TERRAINS_CFG (every terrain type):
  python scripts/visualize_terrain.py --preset all

  # Faster load with a smaller grid:
  python scripts/visualize_terrain.py --rows 4 --cols 4

See also:
  python scripts/visualize_robot.py
  python scripts/visualize_robot.py --mode zero
  python scripts/visualize_robot.py --mode random --task Mbf-Rough
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace

import mujoco
import mujoco.viewer
import torch

from mjlab.terrains.config import ALL_TERRAINS_CFG, ROUGH_TERRAINS_CFG
from mjlab.terrains.terrain_entity import TerrainEntity, TerrainEntityCfg


def _mbf_rough_terrain_cfg():
  """Build the same terrain generator config Mbf-Rough trains against."""
  sys.path.insert(0, ".")
  from src.tasks.velocity.config.mbf.env_cfgs import mbf_rough_env_cfg

  cfg = mbf_rough_env_cfg()
  assert cfg.scene.terrain is not None
  assert cfg.scene.terrain.terrain_generator is not None
  return cfg.scene.terrain.terrain_generator


def main() -> None:
  parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
  )
  parser.add_argument(
    "--preset",
    choices=("mbf", "default", "all"),
    default="mbf",
    help="'mbf' (Mbf-Rough), 'default' (mjlab ROUGH_TERRAINS_CFG), "
    "'all' (mjlab ALL_TERRAINS_CFG).",
  )
  parser.add_argument(
    "--rows",
    type=int,
    default=None,
    help="Override terrain generator num_rows (smaller = faster load).",
  )
  parser.add_argument(
    "--cols",
    type=int,
    default=None,
    help="Override terrain generator num_cols (smaller = faster load).",
  )
  args = parser.parse_args()

  if args.preset == "mbf":
    gen = _mbf_rough_terrain_cfg()
  elif args.preset == "default":
    gen = replace(ROUGH_TERRAINS_CFG)
  else:
    gen = replace(ALL_TERRAINS_CFG)

  if args.rows is not None:
    gen.num_rows = args.rows
  if args.cols is not None:
    gen.num_cols = args.cols

  print(
    f"[INFO] preset={args.preset} "
    f"size={gen.size} rows={gen.num_rows} cols={gen.num_cols} "
    f"border={gen.border_width}"
  )
  print("[INFO] sub-terrains:")
  for name, sub in gen.sub_terrains.items():
    print(f"        {name:25s} proportion={sub.proportion}")

  device = "cuda" if torch.cuda.is_available() else "cpu"
  terrain = TerrainEntity(
    TerrainEntityCfg(terrain_type="generator", terrain_generator=gen),
    device=device,
  )
  mujoco.viewer.launch(terrain.spec.compile())


if __name__ == "__main__":
  main()
