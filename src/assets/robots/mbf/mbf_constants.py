"""MBF quadruped constants."""

from pathlib import Path

import mujoco

from src import SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

MBF_XML: Path = (
  SRC_PATH / "assets" / "robots" / "mbf" / "xmls" / "mbf.xml"
)
assert MBF_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, MBF_XML.parent / "assets", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(MBF_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


##
# Actuator config.
##
# Matches the PD gains used on the real MBF hardware (Kp=20, Kd=0.5).
# Armatures (rotor inertia reflected through each joint's gearbox) are kept
# from the MJCF defaults: 0.02 for shoulder/hip, 0.01 for knee.

STIFFNESS = 20.0
DAMPING = 0.5

ARMATURE_HIP_SHOULDER = 0.02
ARMATURE_KNEE = 0.01

EFFORT_LIMIT = 10.0  # Nm, matches the per-class motor forcerange in the MJCF.

MBF_HIP_SHOULDER_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_joint", ".*_shoulder_joint"),
  stiffness=STIFFNESS,
  damping=DAMPING,
  effort_limit=EFFORT_LIMIT,
  armature=ARMATURE_HIP_SHOULDER,
)

MBF_KNEE_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_knee_joint",),
  stiffness=STIFFNESS,
  damping=DAMPING,
  effort_limit=EFFORT_LIMIT,
  armature=ARMATURE_KNEE,
)

##
# Initial state.
##
# Crouched, all-feet-on-ground default pose. Hip joints rotate the thigh
# forward (positive about y), knees fold the calf back (negative).

INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.21),
  joint_pos={
    ".*_shoulder_joint": 0.0,
    ".*_hip_joint": 0.7,
    ".*_knee_joint": -1.4,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

_FOOT_REGEX = r"^(fl|fr|rl|rr)_foot_collision$"

# Enable all collisions; foot geoms get higher friction priority and softer
# contact (solref) for stability with sphere feet.
FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  solref=(0.01, 1),
  condim={_FOOT_REGEX: 6, ".*_collision": 1},
  priority={_FOOT_REGEX: 1},
  friction={_FOOT_REGEX: (1.0, 5e-3, 5e-4)},
)

# Disable all body collisions except the feet (used for flat-only training).
FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(_FOOT_REGEX,),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
  solimp=(0.9, 0.95, 0.023),
)

##
# Final config.
##

MBF_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    MBF_HIP_SHOULDER_ACTUATOR_CFG,
    MBF_KNEE_ACTUATOR_CFG,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_mbf_robot_cfg() -> EntityCfg:
  """Return a fresh MBF robot configuration instance."""
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=MBF_ARTICULATION,
  )


# Per-joint action scale: 0.25 * effort_limit / stiffness.
MBF_ACTION_SCALE: dict[str, float] = {}
for _a in MBF_ARTICULATION.actuators:
  assert isinstance(_a, BuiltinPositionActuatorCfg)
  _e = _a.effort_limit
  _s = _a.stiffness
  assert _e is not None
  for _n in _a.target_names_expr:
    MBF_ACTION_SCALE[_n] = 0.25 * _e / _s


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_mbf_robot_cfg())

  viewer.launch(robot.spec.compile())
