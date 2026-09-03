"""MBF quadruped constants."""

from pathlib import Path

import mujoco

from src import SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg, DelayedActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##
# v2 model: kinematics / inertias / mesh collision from mbf_description URDF,
# with joint armature/damping/frictionloss identified in scripts/sysid/.

MBF_XML: Path = (
  SRC_PATH / "assets" / "robots" / "mbf_v2" / "mbf.xml"
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
# PD gains match real MBF hardware (Kp=20, Kd=0.5).
# frictionloss identified from data/sysid/ by scripts/sysid/identify_actuator.py
# (see scripts/sysid/README.md). ARMATURE stays at 0.003: the bench data cannot
# identify it -- replay RMSE is flat for anything in 0.0024-0.0060 -- so the
# joint_armature DR range carries that uncertainty instead.
# Note: BuiltinPositionActuatorCfg overwrites MJCF joint armature/frictionloss,
# so these must stay in sync with mbf_v2/mbf.xml.
# Physical viscous damping stays in the MJCF joint defaults (~0.135–0.176).

STIFFNESS = 20.0
DAMPING = 0.5

ARMATURE = 0.003
EFFORT_LIMIT = 10.0  # Nm

# Physics dt = 5 ms; delay_max_lag=4 → up to 20 ms command delay (sim2real).
# Copy armature/frictionloss onto the wrapper so DR helpers that read the
# outer ActuatorCfg see the nominal values.
def _delayed(base: BuiltinPositionActuatorCfg) -> DelayedActuatorCfg:
  return DelayedActuatorCfg(
    base_cfg=base,
    delay_min_lag=0,
    delay_max_lag=4,
    delay_target="position",
    armature=base.armature,
    frictionloss=base.frictionloss,
  )


MBF_SHOULDER_ACTUATOR_CFG = _delayed(
  BuiltinPositionActuatorCfg(
    target_names_expr=(".*_shoulder_joint",),
    stiffness=STIFFNESS,
    damping=DAMPING,
    effort_limit=EFFORT_LIMIT,
    armature=ARMATURE,
    frictionloss=0.228,
  )
)

MBF_HIP_ACTUATOR_CFG = _delayed(
  BuiltinPositionActuatorCfg(
    target_names_expr=(".*_hip_joint",),
    stiffness=STIFFNESS,
    damping=DAMPING,
    effort_limit=EFFORT_LIMIT,
    armature=ARMATURE,
    frictionloss=0.092,
  )
)

MBF_KNEE_ACTUATOR_CFG = _delayed(
  BuiltinPositionActuatorCfg(
    target_names_expr=(".*_knee_joint",),
    stiffness=STIFFNESS,
    damping=DAMPING,
    effort_limit=EFFORT_LIMIT,
    armature=ARMATURE,
    frictionloss=0.146,
  )
)

##
# Initial state.
##
# Crouched, all-feet-on-ground default pose. Hip joints rotate the thigh
# forward (positive about y), knees fold the calf back (negative).
# Root height ≈ 0.206 m with L_thigh=L_calf=0.12 and foot sphere r=0.022.

INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.206),
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
    MBF_SHOULDER_ACTUATOR_CFG,
    MBF_HIP_ACTUATOR_CFG,
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
  _base = _a.base_cfg if isinstance(_a, DelayedActuatorCfg) else _a
  assert isinstance(_base, BuiltinPositionActuatorCfg)
  _e = _base.effort_limit
  _s = _base.stiffness
  assert _e is not None
  for _n in _base.target_names_expr:
    # MBF_ACTION_SCALE[_n] = 0.25 * _e / _s
    MBF_ACTION_SCALE[_n] = 0.25 


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_mbf_robot_cfg())

  viewer.launch(robot.spec.compile())
