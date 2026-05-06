"""MBF velocity environment configurations."""

from typing import Literal

from src.assets.robots import (
  get_mbf_robot_cfg,
)
from src.assets.robots.mbf.mbf_constants import FEET_ONLY_COLLISION
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import TerminationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, RayCastSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from src.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

TerrainType = Literal["rough", "flat"]

##
# MBF anatomy.
##

_BASE_BODY = "chassis"
_FOOT_QUADRANTS = ("fl", "fr", "rl", "rr")
_FOOT_BODY_NAMES = tuple(f"{q}_foot_link" for q in _FOOT_QUADRANTS)
_FOOT_SITE_NAMES = tuple(f"{q}_foot" for q in _FOOT_QUADRANTS)
_FOOT_GEOM_NAMES = tuple(f"{q}_foot_collision" for q in _FOOT_QUADRANTS)
_KNEE_GEOM_NAMES = tuple(f"{q}_knee_link_collision" for q in _FOOT_QUADRANTS)


def mbf_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create MBF rough-terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500

  cfg.scene.entities = {"robot": get_mbf_robot_cfg()}

  # Set raycast sensor frame to MBF's chassis.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      sensor.frame.name = _BASE_BODY

  # Foot/ground contact sensor (per-foot air time, contact, force).
  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="geom", pattern=_FOOT_GEOM_NAMES, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  # Any non-foot, non-knee collision against the terrain → illegal contact
  # (used for termination). Feet are excluded because they're supposed to
  # touch the ground; knees are excluded so that brief brushing during
  # stair / rough-terrain locomotion doesn't end the episode.
  nonfoot_ground_cfg = ContactSensorCfg(
    name="nonfoot_ground_touch",
    primary=ContactMatch(
      mode="geom",
      entity="robot",
      pattern=r".*_collision\d*$",
      exclude=tuple(_FOOT_GEOM_NAMES) + tuple(_KNEE_GEOM_NAMES),
      # exclude=tuple(_FOOT_GEOM_NAMES),
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    nonfoot_ground_cfg,
  )

  ##
  # Terrain generation tuned for MBF (smaller patches, gentler features).
  ##
  assert cfg.scene.terrain is not None
  gen = cfg.scene.terrain.terrain_generator
  assert gen is not None

  gen.curriculum = True

  gen.sub_terrains.pop("hf_pyramid_slope", None)
  gen.sub_terrains.pop("hf_pyramid_slope_inv", None)

  # Pyramid stairs geometry: stairs_radius = (size - 2*border - platform) / 2,
  # and num_steps = floor(stairs_radius / step_width). The mjlab defaults
  # (platform_width=3.0, sub-terrain border_width=1.0) assume an 8 m patch and
  # would leave zero room for stairs in a 5 m patch, so we shrink them.
  # if "pyramid_stairs" in gen.sub_terrains:
  #   sub = gen.sub_terrains["pyramid_stairs"]
  #   sub.step_height_range = (0.0, 0.1)  # 0–5 cm; default 0–10 is half a leg
  #   sub.step_width = 0.20                 # tread depth
  #   sub.platform_width = 1.0
  #   sub.border_width = 0.25
  #   # stairs_radius = (5 - 0.5 - 1.0)/2 = 1.75 m  →  ~8 stair rings
  # if "pyramid_stairs_inv" in gen.sub_terrains:
  #   sub = gen.sub_terrains["pyramid_stairs_inv"]
  #   sub.step_height_range = (0.0, 0.05)
  #   sub.step_width = 0.20
  #   sub.platform_width = 1.0
  #   sub.border_width = 0.25
  # if "random_rough" in gen.sub_terrains:
  #   gen.sub_terrains["random_rough"].noise_range = (0.01, 0.05)
  #   gen.sub_terrains["random_rough"].noise_step = 0.01
  # if "wave_terrain" in gen.sub_terrains:
  #   gen.sub_terrains["wave_terrain"].amplitude_range = (0.0, 0.10)
  #   gen.sub_terrains["wave_terrain"].num_waves = 3

  # cfg.scene.terrain.max_init_terrain_level = 3

  ##
  # Actions: per-joint scale 
  ##
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)

  ##
  # Viewer.
  ##
  cfg.viewer.body_name = _BASE_BODY
  cfg.viewer.distance = 1.5
  cfg.viewer.elevation = -10.0

  ##
  # Asymmetric actor / critic observations.
  # - Actor: only sensors available on the real MBF (no base lin vel, no
  #   height scan). With a 5-frame history the policy can infer velocity
  #   from proprioception alone.
  # - Critic: keeps privileged base_lin_vel and height_scan to speed up
  #   value learning during training.
  ##
  cfg.observations["actor"].terms.pop("base_lin_vel", None)
  cfg.observations["actor"].terms.pop("height_scan", None)
  cfg.observations["actor"].history_length = 10
  cfg.observations["critic"].history_length = 5

  cfg.observations["critic"].terms["foot_height"].params["asset_cfg"].site_names = (
    _FOOT_SITE_NAMES
  )

  ##
  # Domain randomisation.
  ##
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = _FOOT_GEOM_NAMES
  cfg.events["base_com"].params["asset_cfg"].body_names = (_BASE_BODY,)

  ##
  # Rewards (variable_posture stds tuned for MBF joint ranges).
  ##
  cfg.rewards["pose"].params["std_standing"] = {
    r".*_shoulder_joint": 0.05,
    r".*_hip_joint": 0.10,
    r".*_knee_joint": 0.15,
  }
  cfg.rewards["pose"].params["std_walking"] = {
    r".*_shoulder_joint": 0.15,
    r".*_hip_joint": 0.35,
    r".*_knee_joint": 0.50,
  }
  cfg.rewards["pose"].params["std_running"] = {
    r".*_shoulder_joint": 0.15,
    r".*_hip_joint": 0.35,
    r".*_knee_joint": 0.50,
  }

  # Trot gait: diagonal pairs in phase, opposing pair offset by 0.5.
  cfg.rewards["foot_gait"].params["offset"] = [0.0, 0.5, 0.5, 0.0]

  cfg.rewards["body_orientation_l2"].params["asset_cfg"].body_names = (_BASE_BODY,)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = (_BASE_BODY,)
  cfg.rewards["foot_clearance"].params["asset_cfg"].site_names = _FOOT_SITE_NAMES
  cfg.rewards["foot_clearance"].params["target_height"] = 0.12

  cfg.rewards["foot_slip"].params["asset_cfg"].site_names = _FOOT_SITE_NAMES

  ##
  # Terminations.
  ##
  cfg.terminations["illegal_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": nonfoot_ground_cfg.name, "force_threshold": 10.0},
  )

  ##
  # Velocity command ranges (smaller robot → smaller speeds).
  ##
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.ranges.lin_vel_x = (-1.0, 1.0)
  twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)
  twist_cmd.ranges.ang_vel_z = (-1.0, 1.0)

  # Apply play-mode overrides.
  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )
    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def mbf_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create MBF flat-terrain velocity configuration."""
  cfg = mbf_rough_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Remove raycast sensor (rough-terrain only).
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  cfg.observations["actor"].terms.pop("height_scan", None)
  cfg.observations["critic"].terms.pop("height_scan", None)

  # Use a simpler collision set on flat ground (feet only) to speed sim up.
  # robot_cfg = cfg.scene.entities["robot"]
  # robot_cfg.collisions = (FEET_ONLY_COLLISION,)

  # No need for terrain curriculum on a flat plane.
  cfg.curriculum.pop("terrain_levels", None)

  # On flat ground a single-frame observation is sufficient for the actor.
  cfg.observations["actor"].history_length = 1

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-0.5, 1.0)
    twist_cmd.ranges.lin_vel_y = (-0.3, 0.3)
    twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)

  return cfg
