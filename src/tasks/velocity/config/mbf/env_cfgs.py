"""MBF velocity environment configurations."""

from typing import Literal

from src.assets.robots import (
  get_mbf_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import TerminationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, RayCastSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from src.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
import src.tasks.velocity.mdp as local_mdp

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


def _all_joints_cfg() -> SceneEntityCfg:
  """Fresh SceneEntityCfg per use — shared instances break after resolve()."""
  return SceneEntityCfg("robot", joint_names=(".*",))


def _add_sim2real_domain_rand(cfg: ManagerBasedRlEnvCfg) -> None:
  """Extra DR knobs for sim2sim / sim2real (ranges from LeggedGym-Ex ID)."""
  # Widen foot friction toward real-world variability.
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = _FOOT_GEOM_NAMES
  cfg.events["foot_friction"].params["ranges"] = (0.2, 1.7)

  cfg.events["base_com"].params["asset_cfg"].body_names = (_BASE_BODY,)
  cfg.events["base_com"].params["ranges"] = {
    0: (-0.03, 0.03),
    1: (-0.03, 0.03),
    2: (-0.03, 0.03),
  }

  # Chassis mass payload variation.
  cfg.events["base_mass"] = EventTermCfg(
    mode="startup",
    func=dr.body_mass,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=(_BASE_BODY,)),
      "operation": "scale",
      "ranges": (0.8, 1.25),
    },
  )

  # Identified armature band ~[0.002, 0.012] (nominal 0.003).
  cfg.events["joint_armature"] = EventTermCfg(
    mode="startup",
    func=dr.joint_armature,
    params={
      "asset_cfg": _all_joints_cfg(),
      "operation": "abs",
      "ranges": (0.002, 0.012),
    },
  )

  # Viscous joint damping ~0.05–0.3 (ID ~0.14–0.18).
  cfg.events["joint_damping"] = EventTermCfg(
    mode="startup",
    func=dr.joint_damping,
    params={
      "asset_cfg": _all_joints_cfg(),
      "operation": "abs",
      "ranges": (0.05, 0.30),
    },
  )

  # Coulomb frictionloss ~0.02–0.3 (ID ~0.08–0.22).
  cfg.events["joint_friction"] = EventTermCfg(
    mode="startup",
    func=dr.joint_friction,
    params={
      "asset_cfg": _all_joints_cfg(),
      "operation": "abs",
      "ranges": (0.02, 0.30),
    },
  )

  # PD gain scale around hardware Kp=20, Kd=0.5.
  # Do not pass actuator_names=(".*",): that resolves to 12 joint/ctrl indices,
  # while asset.actuators only has 3 DelayedActuator groups.
  cfg.events["pd_gains"] = EventTermCfg(
    mode="startup",
    func=dr.pd_gains,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "kp_range": (0.8, 1.2),
      "kd_range": (0.7, 1.3),
      "operation": "scale",
      "distribution": "uniform",
    },
  )

  # Per-episode command delay (0–4 physics steps @ 5 ms → 0–20 ms).
  cfg.events["actuator_delay"] = EventTermCfg(
    mode="reset",
    func=dr.sync_actuator_delays,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "lag_range": (0, 4),
    },
  )

  # Small joint-angle reset noise (was zero).
  cfg.events["reset_robot_joints"].params["position_range"] = (-0.1, 0.1)
  cfg.events["reset_robot_joints"].params["velocity_range"] = (-0.5, 0.5)

  # Slightly stronger / more frequent pushes for robustness.
  cfg.events["push_robot"].interval_range_s = (4.0, 6.0)
  cfg.events["push_robot"].params["velocity_range"] = {
    "x": (-0.6, 0.6),
    "y": (-0.6, 0.6),
    "z": (-0.4, 0.4),
    "roll": (-0.6, 0.6),
    "pitch": (-0.6, 0.6),
    "yaw": (-0.8, 0.8),
  }

  # IMU mounting misalignment: the real IMU is never perfectly aligned with
  # the chassis frame. imu_ang_vel / projected_gravity are read straight off
  # the chassis body, so a fixed per-env orientation offset simulates a few
  # degrees of mounting/calibration error (sampled once at startup, composed
  # with the default orientation — does not accumulate across resets).
  cfg.events["imu_mount_offset"] = EventTermCfg(
    mode="startup",
    func=dr.body_quat,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=(_BASE_BODY,)),
      "roll_range": (-0.05, 0.05),
      "pitch_range": (-0.05, 0.05),
      "yaw_range": (-0.05, 0.05),
    },
  )

  # Actuator effort-limit variability (payload / battery sag / motor wear).
  # Do not pass actuator_names=(".*",) here either — see the pd_gains note
  # above; the default SceneEntityCfg("robot") already selects all 3
  # DelayedActuator groups. Uses local_mdp.effort_limits_delayed, not
  # dr.effort_limits: mjlab's dr.effort_limits is missing the
  # DelayedActuator-unwrap that dr.pd_gains already has, so it raises
  # TypeError on MBF's actuators (all wrapped in DelayedActuatorCfg) — see
  # src/tasks/velocity/mdp/dr_extras.py.
  cfg.events["effort_limits"] = EventTermCfg(
    mode="startup",
    func=local_mdp.effort_limits_delayed,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "effort_limit_range": (0.7, 1.15),
      "operation": "scale",
    },
  )

  # Joint zero-offset / calibration error: shifts qpos0, the pose the PD
  # controller actually regulates to. Distinct from encoder_bias above, which
  # only biases the *reading* fed to the policy, not the physical setpoint.
  cfg.events["joint_default_pos"] = EventTermCfg(
    mode="startup",
    func=dr.joint_default_pos,
    params={
      "asset_cfg": _all_joints_cfg(),
      "operation": "add",
      "ranges": (-0.03, 0.03),
    },
  )

  cfg.events["link_inertia"] = EventTermCfg(
    mode="startup",
    func=local_mdp.pseudo_inertia_chunked,
    params={
      "asset_cfg": SceneEntityCfg(
        "robot", body_names=r".*_(shoulder|hip|knee|foot)_link"
      ),
      "alpha_range": (-0.1, 0.1),
    },
  )

  cfg.events["wind_gust"] = EventTermCfg(
    mode="step",
    func=envs_mdp.apply_body_impulse,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=(_BASE_BODY,)),
      "force_range": (-15.0, 15.0),
      "torque_range": (-2.0, 2.0),
      "duration_s": (0.1, 0.3),
      "cooldown_s": (3.0, 6.0),
    },
  )


def _tune_terrain_rewards(cfg: ManagerBasedRlEnvCfg, *, rough: bool) -> None:
  """Bias rewards toward clear, rhythmic stepping on uneven ground."""
  cfg.rewards["pose"].params["std_standing"] = {
    r".*_shoulder_joint": 0.05,
    r".*_hip_joint": 0.10,
    r".*_knee_joint": 0.15,
  }
  # Looser walking/running posture so legs can adapt to terrain.
  cfg.rewards["pose"].params["std_walking"] = {
    r".*_shoulder_joint": 0.20,
    r".*_hip_joint": 0.45,
    r".*_knee_joint": 0.60,
  }
  cfg.rewards["pose"].params["std_running"] = {
    r".*_shoulder_joint": 0.25,
    r".*_hip_joint": 0.55,
    r".*_knee_joint": 0.70,
  }
  cfg.rewards["pose"].weight = 0.35 if rough else 0.5

  # Trot gait: diagonal pairs in phase, opposing pair offset by 0.5.
  cfg.rewards["foot_gait"].params["offset"] = [0.0, 0.5, 0.5, 0.0]
  cfg.rewards["foot_gait"].weight = 0.75 if rough else 0.5

  cfg.rewards["body_orientation_l2"].params["asset_cfg"].body_names = (_BASE_BODY,)
  cfg.rewards["body_orientation_l2"].weight = -1.5 if rough else -1.0
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = (_BASE_BODY,)

  if not rough:
    cfg.rewards["foot_clearance"].params["asset_cfg"].site_names = _FOOT_SITE_NAMES
    cfg.rewards["foot_clearance"].params["target_height"] = 0.10
    cfg.rewards["foot_clearance"].weight = -1.0
  else: 
    cfg.rewards["foot_clearance"].weight = 0.0

  cfg.rewards["foot_slip"].params["asset_cfg"].site_names = _FOOT_SITE_NAMES
  cfg.rewards["foot_slip"].weight = -0.4 if rough else -0.25



def mbf_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create MBF rough-terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 500

  cfg.sim.njmax = 400

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
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=1,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    nonfoot_ground_cfg,
  )

  ##
  # Terrain generation tuned for MBF (gentler features for a ~0.2 m robot).
  ##
  assert cfg.scene.terrain is not None
  gen = cfg.scene.terrain.terrain_generator
  assert gen is not None

  gen.sub_terrains.pop("hf_pyramid_slope", None)
  gen.sub_terrains.pop("hf_pyramid_slope_inv", None)

  # if "pyramid_stairs" in gen.sub_terrains:
  #   sub = gen.sub_terrains["pyramid_stairs"]
  #   sub.step_height_range = (0.0, 0.08)
  #   sub.step_width = 0.25
  # if "pyramid_stairs_inv" in gen.sub_terrains:
  #   sub = gen.sub_terrains["pyramid_stairs_inv"]
  #   sub.step_height_range = (0.0, 0.06)
  #   sub.step_width = 0.25
  if "random_rough" in gen.sub_terrains:
    gen.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
    gen.sub_terrains["random_rough"].noise_step = 0.01
  if "wave_terrain" in gen.sub_terrains:
    gen.sub_terrains["wave_terrain"].amplitude_range = (0.0, 0.10)
    gen.sub_terrains["wave_terrain"].num_waves = 3

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
  # cfg.observations["actor"].terms.pop("height_scan", None)

  # Try removing height scan overall, check to see if it can prevent nan values 
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  cfg.observations["actor"].terms.pop("height_scan", None)
  cfg.observations["critic"].terms.pop("height_scan", None)

  cfg.observations["actor"].history_length = 5
  cfg.observations["critic"].history_length = 1

  cfg.observations["critic"].terms["foot_height"].params["asset_cfg"].site_names = (
    _FOOT_SITE_NAMES
  )

  ##
  # Domain randomisation + terrain rewards.
  ##
  _add_sim2real_domain_rand(cfg)
  _tune_terrain_rewards(cfg, rough=True)

  ##
  # Terminations.
  ##
  cfg.terminations["illegal_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": nonfoot_ground_cfg.name, "force_threshold": 10.0},
  )
  # Safety net for a rare physics blow-up (e.g. an unresolved deep mesh-edge
  # penetration): existing terminations use ordinary comparisons, which are
  # always False against NaN, so a diverged env can otherwise never reset
  # and its NaN state rides through to the next observation. See
  # local_mdp.numerical_divergence's docstring.
  cfg.terminations["numerical_divergence"] = TerminationTermCfg(
    func=local_mdp.numerical_divergence,
    params={},
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
  cfg.observations["critic"].history_length = 1

  # Re-apply flat-friendly reward weights (rough path set rough=True).
  _tune_terrain_rewards(cfg, rough=False)
  cfg.rewards["track_linear_velocity"].weight = 1.0
  cfg.rewards["track_linear_velocity"].params["std"] = (0.25) ** 0.5
  cfg.rewards["action_rate_l2"].weight = -0.05

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-0.5, 1.0)
    twist_cmd.ranges.lin_vel_y = (-0.3, 0.3)
    twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)

  return cfg
