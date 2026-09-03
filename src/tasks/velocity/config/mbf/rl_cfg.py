"""RL configuration for MBF velocity task."""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def _ppo(experiment_name: str, max_iterations: int) -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name=experiment_name,
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=max_iterations,
    # max_iterations=10001,
  )


def mbf_flat_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """PPO config for MBF flat-terrain velocity task."""
  # Flat terrain converges quickly; 5k iters is usually plenty.
  return _ppo(experiment_name="mbf_velocity_flat", max_iterations=2501)


def mbf_rough_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """PPO config for MBF rough-terrain velocity task."""
  return _ppo(experiment_name="mbf_velocity_rough", max_iterations=5001)
