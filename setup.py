"""Installation script for the 'mbf_rl_lab' python package."""

from setuptools import setup

INSTALL_REQUIRES = [
  "mjlab==1.2.0",
  "mujoco-warp==3.5.0",
  # mujoco-warp 3.5.0 declares only `mujoco>=3.4.0`, but mujoco 3.8.0 removed
  # the `mjENBL_MULTICCD` enum that mujoco-warp 3.5.0 still references. Cap to
  # the last compatible release.
  "mujoco>=3.4.0,<3.8.0",
  "warp-lang>=1.12.0,<1.13.0"
]

setup(
  name="mbf_rl_lab",
  packages=["src"],
  version="0.0.1",
  install_requires=INSTALL_REQUIRES,
)
