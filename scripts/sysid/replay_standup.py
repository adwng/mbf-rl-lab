"""Replay a recorded real standup through the mjlab `mbf` model and compare
joint tracking (sim vs real), to validate how close the simulation actuator +
body model is to hardware.

MuJoCo/mjlab port of `LeggedGym-Ex/legged_gym/scripts/mbf_standup_replay.py`,
keeping that script's structure (config block, SIGN/OFFSET calibration,
--auto-offset, actuator overrides, parameter sweep + heatmap) and swapping
Isaac Gym/PhysX for MuJoCo.

It feeds the recorded per-joint q_des(t) into the SAME control law used during
training -- tau = kp*(q_des - q) - kd*qd, clipped to the joint effort limit --
which is exactly what mjlab's `BuiltinPositionActuatorCfg` compiles to (a MuJoCo
`position` actuator with gain kp and bias [0, -kp, -kd]). The model comes from
`Entity(get_mbf_robot_cfg())`, so armature, damping, frictionloss, collisions
and solver settings are the ones training uses; only a ground plane is added.
It then overlays the sim joint angles on the recorded real ones and reports
per-joint RMSE / peak error.

Input CSV schema (as written by standup_test.py):
    t, <joint>_qdes, <joint>_q, <joint>_qd, ...      (one row per control tick)
The set of joints is auto-detected from the `*_qdes` columns.

FAIRNESS (so a mismatch means "model error", not "different setup"):
  * kp/kd/effort/dt/armature default to the training config and are only
    overridden if you pass the flags.
  * This runs deterministically: no domain randomization.
  * The base pose is NOT logged (no IMU), so the sim base is settled onto the
    ground rather than matched. Base drift is free to differ.

CALIBRATION (real joint frame -> MJCF joint frame):
  q_mjcf = SIGN * q_real + OFFSET. SIGN handles a flipped axis; OFFSET is the
  encoder zero. With --auto-offset (default) each joint's OFFSET is chosen so
  the transformed real q at the END of the log equals the MJCF standing pose
  (STANDING_POSE) -- the log ends standing, so that is the reliable anchor.
  (The Isaac Gym original anchored t=0 to a folded rest pose instead, because
  its URDF documented one; mbf_v2 documents the standing pose.)

  ALWAYS CHECK THE REPORTED BASE HEIGHT, not just per-joint RMSE. Joint angles
  can track well while the leg geometry is inverted: setting SIGN["*_knee"]=-1
  instead of using the offset makes every trace look plausible while the robot
  starts tall and SINKS (0.255 -> 0.195 m) rather than standing up. The base
  trajectory is the only thing that catches it, so it is printed and checked.

Usage:
    python scripts/sysid/replay_standup.py
    python scripts/sysid/replay_standup.py --csv data/sysid/standup_mit_log.csv --viewer
    # sweep the actuator to find the fairest match:
    python scripts/sysid/replay_standup.py \
        --sweep-armature 0.002,0.005,0.01,0.02 --sweep-damping 0.0,0.135,0.4,0.8
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[2]

# --------------------------------------------------------------------------- #
# Config -- keep in sync with velocity_env_cfg.py / mbf_constants.py           #
# --------------------------------------------------------------------------- #
SIM_DT = 0.005          # velocity_env_cfg.py sim.mujoco.timestep
GRAVITY = (0.0, 0.0, -9.81)
GROUND_FRICTION = 1.0

DEFAULT_KP = 20.0       # mbf_constants.STIFFNESS
DEFAULT_KD = 0.5        # mbf_constants.DAMPING
EFFORT_LIMIT = 10.0     # mbf_constants.EFFORT_LIMIT
# Identified in scripts/sysid/ (see README.md). Armature is NOT identifiable
# from the bench sine data -- replay RMSE there is flat over 0.0024-0.0060 --
# which makes it the most interesting thing to sweep here.
ARMATURE = 0.003        # reflected rotor inertia [kg*m^2]
JOINT_DAMPING = 0.135   # viscous [N*m*s/rad]; per-joint 0.135/0.175/0.176
JOINT_FRICTION = 0.146  # Coulomb [N*m];       per-joint 0.228/0.092/0.146

# Initial base height [m]. Only a fallback: the real height is solved by forward
# kinematics from the folded start pose so the lowest contact geom rests on the
# ground. Dropping the robot from a guessed height instead makes the front legs
# absorb an impact the real ones never saw, which biases the whole replay.
BASE_INIT_Z = 0.20
SETTLE_SECONDS = 0.5    # hold q_des[0] so the robot rests on the ground first

# MJCF standing pose (matches mbf_constants.INIT_STATE.joint_pos).
STANDING_POSE = {"shoulder": 0.0, "hip": 0.7, "knee": -1.4}


def standing_pose_for(csv_joint: str) -> float:
    for key, val in STANDING_POSE.items():
        if key in csv_joint:
            return val
    return 0.0


# Map a CSV joint name -> MJCF joint name. standup_test.py logs "fl_hip" etc.,
# the MJCF calls them "fl_hip_joint". Override here if your CSV differs.
def csv_to_mjcf_joint(name: str) -> str:
    return name if name.endswith("_joint") else f"{name}_joint"


# Real-joint -> MJCF-joint frame transform, per CSV joint name:
#   q_mjcf = SIGN * q_real + OFFSET
# No joint is sign-flipped for mbf_v2: the knees need an OFFSET (~-2.7 rad), not
# a flip. Setting a knee to -1.0 here is the failure mode described above.
SIGN: dict[str, float] = {}

# Manual fallback, used with --no-auto-offset. These are the auto-calibrated
# values for the shipped standup_mit_log.csv; the per-leg spread is real
# encoder-zero variation, not noise.
OFFSET = {
  "fl_shoulder": -0.10, "fl_hip": +0.00, "fl_knee": -2.71,
  "fr_shoulder": -0.06, "fr_hip": -0.36, "fr_knee": -2.44,
  "rl_shoulder": -0.08, "rl_hip": -0.05, "rl_knee": -2.63,
  "rr_shoulder": -0.09, "rr_hip": -0.07, "rr_knee": -2.74,
}


# --------------------------------------------------------------------------- #
def build_sim(args) -> mujoco.MjModel:
    """The env's own robot spec (actuators, collisions, solver), plus ground."""
    import sys

    sys.path.insert(0, str(REPO))
    from mjlab.entity.entity import Entity  # noqa: PLC0415

    from src.assets.robots.mbf.mbf_constants import get_mbf_robot_cfg  # noqa: PLC0415

    spec = Entity(get_mbf_robot_cfg()).spec
    spec.option.timestep = SIM_DT
    spec.option.gravity = list(GRAVITY)
    spec.worldbody.add_geom(
        name="ground",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[0.0, 0.0, 0.05],
        pos=[0.0, 0.0, 0.0],
        friction=[args.ground_friction, 0.005, 0.0001],
    )
    return spec.compile()


def load_csv(path: str):
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    cols = list(rows[0].keys())
    joints = [c[: -len("_qdes")] for c in cols if c.endswith("_qdes")]
    if not joints:
        raise ValueError("No '*_qdes' columns found in CSV.")

    def col(name):
        return np.array(
            [float(r[name]) if r[name] not in ("", "nan") else np.nan for r in rows]
        )

    t = col("t") if "t" in cols else np.arange(len(rows)) * SIM_DT
    return t, joints, {j: col(f"{j}_qdes") for j in joints}, {j: col(f"{j}_q") for j in joints}


def _parse_list(s):
    return [float(x) for x in str(s).split(",") if x.strip() != ""]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=str(REPO / "data" / "sysid" / "standup_mit_log.csv"),
                    help="recorded standup CSV")
    ap.add_argument("--kp", type=float, default=DEFAULT_KP)
    ap.add_argument("--kd", type=float, default=DEFAULT_KD)
    ap.add_argument("--effort", type=float, default=EFFORT_LIMIT)
    ap.add_argument("--joint-damping", type=float, default=JOINT_DAMPING,
                    help="viscous joint damping [N*m*s/rad] (MuJoCo dof damping)")
    ap.add_argument("--joint-friction", type=float, default=JOINT_FRICTION,
                    help="Coulomb joint friction [N*m] (MuJoCo dof frictionloss)")
    ap.add_argument("--armature", type=float, default=ARMATURE,
                    help="reflected rotor inertia [kg*m^2] (sets oscillation freq / accel)")
    ap.add_argument("--ground-friction", type=float, default=GROUND_FRICTION,
                    help="ground sliding friction (foot slip)")
    ap.add_argument("--base-init-z", type=float, default=BASE_INIT_Z,
                    help="fallback base height [m] before the FK ground solve")
    ap.add_argument("--settle-seconds", type=float, default=SETTLE_SECONDS,
                    help="hold q_des[0] this long to rest on the ground before replay")
    ap.add_argument("--no-auto-offset", dest="auto_offset", action="store_false",
                    help="use the manual OFFSET dict instead of auto-aligning the "
                         "final pose to STANDING_POSE")
    ap.set_defaults(auto_offset=True)
    ap.add_argument("--sweep-armature", type=str, default=None,
                    help="comma list of armature values to sweep, e.g. 0.002,0.005,0.01,0.02")
    ap.add_argument("--sweep-damping", type=str, default=None,
                    help="comma list of joint-damping values to sweep, e.g. 0.0,0.135,0.4,0.8")
    ap.add_argument("--viewer", action="store_true", help="show the sim viewer")
    ap.add_argument("--out", default=None, help="output plot path (png)")
    args = ap.parse_args()

    t, joints, q_des_csv, q_real_csv = load_csv(args.csv)
    n = len(t)
    dt_med = float(np.median(np.diff(t))) if n > 1 else SIM_DT
    n_sub = max(1, int(round(dt_med / SIM_DT)))
    if abs(dt_med - SIM_DT * n_sub) > 1e-4:
        print(f"[warn] CSV median dt={dt_med*1e3:.2f} ms is not a multiple of sim dt="
              f"{SIM_DT*1e3:.2f} ms; stepping {n_sub}x per row anyway.")

    model = build_sim(args)

    # Map each CSV joint -> (actuator index, qpos address, dof address).
    act_of, qadr_of, dadr_of = {}, {}, {}
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        jid = model.actuator_trnid[i, 0]
        act_of[name] = i
        qadr_of[name] = model.jnt_qposadr[jid]
        dadr_of[name] = model.jnt_dofadr[jid]

    sign, offset, idx_of = {}, {}, {}
    for j in joints:
        mjcf_name = csv_to_mjcf_joint(j)
        if mjcf_name not in act_of:
            print(f"[warn] CSV joint '{j}' -> '{mjcf_name}' not actuated in the MJCF; skipping.")
            continue
        idx_of[j] = act_of[mjcf_name]
        sign[j] = SIGN.get(j, 1.0)
        if args.auto_offset:
            # Anchor the END of the log (standing) to the MJCF standing pose.
            offset[j] = standing_pose_for(j) - sign[j] * float(q_real_csv[j][-1])
        else:
            offset[j] = OFFSET.get(j, 0.0)

    mapped = [j for j in joints if j in idx_of]
    print(f"calibration ({'AUTO' if args.auto_offset else 'MANUAL'} offset):")
    for j in mapped:
        print(f"  {j:<14} sign={sign[j]:+.0f} offset={offset[j]:+.3f} "
              f"(real q_end={q_real_csv[j][-1]:+.3f} -> mjcf "
              f"{sign[j]*q_real_csv[j][-1]+offset[j]:+.3f})")

    # Warn if the mapped command leaves the model's joint limits.
    for j in mapped:
        jid = model.actuator_trnid[idx_of[j], 0]
        lo, hi = model.jnt_range[jid]
        v = sign[j] * q_des_csv[j] + offset[j]
        if v.min() < lo - 1e-3 or v.max() > hi + 1e-3:
            print(f"[warn] {j}: mapped q_des [{v.min():+.2f},{v.max():+.2f}] outside "
                  f"MJCF range [{lo:+.2f},{hi:+.2f}] -- will be clamped")

    qdes_all = np.stack([
        np.array([sign[j] * q_des_csv[j][k] + offset[j] for j in mapped]) for k in range(n)
    ])
    real_mjcf = {j: sign[j] * q_real_csv[j] + offset[j] for j in mapped}
    ctrl_idx = [idx_of[j] for j in mapped]
    qadr = [qadr_of[csv_to_mjcf_joint(j)] for j in mapped]
    dadr = [dadr_of[csv_to_mjcf_joint(j)] for j in mapped]

    def set_actuator(armature, damping, friction):
        """MuJoCo equivalent of Isaac Gym's per-DOF property block. Mutating the
        compiled model in place keeps sweeps cheap (no recompile)."""
        for i, d in zip(ctrl_idx, dadr):
            model.dof_armature[d] = armature
            model.dof_damping[d] = damping
            model.dof_frictionloss[d] = friction
            # position actuator: tau = kp*(target - q) - kd*qd
            model.actuator_gainprm[i, 0] = args.kp
            model.actuator_biasprm[i, 1] = -args.kp
            model.actuator_biasprm[i, 2] = -args.kd
            model.actuator_forcerange[i] = (-args.effort, args.effort)

    def run_once(armature, damping, friction):
        """Reset state, set actuator props, settle, replay.

        Returns (sim_q, rmse, base_z, tilt). The base trajectory is what tells
        you whether the robot actually stood up -- joint RMSE does not.
        """
        set_actuator(armature, damping, friction)
        data = mujoco.MjData(model)
        data.qpos[2] = args.base_init_z
        data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        for k in range(len(mapped)):
            data.qpos[qadr[k]] = qdes_all[0, k]
        mujoco.mj_forward(model, data)
        contact_z = [
            data.geom_xpos[g, 2] - model.geom_size[g, 0]
            for g in range(model.ngeom)
            if model.geom_contype[g] and model.geom_type[g] != mujoco.mjtGeom.mjGEOM_PLANE
        ]
        data.qpos[2] -= min(contact_z)  # rest the lowest contact geom on z = 0

        data.ctrl[ctrl_idx] = qdes_all[0]
        for _ in range(int(args.settle_seconds / SIM_DT)):
            mujoco.mj_step(model, data)

        sim_q = np.zeros((n, len(mapped)))
        base_z = np.zeros(n)
        tilt = np.zeros(n)
        rot = np.zeros(9)
        for k in range(n):
            sim_q[k] = [data.qpos[a] for a in qadr]
            base_z[k] = data.qpos[2]
            mujoco.mju_quat2Mat(rot, data.qpos[3:7])
            tilt[k] = np.degrees(np.arccos(np.clip(rot[8], -1.0, 1.0)))
            data.ctrl[ctrl_idx] = qdes_all[k]
            for _ in range(n_sub):
                mujoco.mj_step(model, data)
            if not np.all(np.isfinite(data.qpos)):
                raise RuntimeError(f"sim diverged at row {k}")
        rmse = {j: float(np.sqrt(np.mean((sim_q[:, i] - real_mjcf[j]) ** 2)))
                for i, j in enumerate(mapped)}
        return sim_q, rmse, base_z, tilt

    def report_base(base_z, tilt):
        print(f"\nbase height: {base_z[0]:.3f} -> {base_z[-1]:.3f} m "
              f"(MJCF standing height 0.206), max tilt {tilt.max():.1f} deg")
        if base_z[-1] < base_z[0] + 0.02:
            print("  *** THE ROBOT DID NOT STAND UP. ***")
            print("  Either the joint calibration is wrong or this log is not a standup.")
            print("  Joint angles can track well while the leg geometry is inverted, so")
            print("  check this line, not just the per-joint RMSE below.")
        else:
            print("  stood up.")

    if args.viewer:
        import mujoco.viewer as mj_viewer  # noqa: PLC0415  (plain import shadows `mujoco`)

        set_actuator(args.armature, args.joint_damping, args.joint_friction)
        data = mujoco.MjData(model)
        data.qpos[2] = args.base_init_z
        for k in range(len(mapped)):
            data.qpos[qadr[k]] = qdes_all[0, k]
        with mj_viewer.launch_passive(model, data) as v:
            while v.is_running():
                for row in qdes_all:
                    data.ctrl[ctrl_idx] = row
                    for _ in range(n_sub):
                        mujoco.mj_step(model, data)
                    v.sync()
        return

    arms = _parse_list(args.sweep_armature) if args.sweep_armature else None
    damps = _parse_list(args.sweep_damping) if args.sweep_damping else None

    if arms or damps:
        arms = arms or [args.armature]
        damps = damps or [args.joint_damping]
        grid = np.zeros((len(arms), len(damps)))
        print("\n=== sweep: mean joint RMSE [rad] ===")
        print("arm\\damp " + "".join(f"{d:>9.3f}" for d in damps))
        best = None
        for ia, a in enumerate(arms):
            row = []
            for idd, dmp in enumerate(damps):
                _, rmse, _, _ = run_once(a, dmp, args.joint_friction)
                m = float(np.mean(list(rmse.values())))
                grid[ia, idd] = m
                row.append(m)
                if best is None or m < best[0]:
                    best = (m, a, dmp)
            print(f"{a:>8.3f} " + "".join(f"{v:>9.4f}" for v in row))
        spread = (grid.max() - grid.min()) / grid.mean()
        print(f"\nBEST: armature={best[1]:.4f}  damping={best[2]:.4f}  "
              f"friction={args.joint_friction:.3f}  mean RMSE={best[0]:.4f} rad")
        print(f"grid spread: {100*spread:.1f}% of the mean "
              f"({grid.min():.4f}-{grid.max():.4f} rad)")
        if spread < 0.05:
            print("  *** THE SWEEP IS UNINFORMATIVE -- do not read anything into BEST. ***")
            print("  A standup is quasi-static and contact-dominated, so it barely")
            print("  responds to armature or damping at all. Identify those from the")
            print("  bench sine logs with identify_actuator.py instead.")
        else:
            print("NOTE: the legs are loaded here, so this grid is shaped by the contact")
            print("model as much as by the actuator. Cross-check any result against")
            print("identify_actuator.py on the sine logs.")
        _plot_heatmap(arms, damps, grid, args)
        sim_q, rmse, base_z, tilt = run_once(best[1], best[2], args.joint_friction)
        report_base(base_z, tilt)
        _plot_overlay(t, mapped, real_mjcf, qdes_all, sim_q, rmse, args,
                      title=f"BEST arm={best[1]:.4f} damp={best[2]:.3f} "
                            f"fric={args.joint_friction:.2f}")
    else:
        sim_q, rmse, base_z, tilt = run_once(
            args.armature, args.joint_damping, args.joint_friction)
        report_base(base_z, tilt)
        print("\n=== sim vs real joint tracking ===")
        print(f"{'joint':<16}{'RMSE[rad]':>12}{'maxerr[rad]':>14}{'realRMSE':>12}")
        for i, j in enumerate(mapped):
            err = sim_q[:, i] - real_mjcf[j]
            r_real = np.sqrt(np.mean((qdes_all[:, i] - real_mjcf[j]) ** 2))
            print(f"{j:<16}{rmse[j]:>12.4f}{np.max(np.abs(err)):>14.4f}{r_real:>12.4f}")
        print(f"\nmean RMSE: {np.mean(list(rmse.values())):.4f} rad")
        _plot_overlay(t, mapped, real_mjcf, qdes_all, sim_q, rmse, args,
                      title=f"kp={args.kp} kd={args.kd} arm={args.armature:.4f} "
                            f"damp={args.joint_damping:.3f} fric={args.joint_friction:.2f}")


def _plot_overlay(t, mapped, real_mjcf, qdes_all, sim_q, rmse, args, title):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"[warn] plotting skipped: {e}")
        return
    ncol = 3
    nrow = int(np.ceil(len(mapped) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 2.7 * nrow), squeeze=False)
    for ax_i, j in enumerate(mapped):
        ax = axes[ax_i // ncol][ax_i % ncol]
        ax.plot(t, qdes_all[:, ax_i], "k--", lw=1.0, alpha=0.55, label="commanded")
        ax.plot(t, real_mjcf[j], lw=1.6, label="real")
        ax.plot(t, sim_q[:, ax_i], lw=1.6, alpha=0.85, label="sim")
        ax.set_title(f"{j}  (RMSE {rmse[j]:.3f})", fontsize=9)
        ax.grid(True, alpha=0.3)
        if ax_i == 0:
            ax.legend(fontsize=8)
    for k in range(len(mapped), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.suptitle(f"mbf standup replay: sim vs real ({title})")
    fig.tight_layout()
    out = args.out or os.path.splitext(args.csv)[0] + "_mjlab_replay.png"
    fig.savefig(out, dpi=120)
    print(f"Saved plot -> {out}")


def _plot_heatmap(arms, damps, grid, args):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"[warn] heatmap skipped: {e}")
        return
    fig, ax = plt.subplots(figsize=(1.2 * len(damps) + 3, 1.0 * len(arms) + 2))
    im = ax.imshow(grid, origin="lower", aspect="auto", cmap="viridis_r")
    ax.set_xticks(range(len(damps))); ax.set_xticklabels([f"{d:.3f}" for d in damps])
    ax.set_yticks(range(len(arms))); ax.set_yticklabels([f"{a:.4f}" for a in arms])
    ax.set_xlabel("joint damping [Nms/rad]"); ax.set_ylabel("armature [kg m^2]")
    spread = (grid.max() - grid.min()) / grid.mean()
    ax.set_title(f"mean joint RMSE [rad] (lower = better)\n"
                 f"total spread {100*spread:.1f}% of mean"
                 + ("  --  UNINFORMATIVE" if spread < 0.05 else ""))
    for ia in range(len(arms)):
        for idd in range(len(damps)):
            ax.text(idd, ia, f"{grid[ia, idd]:.3f}", ha="center", va="center",
                    color="w", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    out = (args.out and os.path.splitext(args.out)[0] + "_sweep.png") or \
        os.path.splitext(args.csv)[0] + "_mjlab_sweep.png"
    fig.savefig(out, dpi=120)
    print(f"Saved sweep heatmap -> {out}")


if __name__ == "__main__":
    main()
