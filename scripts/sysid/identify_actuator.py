"""Identify MBF joint actuator parameters against the *actual* MuJoCo plant.

This supersedes ``LeggedGym-Ex/legged_gym/sim2real/identify_actuator.py``, which
fit a hand-rolled 1-DOF ODE and then subtracted a hardcoded structural inertia
(``leg_inertia.iy_about_hip``) that was (a) derived for the hip joint only yet
applied to all three joints, and (b) based on URDF link masses that ``mbf_v2``
has since changed. See ``scripts/sysid/README.md``.

Physics
-------
Each bench log drives a single joint while every other joint is held. That is
exactly a 1-DOF system:

    (I_struct + armature) * qddot = tau_cmd - b*qd - c*sign(qd) - tau_grav(q)

where

  I_struct   composite inertia of the moving subtree about the joint axis,
             read from the compiled mjlab model's mass matrix (NOT hardcoded).
             Depends only on *downstream* joint angles, so it is exact for the
             knee and pose-dependent for the hip/shoulder (see --pose).
  armature   rotor inertia reflected to the joint. Physically shared by all
             three joints (same GIM6010-8); --mode pooled enforces that.
             NOTE: this data cannot identify armature. Replay RMSE is flat at
             7.2 mrad for armature anywhere in 0.0024-0.0060 -- the fit pins the
             *total* effective inertia, and the split between rotor inertia and
             the unlogged bench pose's structural inertia is unconstrained.
             Pooling was expected to break that and does not. Treat the fitted
             armature as a band, not a value; damping/frictionloss are solid.
  b, c       MuJoCo joint ``damping`` / ``frictionloss``.
  tau_grav   A*sin(q) + B*cos(q). This is the exact gravity torque of a rigid
             subtree on a revolute joint in uniform gravity, for *any* mounting
             orientation and *any* encoder zero offset -- which matters because
             the logged angles do not share the MJCF's zero (logged shoulder q
             is entirely outside the model's joint range).

The plant is integrated by MuJoCo itself, so ``frictionloss`` is the real
solver-side stiction constraint rather than a ``tanh`` stand-in, and ``armature``
and the integrator match what the RL env will actually run.

The commanded torque reproduces the GIM6010-8 firmware law, which closes its PD
loop locally at kHz on a 200 Hz command and includes a velocity feedforward:

    tau_cmd = clip(kp*(qdes - q) + kd*(qd_des - qd), +/- effort)

so the PD is recomputed every physics substep from the live state while qdes is
held for the control tick. Dropping the feedforward term biases the identified
damping. (The logged ``tau``/``iq*`` channels are unusable -- ``tau`` is
identically zero and the current channels are mis-scaled -- so the torque must
be reconstructed this way.)

Usage
-----
    # pooled fit (recommended): one shared armature, per-joint b/c/gravity
    python scripts/sysid/identify_actuator.py --mode pooled

    # per-joint armature, to see the spread the shared-armature fit resolves
    python scripts/sysid/identify_actuator.py --mode per-joint

    # score the parameters currently in the repo, no fitting
    python scripts/sysid/identify_actuator.py --mode eval-current

    # how much does the unknown bench pose move the answer?
    python scripts/sysid/identify_actuator.py --mode pooled --pose-sweep
"""

from __future__ import annotations

import argparse
import csv
import itertools
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.signal import savgol_filter

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "sysid"

JOINTS = ("shoulder", "hip", "knee")
# Joints downstream of each tested joint, i.e. the ones whose (unlogged) bench
# angle changes the moving subtree's inertia about the tested axis.
DOWNSTREAM = {"shoulder": ("hip", "knee"), "hip": ("knee",), "knee": ()}

KP, KD, EFFORT = 20.0, 0.5, 10.0


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
@dataclass
class Log:
    joint: str          # short name, e.g. "hip"
    channel: str        # log column prefix, e.g. "fl_hip"
    dt: float
    q: np.ndarray
    qdes: np.ndarray
    qd: np.ndarray
    qd_des: np.ndarray  # firmware velocity feedforward, d(qdes)/dt


def load_log(path: Path, joint: str) -> Log:
    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    cols = rows[0].keys()
    channel = next(c[: -len("_qdes")] for c in cols if c.endswith("_qdes"))

    def col(name: str) -> np.ndarray:
        return np.array([float(r[f"{channel}_{name}"]) for r in rows])

    t = np.array([float(r["t"]) for r in rows])
    dt = float(np.median(np.diff(t)))
    qdes = col("qdes")
    return Log(
        joint=joint,
        channel=channel,
        dt=dt,
        q=col("q"),
        qdes=qdes,
        qd=col("qd"),
        qd_des=savgol_filter(qdes, 11, 3, deriv=1, delta=dt),
    )


# --------------------------------------------------------------------------- #
# structural inertia, from the real mjlab model
# --------------------------------------------------------------------------- #
def structural_inertia(joint: str, pose: dict[str, float], leg: str = "fl") -> float:
    """Composite inertia of the moving subtree about `joint`'s axis, minus the
    model's own armature (which this script re-identifies)."""
    import sys

    sys.path.insert(0, str(REPO))
    from src.assets.robots.mbf.mbf_constants import get_spec  # noqa: PLC0415

    model = get_spec().compile()
    data = mujoco.MjData(model)
    for name, angle in pose.items():
        jid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, f"{leg}_{name}_joint"
        )
        data.qpos[model.jnt_qposadr[jid]] = angle
    mujoco.mj_forward(model, data)

    full = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, full, data.qM)
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{leg}_{joint}_joint")
    dof = model.jnt_dofadr[jid]
    return float(full[dof, dof] - model.dof_armature[dof])


# --------------------------------------------------------------------------- #
# 1-DOF MuJoCo plant
# --------------------------------------------------------------------------- #
def build_plant(
    inertia: float, armature: float, damping: float, frictionloss: float, timestep: float
) -> mujoco.MjModel:
    """A single hinge carrying `inertia`, with the env's solver settings.

    Gravity is disabled here; the fitted A*sin(q) + B*cos(q) supplies the bench
    gravity torque, whose orientation and encoder offset are both unknown.
    """
    spec = mujoco.MjSpec()
    spec.option.timestep = timestep
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_EULER
    spec.option.iterations = 10
    spec.option.ls_iterations = 20
    spec.option.gravity = [0.0, 0.0, 0.0]

    body = spec.worldbody.add_body(name="link")
    body.mass = 1e-9
    body.inertia = [inertia, inertia, inertia]  # COM at the joint origin
    body.add_joint(
        name="j",
        type=mujoco.mjtJoint.mjJNT_HINGE,
        axis=[0.0, 1.0, 0.0],
        armature=max(armature, 0.0),
        damping=max(damping, 0.0),
        frictionloss=max(frictionloss, 0.0),
    )
    return spec.compile()


def simulate(
    model: mujoco.MjModel,
    log: Log,
    grav_a: float,
    grav_b: float,
    delay: int,
    substeps: int,
) -> np.ndarray:
    """Replay the recorded command through the plant; return simulated q."""
    data = mujoco.MjData(model)
    data.qpos[0] = log.q[0]
    data.qvel[0] = log.qd[0]

    n = len(log.q)
    out = np.empty(n)
    for k in range(n):
        out[k] = data.qpos[0]
        j = max(k - delay, 0)
        target, target_vel = log.qdes[j], log.qd_des[j]
        for _ in range(substeps):
            q, qd = data.qpos[0], data.qvel[0]
            tau = np.clip(KP * (target - q) + KD * (target_vel - qd), -EFFORT, EFFORT)
            data.qfrc_applied[0] = tau + grav_a * np.sin(q) + grav_b * np.cos(q)
            mujoco.mj_step(model, data)
        if not np.isfinite(data.qpos[0]):
            return np.full(n, 1e3)  # diverged; let the optimiser walk away
    return out


# --------------------------------------------------------------------------- #
# fitting
# --------------------------------------------------------------------------- #
# Per-joint free parameters, in the order they are packed into the vector.
PER_JOINT = ("damping", "frictionloss", "grav_a", "grav_b")
BOUNDS = {
    "armature": (0.0, 0.03),
    "damping": (0.0, 1.0),
    "frictionloss": (0.0, 1.0),
    "grav_a": (-5.0, 5.0),
    "grav_b": (-5.0, 5.0),
}
GUESS = {"armature": 0.005, "damping": 0.17, "frictionloss": 0.15, "grav_a": 0.0, "grav_b": 0.0}


def _unpack(x: np.ndarray, joints: list[str], shared_armature: bool) -> dict:
    out: dict = {}
    if shared_armature:
        armature = x[0]
        rest = x[1:]
    else:
        armature = None
        rest = x
    stride = len(PER_JOINT) + (0 if shared_armature else 1)
    for i, j in enumerate(joints):
        chunk = rest[i * stride : (i + 1) * stride]
        vals = dict(zip(PER_JOINT, chunk[: len(PER_JOINT)]))
        vals["armature"] = armature if shared_armature else chunk[-1]
        out[j] = vals
    return out


def _pack(joints: list[str], shared_armature: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x0, lo, hi = [], [], []
    if shared_armature:
        x0.append(GUESS["armature"])
        lo.append(BOUNDS["armature"][0])
        hi.append(BOUNDS["armature"][1])
    keys = list(PER_JOINT) + ([] if shared_armature else ["armature"])
    for _ in joints:
        for k in keys:
            x0.append(GUESS[k])
            lo.append(BOUNDS[k][0])
            hi.append(BOUNDS[k][1])
    return np.array(x0), np.array(lo), np.array(hi)


def residuals(
    x: np.ndarray,
    logs: dict[str, Log],
    inertia: dict[str, float],
    delays: dict[str, int],
    substeps: int,
    shared_armature: bool,
) -> np.ndarray:
    params = _unpack(x, list(logs), shared_armature)
    chunks = []
    for j, log in logs.items():
        p = params[j]
        model = build_plant(
            inertia[j], p["armature"], p["damping"], p["frictionloss"], log.dt / substeps
        )
        sim = simulate(model, log, p["grav_a"], p["grav_b"], delays[j], substeps)
        chunks.append(sim - log.q)
    return np.concatenate(chunks)


def fit(
    logs: dict[str, Log],
    inertia: dict[str, float],
    delays: dict[str, int],
    substeps: int,
    shared_armature: bool,
) -> dict:
    joints = list(logs)
    x0, lo, hi = _pack(joints, shared_armature)
    sol = least_squares(
        residuals,
        x0,
        bounds=(lo, hi),
        args=(logs, inertia, delays, substeps, shared_armature),
        xtol=1e-10,
        ftol=1e-10,
        diff_step=1e-4,
    )
    return _unpack(sol.x, joints, shared_armature)


def rmse(
    logs: dict[str, Log],
    inertia: dict[str, float],
    params: dict,
    delays: dict[str, int],
    substeps: int,
) -> dict[str, float]:
    out = {}
    for j, log in logs.items():
        p = params[j]
        model = build_plant(
            inertia[j], p["armature"], p["damping"], p["frictionloss"], log.dt / substeps
        )
        sim = simulate(model, log, p["grav_a"], p["grav_b"], delays[j], substeps)
        out[j] = float(np.sqrt(np.mean((sim - log.q) ** 2)) * 1e3)  # mrad
    return out


def search_delays(
    logs: dict[str, Log],
    inertia: dict[str, float],
    substeps: int,
    max_delay: int,
    shared_armature: bool,
) -> dict[str, int]:
    """Pick each joint's command delay by fitting it alone at each lag."""
    best = {}
    for j, log in logs.items():
        scores = {}
        for d in range(max_delay + 1):
            one = {j: log}
            params = fit(one, {j: inertia[j]}, {j: d}, substeps, shared_armature=False)
            scores[d] = rmse(one, {j: inertia[j]}, params, {j: d}, substeps)[j]
        best[j] = min(scores, key=scores.get)
        detail = "  ".join(f"{d}:{v:.1f}" for d, v in scores.items())
        print(f"  {j:9s} lag mrad by steps -> {detail}   picked {best[j]}")
    return best


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def current_repo_params(joints: list[str], leg: str = "fl") -> dict:
    """Read what the repo's model actually ships today, so this baseline keeps
    tracking the MJCF instead of drifting out of date."""
    import sys

    sys.path.insert(0, str(REPO))
    from src.assets.robots.mbf.mbf_constants import get_spec  # noqa: PLC0415

    model = get_spec().compile()
    out = {}
    for j in joints:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{leg}_{j}_joint")
        dof = model.jnt_dofadr[jid]
        out[j] = dict(
            armature=float(model.dof_armature[dof]),
            damping=float(model.dof_damping[dof]),
            frictionloss=float(model.dof_frictionloss[dof]),
        )
    return out


def report(params: dict, errs: dict[str, float], inertia: dict[str, float], delays: dict[str, int]) -> None:
    print(f"\n{'joint':10s}{'I_struct':>10s}{'armature':>10s}{'damping':>9s}"
          f"{'friction':>10s}{'grav A':>9s}{'grav B':>9s}{'lag':>5s}{'RMSE':>9s}")
    print("-" * 81)
    for j, p in params.items():
        print(f"{j:10s}{inertia[j]:10.5f}{p['armature']:10.5f}{p['damping']:9.4f}"
              f"{p['frictionloss']:10.4f}{p.get('grav_a', 0):9.3f}{p.get('grav_b', 0):9.3f}"
              f"{delays[j]:5d}{errs[j]:8.1f}m")
    print(f"\nmean replay RMSE: {np.mean(list(errs.values())):.1f} mrad")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--mode", choices=("pooled", "per-joint", "eval-current"), default="pooled")
    ap.add_argument("--joints", nargs="+", default=list(JOINTS), choices=JOINTS)
    ap.add_argument("--substeps", type=int, default=1,
                    help="physics substeps per 5 ms control tick. Default 1 matches the env's "
                         "own 5 ms sim.timestep, which is what you want for values going into "
                         "the MJCF. Higher values approach the continuous-time plant and want "
                         "~25%% less damping -- do not mix the two.")
    ap.add_argument("--max-delay", type=int, default=3, help="max command lag searched, in control ticks")
    ap.add_argument("--pose", default="hip=-0.8,knee=-1.6",
                    help="bench angles [rad] of the unlogged downstream joints")
    ap.add_argument("--pose-sweep", action="store_true",
                    help="report how much the unknown bench pose moves I_struct")
    args = ap.parse_args()

    pose = {}
    for item in args.pose.split(","):
        if item.strip():
            k, v = item.split("=")
            pose[k.strip()] = float(v)

    logs = {j: load_log(args.data / f"{j}_sine_test_log.csv", j) for j in args.joints}
    inertia = {j: structural_inertia(j, pose) for j in args.joints}

    print(f"bench pose for unlogged downstream joints: {pose}")
    for j in args.joints:
        dep = DOWNSTREAM[j]
        note = "exact (no downstream joints)" if not dep else f"depends on {', '.join(dep)}"
        print(f"  I_struct[{j:9s}] = {inertia[j]:.5f} kg m^2   {note}")

    if args.pose_sweep:
        print("\nI_struct over plausible bench poses:")
        grid = [-2.2, -1.6, -1.0, -0.4, 0.0]
        for j in args.joints:
            dep = DOWNSTREAM[j]
            if not dep:
                print(f"  {j:9s} {inertia[j]:.5f} (invariant)")
                continue
            vals = [
                structural_inertia(j, dict(zip(dep, combo)))
                for combo in itertools.product(grid, repeat=len(dep))
            ]
            print(f"  {j:9s} {min(vals):.5f} .. {max(vals):.5f}  "
                  f"({100 * (max(vals) - min(vals)) / np.mean(vals):.0f}% spread)")

    if args.mode == "eval-current":
        live = current_repo_params(args.joints)
        params = {j: {**live[j], "grav_a": 0.0, "grav_b": 0.0} for j in args.joints}
        print("\nScoring the parameters currently in mbf_v2/mbf.xml (no gravity term fitted,")
        print("so this sits above the fitted floor even when damping/friction are optimal).")
        delays = {j: 0 for j in args.joints}
        report(params, rmse(logs, inertia, params, delays, args.substeps), inertia, delays)
        return

    print("\nsearching command delay:")
    delays = search_delays(logs, inertia, args.substeps, args.max_delay, shared_armature=False)

    shared = args.mode == "pooled"
    print(f"\nfitting ({'shared' if shared else 'per-joint'} armature)...")
    params = fit(logs, inertia, delays, args.substeps, shared)
    errs = rmse(logs, inertia, params, delays, args.substeps)
    report(params, errs, inertia, delays)

    print("\n" + "=" * 62)
    print("mbf_v2/mbf.xml joint defaults")
    print("=" * 62)
    for j, p in params.items():
        print(f'  <joint armature="{p["armature"]:.4f}" '
              f'damping="{p["damping"]:.3f}" frictionloss="{p["frictionloss"]:.3f}"/>  <!-- {j} -->')


if __name__ == "__main__":
    main()
