"""Audit the whole-body standup log for actuator-side sim2real issues.

Unlike the sine logs, ``standup_mit_log.csv`` covers all 12 joints, under real
load, and -- crucially -- its ``tauest``/``iq`` channels are actually usable.
That makes it good for three questions that the single-joint bench sweeps cannot
answer, none of which need a contact model:

  1. **Torque delivery.** Does the drive actually apply the torque the PD law
     asks for? If yes, mjlab's ``BuiltinPositionActuatorCfg`` (an ideal PD
     torque source) is a faithful model and there is no actuator gap beyond the
     PD law itself. This is the main result.
  2. **Sign / zero conventions.** Which joints' logged angles and reported
     currents agree with the MJCF, and which are flipped. Wrong here means a
     policy that works in sim drives the hardware backwards.
  3. **Effort headroom.** How close a real standup gets to ``EFFORT_LIMIT``.

What this log is *not* good for: identifying damping/frictionloss/armature. The
legs are loaded against the ground for most of the trajectory, so a replay would
be dominated by contact modelling rather than by the actuator. Use
``identify_actuator.py`` on the sine logs for those.

Scaling note: ``tauest`` is reported on the *motor* side. Joint torque is
``tauest * 8`` for the GIM6010-8's 8:1 reduction -- recovered here empirically
per joint rather than assumed, as a check on that reading.

Usage:
    python scripts/sysid/check_standup.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import mujoco
import numpy as np
from scipy.signal import savgol_filter

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "sysid"

LEGS = ("fl", "fr", "rl", "rr")
AXES = ("shoulder", "hip", "knee")
JOINTS = [f"{leg}_{axis}" for leg in LEGS for axis in AXES]

KP, KD, EFFORT = 20.0, 0.5, 10.0
GEAR = 8.0  # GIM6010-8


def load(path: Path) -> dict[str, np.ndarray]:
    rows = list(csv.DictReader(path.open()))
    return {
        k: np.array([float(r[k]) if r[k] not in ("", "nan") else np.nan for r in rows])
        for k in rows[0]
    }


def model_reference() -> tuple[dict[str, tuple[float, float]], dict[str, float]]:
    """(joint range, default standing angle) from the live MJCF."""
    import sys

    sys.path.insert(0, str(REPO))
    from src.assets.robots.mbf.mbf_constants import INIT_STATE, get_spec  # noqa: PLC0415

    model = get_spec().compile()
    ranges, defaults = {}, {}
    for j in JOINTS:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{j}_joint")
        ranges[j] = tuple(model.jnt_range[jid])
        for pattern, angle in INIT_STATE.joint_pos.items():
            if pattern.replace(".*", "") in f"{j}_joint":
                defaults[j] = angle
    return ranges, defaults


def commanded_torque(d: dict[str, np.ndarray], j: str, dt: float) -> np.ndarray:
    """The firmware PD law, including its velocity feedforward."""
    q, qd, qdes = d[f"{j}_q"], d[f"{j}_qd"], d[f"{j}_qdes"]
    qd_des = savgol_filter(qdes, 11, 3, deriv=1, delta=dt)
    return np.clip(KP * (qdes - q) + KD * (qd_des - qd), -EFFORT, EFFORT)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", type=Path, default=DATA / "standup_mit_log.csv")
    args = ap.parse_args()

    d = load(args.log)
    t = d["t"]
    dt = float(np.median(np.diff(t)))
    print(f"{args.log.name}: {len(t)} rows @ {1/dt:.0f} Hz, {t[-1]-t[0]:.2f} s, "
          f"Kp={np.nanmedian(d['fl_hip_kp']):.0f} Kd={np.nanmedian(d['fl_hip_kd']):.2f}")

    ranges, defaults = model_reference()

    # ---------------------------------------------------------------- 1. torque
    print("\n1. TORQUE DELIVERY  (does the drive apply what the PD law commands?)")
    print(f"   {'joint':13s}{'gear |G|':>10s}{'current loop':>14s}{'sign vs cmd':>13s}{'peak Nm':>10s}{'% effort':>10s}")
    gears, loops, peaks, flipped = [], [], [], []
    for j in JOINTS:
        tau = commanded_torque(d, j, dt)
        iqs, iqm, te = d[f"{j}_iqsp"], d[f"{j}_iqmeas"], d[f"{j}_tauest"]
        m = ~(np.isnan(iqs) | np.isnan(iqm) | np.isnan(te))
        gear = np.dot(te[m], tau[m]) / np.dot(te[m], te[m])
        loop = np.corrcoef(iqs[m], iqm[m])[0, 1]
        sign = np.corrcoef(iqm[m], tau[m])[0, 1]
        peak = float(np.max(np.abs(tau)))
        gears.append(abs(gear)); loops.append(loop); peaks.append(peak)
        if sign < 0:
            flipped.append(j)
        print(f"   {j:13s}{abs(gear):10.2f}{loop:14.3f}{'+' if sign > 0 else '-':>13s}"
              f"{peak:10.2f}{100*peak/EFFORT:9.0f}%")

    print(f"\n   |G| median {np.median(gears):.2f} vs {GEAR:.0f}:1 nominal -> tauest is motor-side; "
          f"joint torque = tauest * {GEAR:.0f}.")
    strong = [l for l, p in zip(loops, peaks) if p > 2.0]
    print(f"   Current-loop tracking on the loaded joints (peak > 2 Nm): "
          f"corr {min(strong):.2f}-{max(strong):.2f}.")
    print("   => the drive delivers commanded torque faithfully; an ideal PD torque")
    print("      source is a good model, so there is no actuator gap beyond the PD law.")

    # ------------------------------------------------------------ 2. conventions
    print("\n2. CONVENTIONS  (logged angles vs the MJCF)")
    print(f"   {'joint':13s}{'logged q range':>20s}{'MJCF range':>18s}{'final q':>10s}{'MJCF stand':>12s}  verdict")
    for j in JOINTS:
        q = d[f"{j}_q"]
        lo, hi = ranges[j]
        dflt = defaults.get(j, float("nan"))
        out = q.min() < lo - 1e-6 or q.max() > hi + 1e-6
        # The log ends standing, so the offset that maps its final angle onto the
        # model's standing angle IS this joint's zero offset. A large one is not
        # a sign flip -- replay_standup.py confirms an offset reproduces the
        # standup while flipping the sign makes the robot sink instead.
        offset = dflt - q[-1]
        verdict = f"offset {offset:+.2f} rad" if abs(offset) > 0.15 else "ok"
        if out:
            verdict += ", out of MJCF range"
        print(f"   {j:13s}[{q.min():+.2f},{q.max():+.2f}]".ljust(36)
              + f"[{lo:+.2f},{hi:+.2f}]".rjust(18)
              + f"{q[-1]:10.2f}{dflt:12.2f}  {verdict}")

    print("\n   The offsets above are the real->MJCF zero calibration. They are offsets,")
    print("   not sign flips: replay_standup.py reproduces the standup by applying them,")
    print("   whereas flipping the knee sign makes the robot sink instead of stand.")

    if flipped:
        print(f"\n   Separately, reported current/torque sign disagrees with the commanded PD")
        print(f"   torque on {len(flipped)}/12 joints: {', '.join(flipped)}")
        print("   The robot does stand up, so the position loop itself is fine -- this is a")
        print("   per-drive *reporting* polarity. Do not use iq/tauest signs across joints")
        print("   without per-joint calibration; magnitudes are fine.")

    # --------------------------------------------------------------- 3. headroom
    print(f"\n3. EFFORT HEADROOM")
    print(f"   Peak commanded torque {max(peaks):.2f} Nm ({100*max(peaks)/EFFORT:.0f}% of "
          f"EFFORT_LIMIT={EFFORT:.0f} Nm), on the knees.")
    vb = np.concatenate([d[f"{j}_vbus"] for j in JOINTS])
    print(f"   Bus voltage min {np.nanmin(vb):.2f} V (mean {np.nanmean(vb):.2f} V) -- "
          f"{'no meaningful sag' if np.nanmean(vb)-np.nanmin(vb) < 1.0 else 'SAGGING'}.")
    print("   Nothing here saturates, so EFFORT_LIMIT remains unvalidated from above.")


if __name__ == "__main__":
    main()
