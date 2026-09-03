#!/usr/bin/env python3
"""
Plot sine wave test data from CSV log file.

This script creates comprehensive plots showing:
- Position tracking (desired vs actual)
- Velocity tracking
- Torque output
- Current (Iq setpoint vs measured)
- Tracking error
- Bus voltage and current

Usage:
    python3 plot_sine_test.py sine_test_log.csv
    python3 plot_sine_test.py sine_test_log.csv --save
"""

import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path


def plot_sine_test_data(csv_file, save=False):
    """Plot all relevant data from a sine test CSV file."""

    # Read CSV
    df = pd.read_csv(csv_file)

    # Find the joint name (assume first joint in columns)
    joint_cols = [col for col in df.columns if "_qdes" in col]
    if not joint_cols:
        print("No joint data found in CSV!")
        return

    joint_name = joint_cols[0].replace("_qdes", "")
    print(f"Plotting data for joint: {joint_name}")

    # Extract time and data
    t = df["t"].values

    # Position data
    q_des = df[f"{joint_name}_qdes"].values
    q_act = df[f"{joint_name}_q"].values

    # Velocity data
    qd = df[f"{joint_name}_qd"].values

    # Torque data
    tau = df[f"{joint_name}_tau"].values

    # Current data
    iq_sp = df[f"{joint_name}_iqsp"].values
    iq_meas = df[f"{joint_name}_iqmeas"].values

    # Bus data
    vbus = df[f"{joint_name}_vbus"].values
    ibus = df[f"{joint_name}_ibus"].values

    # Calculate tracking error
    error = q_des - q_act

    # Create figure with subplots
    fig, axes = plt.subplots(4, 2, figsize=(14, 12))
    fig.suptitle(
        f"Sine Wave Test Results - {joint_name}", fontsize=16, fontweight="bold"
    )

    # 1. Position Tracking
    ax = axes[0, 0]
    ax.plot(t, q_des, "b-", label="Desired", linewidth=1.5)
    ax.plot(t, q_act, "r-", label="Actual", linewidth=1.5, alpha=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position (rad)")
    ax.set_title("Position Tracking")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Tracking Error
    ax = axes[0, 1]
    ax.plot(t, error, "g-", linewidth=1)
    ax.axhline(y=0, color="k", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Error (rad)")
    ax.set_title(f"Tracking Error (RMS: {np.sqrt(np.mean(error**2)):.4f} rad)")
    ax.grid(True, alpha=0.3)
    # Add error envelope
    ax.fill_between(t, -0.1, 0.1, color="green", alpha=0.1)

    # 3. Velocity
    ax = axes[1, 0]
    ax.plot(t, qd, "m-", linewidth=1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Velocity (rad/s)")
    ax.set_title("Velocity")
    ax.grid(True, alpha=0.3)

    # 4. Torque
    ax = axes[1, 1]
    ax.plot(t, tau, "orange", linewidth=1)
    ax.axhline(y=0, color="k", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Torque (Nm)")
    ax.set_title(f"Torque (Peak: {np.max(np.abs(tau)):.2f} Nm)")
    ax.grid(True, alpha=0.3)

    # 5. Current (Iq)
    ax = axes[2, 0]
    ax.plot(t, iq_sp, "b-", label="Setpoint", linewidth=1)
    ax.plot(t, iq_meas, "r-", label="Measured", linewidth=1, alpha=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Iq (A)")
    ax.set_title("Current (Iq)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. Current Saturation
    ax = axes[2, 1]
    # Calculate saturation indicator
    sat_threshold = 0.85
    sat_mask = (np.abs(iq_sp) > 1.0) & (np.abs(iq_meas) < sat_threshold * np.abs(iq_sp))
    sat_indicator = np.zeros_like(t)
    sat_indicator[sat_mask] = 1.0

    ax.fill_between(t, 0, sat_indicator, color="red", alpha=0.5, label="Saturation")
    ax.plot(t, np.abs(iq_sp), "b-", label="|Iq setpoint|", linewidth=1, alpha=0.5)
    ax.plot(t, np.abs(iq_meas), "r-", label="|Iq measured|", linewidth=1, alpha=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("|Iq| (A)")
    ax.set_title(f"Saturation Detection ({100*np.sum(sat_mask)/len(t):.1f}% saturated)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 7. Bus Voltage
    ax = axes[3, 0]
    ax.plot(t, vbus, "purple", linewidth=1)
    ax.axhline(
        y=24, color="k", linestyle="--", linewidth=0.5, alpha=0.5, label="Target 24V"
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Voltage (V)")
    ax.set_title(
        f"Bus Voltage (Min: {np.nanmin(vbus):.2f}V, Mean: {np.nanmean(vbus):.2f}V)"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 8. Bus Current
    ax = axes[3, 1]
    ax.plot(t, ibus, "brown", linewidth=1)
    ax.axhline(y=0, color="k", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Current (A)")
    ax.set_title(f"Bus Current (Peak: {np.nanmax(np.abs(ibus)):.2f}A)")
    ax.grid(True, alpha=0.3)

    # Adjust layout
    plt.tight_layout()

    # Save or show
    if save:
        output_file = Path(csv_file).stem + "_plots.png"
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {output_file}")
    else:
        plt.show()


def plot_summary_dashboard(csv_file, save=False):
    """Create a focused dashboard with key performance metrics."""

    df = pd.read_csv(csv_file)

    # Find the joint name
    joint_cols = [col for col in df.columns if "_qdes" in col]
    if not joint_cols:
        print("No joint data found in CSV!")
        return

    joint_name = joint_cols[0].replace("_qdes", "")
    t = df["t"].values

    # Extract data
    q_des = df[f"{joint_name}_qdes"].values
    q_act = df[f"{joint_name}_q"].values
    tau = df[f"{joint_name}_tau"].values
    iq_sp = df[f"{joint_name}_iqsp"].values
    iq_meas = df[f"{joint_name}_iqmeas"].values
    vbus = df[f"{joint_name}_vbus"].values

    # Create single figure with summary
    fig = plt.figure(figsize=(15, 8))

    # 1. Position tracking (main plot)
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(t, q_des, "b-", label="Desired", linewidth=1.5)
    ax1.plot(t, q_act, "r-", label="Actual", linewidth=1.5, alpha=0.8)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Position (rad)")
    ax1.set_title("Position Tracking")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    # 2. Tracking error
    ax2 = plt.subplot(2, 3, 2)
    error = q_des - q_act
    ax2.plot(t, error, "g-", linewidth=1)
    ax2.axhline(y=0, color="k", linestyle="--", linewidth=0.5, alpha=0.5)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Error (rad)")
    ax2.set_title(f"Tracking Error\nRMS: {np.sqrt(np.mean(error**2)):.4f} rad")
    ax2.grid(True, alpha=0.3)

    # 3. Torque vs Position phase plot
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(q_act, tau, "b.", alpha=0.3, markersize=1)
    ax3.axhline(y=0, color="k", linestyle="--", linewidth=0.5, alpha=0.5)
    ax3.set_xlabel("Position (rad)")
    ax3.set_ylabel("Torque (Nm)")
    ax3.set_title("Torque vs Position")
    ax3.grid(True, alpha=0.3)

    # 4. Current (Iq)
    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(t, iq_sp, "b-", label="Setpoint", linewidth=1, alpha=0.7)
    ax4.plot(t, iq_meas, "r-", label="Measured", linewidth=1, alpha=0.7)
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Iq (A)")
    ax4.set_title("Current Tracking")
    ax4.legend(loc="upper right")
    ax4.grid(True, alpha=0.3)

    # 5. Bus Voltage
    ax5 = plt.subplot(2, 3, 5)
    ax5.plot(t, vbus, "purple", linewidth=1)
    ax5.axhline(
        y=24, color="k", linestyle="--", linewidth=0.5, alpha=0.5, label="Target"
    )
    ax5.set_xlabel("Time (s)")
    ax5.set_ylabel("Voltage (V)")
    ax5.set_title(f"Bus Voltage\nMin: {np.nanmin(vbus):.2f}V")
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. Performance metrics text
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis("off")

    # Calculate metrics
    rms_error = np.sqrt(np.mean(error**2))
    peak_error = np.max(np.abs(error))
    peak_torque = np.max(np.abs(tau))
    sat_mask = (np.abs(iq_sp) > 1.0) & (np.abs(iq_meas) < 0.85 * np.abs(iq_sp))
    sat_pct = 100 * np.sum(sat_mask) / len(t)

    metrics_text = f"""
    PERFORMANCE SUMMARY - {joint_name}
    ─────────────────────────────────────
    
    Position Tracking:
    • RMS Error:     {rms_error:.4f} rad
    • Peak Error:    {peak_error:.4f} rad
    • Amplitude:     {np.max(q_des) - np.min(q_des):.3f} rad
    
    Torque:
    • Peak Torque:   {peak_torque:.2f} Nm
    • Mean Torque:   {np.mean(np.abs(tau)):.2f} Nm
    
    Current:
    • Peak Iq:       {np.nanmax(np.abs(iq_meas)):.2f} A
    • Saturation:    {sat_pct:.1f}%
    
    Power:
    • Min Voltage:   {np.nanmin(vbus):.2f} V
    • Voltage Drop:  {24 - np.nanmin(vbus):.2f} V
    """

    ax6.text(
        0.1,
        0.5,
        metrics_text,
        fontsize=10,
        verticalalignment="center",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()

    if save:
        output_file = Path(csv_file).stem + "_dashboard.png"
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        print(f"Dashboard saved to: {output_file}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot sine wave test data")
    parser.add_argument("csv_file", help="CSV file from sine test")
    parser.add_argument(
        "--save", action="store_true", help="Save plots instead of showing"
    )
    parser.add_argument(
        "--dashboard", action="store_true", help="Show summary dashboard only"
    )
    args = parser.parse_args()

    if not Path(args.csv_file).exists():
        print(f"Error: File '{args.csv_file}' not found!")
        return

    if args.dashboard:
        plot_summary_dashboard(args.csv_file, args.save)
    else:
        plot_sine_test_data(args.csv_file, args.save)


if __name__ == "__main__":
    main()
