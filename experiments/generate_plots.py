"""
Publication-Quality Plot Generator for ANS Stress & Attack Experiments
======================================================================
Visualizes adaptation dynamics across the 3-phase experimental lifecycle:
  Phase 1 (Nominal) -> Phase 2 (Attack) -> Phase 3 (Recovery)

Generates:
1. Multi-phase partition trajectory timeline with shaded attack regimes.
2. End-to-end and offloading latency across phases.
3. Partition selection distribution before, during, and after attack.
4. Physical degradation (Delta_phys) and persistent control overhead (Delta_ctrl).
"""

import argparse
import glob
import os
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


def get_latest_csv():
    files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV datasets found in {DATA_DIR}")
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]


def plot_experiment(csv_path: str):
    print(f"\n[PLOTTING] Loading dataset: {csv_path}")
    df = pd.read_csv(csv_path)
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    out_dir = os.path.join(PLOTS_DIR, base_name)
    os.makedirs(out_dir, exist_ok=True)

    has_phases = 'phase' in df.columns
    total_frames = len(df)

    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.rcParams['font.sans-serif'] = 'Arial'
    plt.rcParams['axes.edgecolor'] = '#cccccc'
    plt.rcParams['axes.linewidth'] = 0.8

    # -------------------------------------------------------------
    # 1. Timeline of Partition Selection with Phase Shading
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(13, 5.5), dpi=300)

    delay_col = 'offload_delay_s' if 'offload_delay_s' in df.columns else 'delay_s'

    scatter = ax.scatter(
        df['frame'], df['partition'],
        c=df[delay_col], cmap='viridis_r',
        alpha=0.8, s=26, edgecolors='none'
    )
    cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label('Offloading Delay (seconds)', fontsize=11, fontweight='bold')

    # Rolling mean partition line
    rolling_part = df['partition'].rolling(window=20, min_periods=1).mean()
    ax.plot(df['frame'], rolling_part, color='#e74c3c', linewidth=2.5, label='Rolling Mean Partition (w=20)')

    # Add phase shading if available
    if has_phases:
        phases = df['phase'].unique()
        p1_df = df[df['phase'] == 'Phase 1 - Nominal']
        p2_df = df[df['phase'] == 'Phase 2 - Attack']
        p3_df = df[df['phase'] == 'Phase 3 - Recovery']

        if not p1_df.empty:
            ax.axvspan(p1_df['frame'].min(), p1_df['frame'].max(), color='#2ecc71', alpha=0.12, label='Phase 1 (Nominal)')
        if not p2_df.empty:
            ax.axvspan(p2_df['frame'].min(), p2_df['frame'].max(), color='#e74c3c', alpha=0.18, label='Phase 2 (Attack Active)')
        if not p3_df.empty:
            ax.axvspan(p3_df['frame'].min(), p3_df['frame'].max(), color='#f39c12', alpha=0.12, label='Phase 3 (Post-Attack Recovery)')

    scenario_name = df['scenario'].iloc[0] if 'scenario' in df.columns else 'Baseline'
    attack_name = df['attack_type'].iloc[len(df)//2] if 'attack_type' in df.columns else 'None'

    ax.set_title(f'ANS Adaptation Dynamics: Partition Point Evolution\n[Scenario: {scenario_name} | Attack: {attack_name.upper()} | {total_frames} Frames]',
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Frame Index', fontsize=11, fontweight='bold')
    ax.set_ylabel('Selected Partition Point', fontsize=11, fontweight='bold')
    ax.set_yticks(sorted(df['partition'].unique()))
    ax.legend(loc='upper right', frameon=True, fontsize=10)
    plt.tight_layout()

    p1_path = os.path.join(out_dir, "01_partition_adaptation_timeline.png")
    fig.savefig(p1_path)
    plt.close(fig)
    print(f"  -> Generated: {p1_path}")

    # -------------------------------------------------------------
    # 2. End-to-End Latency Trajectory
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(13, 5), dpi=300)
    e2e_col = 'end_to_end_s' if 'end_to_end_s' in df.columns else 'delay_s'
    avg_col = 'avg_end_to_end_s' if 'avg_end_to_end_s' in df.columns else 'avg_s'

    ax.plot(df['frame'], df[e2e_col], color='#3498db', alpha=0.35, linewidth=1.0, label='Instantaneous Latency')
    ax.plot(df['frame'], df[e2e_col].rolling(20, min_periods=1).mean(), color='#2980b9', linewidth=2.0, label='Rolling Mean (w=20)')
    ax.plot(df['frame'], df[avg_col], color='#2c3e50', linewidth=2.5, linestyle='--', label='Cumulative Average')

    if has_phases:
        if not p1_df.empty:
            ax.axvspan(p1_df['frame'].min(), p1_df['frame'].max(), color='#2ecc71', alpha=0.10)
        if not p2_df.empty:
            ax.axvspan(p2_df['frame'].min(), p2_df['frame'].max(), color='#e74c3c', alpha=0.15)
        if not p3_df.empty:
            ax.axvspan(p3_df['frame'].min(), p3_df['frame'].max(), color='#f39c12', alpha=0.10)

    ax.set_title(f'End-to-End Latency Response Across Attack Lifecycle', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Frame Index', fontsize=11, fontweight='bold')
    ax.set_ylabel('Latency (seconds)', fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', frameon=True, fontsize=10)
    plt.tight_layout()

    p2_path = os.path.join(out_dir, "02_latency_trajectory.png")
    fig.savefig(p2_path)
    plt.close(fig)
    print(f"  -> Generated: {p2_path}")

    # -------------------------------------------------------------
    # 3. Partition Distribution Comparison by Phase
    # -------------------------------------------------------------
    if has_phases:
        fig, ax = plt.subplots(figsize=(11, 5), dpi=300)
        phase_groups = df.groupby(['partition', 'phase']).size().unstack(fill_value=0)
        phase_groups.plot(kind='bar', ax=ax, width=0.75, colormap='tab10', edgecolor='#333333', linewidth=0.5)

        ax.set_title('Partition Point Distribution Across Lifecycle Phases', fontsize=13, fontweight='bold', pad=12)
        ax.set_xlabel('Partition Point', fontsize=11, fontweight='bold')
        ax.set_ylabel('Selection Count (Frames)', fontsize=11, fontweight='bold')
        ax.legend(title='Lifecycle Phase', frameon=True)
        plt.tight_layout()

        p3_path = os.path.join(out_dir, "03_partition_distribution_by_phase.png")
        fig.savefig(p3_path)
        plt.close(fig)
        print(f"  -> Generated: {p3_path}")

        # -------------------------------------------------------------
        # 4. Compute Scientific Control Metrics (Delta_phys & Delta_ctrl)
        # -------------------------------------------------------------
        k0_mode = p1_df['partition'].mode()[0] if not p1_df.empty else None
        ka_mode = p2_df['partition'].mode()[0] if not p2_df.empty else None
        k3_mode = p3_df['partition'].mode()[0] if not p3_df.empty else None

        j_k0 = p1_df[e2e_col].mean() if not p1_df.empty else 0.0
        j_ka = p2_df[e2e_col].mean() if not p2_df.empty else 0.0
        j_k3 = p3_df[e2e_col].mean() if not p3_df.empty else 0.0

        delta_phys = (j_ka - j_k0) * 1000.0  # in ms
        delta_ctrl = (j_k3 - j_k0) * 1000.0  # in ms
        lock_in = (k0_mode != k3_mode) if (k0_mode is not None and k3_mode is not None) else False

        print("\n" + "=" * 65)
        print("          SCIENTIFIC CONTROL EVALUATION SUMMARY")
        print("=" * 65)
        print(f"  * Scenario:                 {scenario_name}")
        print(f"  * Injected Attack:          {attack_name.upper()}")
        print(f"  * Total Processed Frames:   {total_frames}")
        print(f"  * Dominant Split (Phase 1): Partition {k0_mode} (Mean: {j_k0*1000:.1f} ms)")
        print(f"  * Dominant Split (Phase 2): Partition {ka_mode} (Mean: {j_ka*1000:.1f} ms)")
        print(f"  * Dominant Split (Phase 3): Partition {k3_mode} (Mean: {j_k3*1000:.1f} ms)")
        print(f"  -------------------------------------------------------------")
        print(f"  * Delta_phys (Direct Degradation):   {delta_phys:+.2f} ms")
        print(f"  * Delta_ctrl (Persistent Overhead):  {delta_ctrl:+.2f} ms")
        print(f"  * Control Lock-in (k^0 != k^rec):     {'YES (Suboptimal Lock)' if lock_in else 'NO (Recovered)'}")
        print("=" * 65 + "\n")

    print(f"[DONE] All plots and metrics generated in: {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate plots for ANS experiment data")
    parser.add_argument('--csv', type=str, default=None, help="Path to CSV dataset (defaults to latest)")
    args = parser.parse_args()

    csv_path = args.csv if args.csv else get_latest_csv()
    plot_experiment(csv_path)


if __name__ == '__main__':
    main()
