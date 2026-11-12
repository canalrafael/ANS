"""
Master Plot Generator & Comparative Campaign Visualizer for ANS (3x3 Matrix)
=============================================================================
Compiles and visualizes all 9 experiments (3 attacks x 3 scenarios = 9000 frames)
to evaluate:
1. Physical Latency Degradation (Delta_phys) across all conditions.
2. Split Point Selection Shifts (Partition 0, 1, 2, 6, 20) under stress.
3. Post-attack persistent overhead and control lock-in (Delta_ctrl).
"""

import glob
import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')
SUMMARY_DIR = os.path.join(PLOTS_DIR, 'campaign_summary')
os.makedirs(SUMMARY_DIR, exist_ok=True)

# 1. Parse all 9 experiment CSVs
files = sorted(glob.glob(os.path.join(DATA_DIR, 'ans_*.csv')))
print(f"[CAMPAIGN] Found {len(files)} experiment runs.")

records = []
for f in files:
    df = pd.read_csv(f)
    fname = os.path.basename(f)
    parts = fname.split('_')
    attack = parts[1]
    scenario = parts[2] + '_' + parts[3]

    p1 = df[df['phase'].str.startswith('Phase 1')]
    p2 = df[df['phase'].str.startswith('Phase 2')]
    p3 = df[df['phase'].str.startswith('Phase 3')]

    j_p1 = p1['end_to_end_s'].mean() * 1000.0
    j_p2 = p2['end_to_end_s'].mean() * 1000.0
    j_p3 = p3['end_to_end_s'].mean() * 1000.0

    delta_phys = j_p2 - j_p1
    delta_ctrl = j_p3 - j_p1

    k0 = p1['partition'].mode()[0]
    ka = p2['partition'].mode()[0]
    k3 = p3['partition'].mode()[0]

    records.append({
        'file': fname,
        'attack': attack,
        'scenario': scenario,
        'j_p1': j_p1,
        'j_p2': j_p2,
        'j_p3': j_p3,
        'delta_phys': delta_phys,
        'delta_ctrl': delta_ctrl,
        'k0': k0,
        'ka': ka,
        'k3': k3,
        'p0_p1': (p1['partition'] == 0).mean() * 100.0,
        'p0_p2': (p2['partition'] == 0).mean() * 100.0,
        'p0_p3': (p3['partition'] == 0).mean() * 100.0,
        'p1_p1': (p1['partition'] == 1).mean() * 100.0,
        'p1_p2': (p2['partition'] == 1).mean() * 100.0,
        'p1_p3': (p3['partition'] == 1).mean() * 100.0,
        'p2_p1': (p1['partition'] == 2).mean() * 100.0,
        'p2_p2': (p2['partition'] == 2).mean() * 100.0,
        'p2_p3': (p3['partition'] == 2).mean() * 100.0,
        'p6_p1': (p1['partition'] == 6).mean() * 100.0,
        'p6_p2': (p2['partition'] == 6).mean() * 100.0,
        'p6_p3': (p3['partition'] == 6).mean() * 100.0,
    })

summary_df = pd.DataFrame(records)

# -------------------------------------------------------------
# Figure 1: Master 3x3 Heatmap of Physical Slowdown (Delta_phys)
# -------------------------------------------------------------
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

attacks = ['memory', 'cpu', 'network']
scenarios = ['scenario_a', 'scenario_b', 'scenario_c']

phys_matrix = np.zeros((3, 3))
ctrl_matrix = np.zeros((3, 3))

for i, atk in enumerate(attacks):
    for j, scn in enumerate(scenarios):
        row = summary_df[(summary_df['attack'] == atk) & (summary_df['scenario'] == scn)]
        if not row.empty:
            phys_matrix[i, j] = row['delta_phys'].iloc[0]
            ctrl_matrix[i, j] = row['delta_ctrl'].iloc[0]

im1 = ax1.imshow(phys_matrix, cmap='YlOrRd', aspect='auto')
cbar1 = plt.colorbar(im1, ax=ax1, pad=0.03)
cbar1.set_label('Delta Physical Degradation (ms)', fontsize=10, fontweight='bold')

ax1.set_xticks(np.arange(3))
ax1.set_yticks(np.arange(3))
ax1.set_xticklabels(['Scenario A\n(50/25/25)', 'Scenario B\n(33/33/34)', 'Scenario C\n(65/10/25)'], fontsize=10, fontweight='bold')
ax1.set_yticklabels(['Memory Contention\n(LPDDR4 / L2 Cache)', 'CPU Saturation\n(Compute 100%)', 'Network Delay\n(Latency + Jitter)'], fontsize=10, fontweight='bold')
ax1.set_title('Physical Degradation (Delta_phys in ms)\nAcross 3x3 Experimental Matrix', fontsize=12, fontweight='bold')

for i in range(3):
    for j in range(3):
        val = phys_matrix[i, j]
        color = 'white' if val > 1500 else 'black'
        ax1.text(j, i, f"+{val:.1f} ms", ha='center', va='center', color=color, fontweight='bold', fontsize=11)

# Figure 1B: Partition 0 (Full Offload) Shift Under Stress
shift_matrix = np.zeros((3, 3))
for i, atk in enumerate(attacks):
    for j, scn in enumerate(scenarios):
        row = summary_df[(summary_df['attack'] == atk) & (summary_df['scenario'] == scn)]
        if not row.empty:
            shift_matrix[i, j] = row['p0_p2'].iloc[0] - row['p0_p1'].iloc[0]

im2 = ax2.imshow(shift_matrix, cmap='Blues', aspect='auto')
cbar2 = plt.colorbar(im2, ax=ax2, pad=0.03)
cbar2.set_label('Shift to Partition 0 (Percentage Points)', fontsize=10, fontweight='bold')

ax2.set_xticks(np.arange(3))
ax2.set_yticks(np.arange(3))
ax2.set_xticklabels(['Scenario A\n(50/25/25)', 'Scenario B\n(33/33/34)', 'Scenario C\n(65/10/25)'], fontsize=10, fontweight='bold')
ax2.set_yticklabels(['Memory Contention', 'CPU Saturation', 'Network Delay'], fontsize=10, fontweight='bold')
ax2.set_title('Offloading Policy Shift: Change in Partition 0 Usage\n(P2 Attack vs P1 Nominal)', fontsize=12, fontweight='bold')

for i in range(3):
    for j in range(3):
        val = shift_matrix[i, j]
        color = 'white' if val > 10 else 'black'
        ax2.text(j, i, f"+{val:.1f}%", ha='center', va='center', color=color, fontweight='bold', fontsize=11)

plt.tight_layout()
out_fig1 = os.path.join(SUMMARY_DIR, '01_master_matrix_phys_and_shift.png')
plt.savefig(out_fig1)
plt.close()
print(f"[PLOT] Generated: {out_fig1}")

# -------------------------------------------------------------
# Figure 2: Partition 1 & Partition 2 Vulnerability Breakdown
# -------------------------------------------------------------
fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(15, 5.5), dpi=300)

# Aggregate across scenarios by attack type
atk_agg = summary_df.groupby('attack').mean(numeric_only=True)
atk_names = ['CPU Saturation', 'Memory Contention', 'Network Delay']
atk_keys = ['cpu', 'memory', 'network']

x = np.arange(len(atk_keys))
w = 0.28

# Partition 1 usage across phases
p1_nom = [atk_agg.loc[k, 'p1_p1'] for k in atk_keys]
p1_atk = [atk_agg.loc[k, 'p1_p2'] for k in atk_keys]
p1_rec = [atk_agg.loc[k, 'p1_p3'] for k in atk_keys]

r1 = ax3.bar(x - w, p1_nom, w, label='Phase 1 (Nominal)', color='#2ecc71', alpha=0.85, edgecolor='black')
r2 = ax3.bar(x, p1_atk, w, label='Phase 2 (Attack Active)', color='#e74c3c', alpha=0.85, edgecolor='black')
r3 = ax3.bar(x + w, p1_rec, w, label='Phase 3 (Recovery)', color='#f39c12', alpha=0.85, edgecolor='black')

ax3.set_ylabel('Partition 1 Selection Frequency (%)', fontsize=11, fontweight='bold')
ax3.set_title('Partition 1 (Conv1_1 - ~100MB Tensor):\nSelective Suppression by Memory Contention', fontsize=12, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(atk_names, fontsize=10, fontweight='bold')
ax3.legend(frameon=True)

for bar_group in [r1, r2, r3]:
    for b in bar_group:
        h = b.get_height()
        ax3.annotate(f'{h:.1f}%', xy=(b.get_x() + b.get_width() / 2, h),
                     xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Partition 2 extinction
p2_nom = [atk_agg.loc[k, 'p2_p1'] for k in atk_keys]
p2_atk = [atk_agg.loc[k, 'p2_p2'] for k in atk_keys]
p2_rec = [atk_agg.loc[k, 'p2_p3'] for k in atk_keys]

r4 = ax4.bar(x - w, p2_nom, w, label='Phase 1 (Nominal)', color='#2ecc71', alpha=0.85, edgecolor='black')
r5 = ax4.bar(x, p2_atk, w, label='Phase 2 (Attack Active)', color='#e74c3c', alpha=0.85, edgecolor='black')
r6 = ax4.bar(x + w, p2_rec, w, label='Phase 3 (Recovery)', color='#f39c12', alpha=0.85, edgecolor='black')

ax4.set_ylabel('Partition 2 Selection Frequency (%)', fontsize=11, fontweight='bold')
ax4.set_title('Partition 2 (Conv1_2):\nComplete Extinction During Local Contention', fontsize=12, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(atk_names, fontsize=10, fontweight='bold')
ax4.legend(frameon=True)

for bar_group in [r4, r5, r6]:
    for b in bar_group:
        h = b.get_height()
        ax4.annotate(f'{h:.1f}%', xy=(b.get_x() + b.get_width() / 2, h),
                     xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
out_fig2 = os.path.join(SUMMARY_DIR, '02_partition_1_and_2_dynamics.png')
plt.savefig(out_fig2)
plt.close()
print(f"[PLOT] Generated: {out_fig2}")
