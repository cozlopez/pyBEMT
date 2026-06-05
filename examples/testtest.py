# -*- coding: utf-8 -*-

import matplotlib.pyplot as pl
import pandas as pd
from pybemt.solver import Solver

# 1. Run the solver sweep and load experimental data
s = Solver('examples\\testdatatata.ini')
df, sections = s.run_sweep('rpm', 20, 700, 1250)

df_exp = pd.read_csv("examples\\tmotor28_data.csv", delimiter=';')

# 2. Initialize a single figure window with 2 separate subplots (stacked vertically)
fig, (ax1, ax2) = pl.subplots(2, 1, figsize=(8, 8), sharex=True)

# --- SUBPLOT 1: THRUST (Top) ---
# BEMT Line (Note: Check if your df uses lowercase 'rpm' or uppercase 'RPM')
ax1.plot(df['rpm'], df['T'], 'C0-', linewidth=2, label='BEMT Thrust')
# Experimental Dots
ax1.plot(df_exp['RPM'], df_exp['T(N)'], 'C0o', label='Experiment')

ax1.set_ylabel('Thrust (N)')
ax1.legend(loc='upper left')
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.set_title('T-Motor 28 Performance Verification')

# --- SUBPLOT 2: POWER (Bottom) ---
# BEMT Line
ax2.plot(df['rpm'], df['P'], 'C1-', linewidth=2, label='BEMT Power')
# Experimental Dots
ax2.plot(df_exp['RPM'], df_exp['P(W)']/1000000, 'C1o', label='Experiment')

ax2.set_xlabel('Rotational Speed (RPM)')
ax2.set_ylabel('Power (MW)')
ax2.legend(loc='upper left')
ax2.grid(True, linestyle='--', alpha=0.5)

# 3. Clean up spacing and render
pl.tight_layout()
pl.show()