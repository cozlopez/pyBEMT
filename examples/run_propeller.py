# -*- coding: utf-8 -*-

import matplotlib.pyplot as pl
import pandas as pd
from math import pi

from pybemt.solver import Solver

# 1. Run the solver and load experimental data
s = Solver('examples\\propeller.ini')
df, sections = s.run_sweep('v_inf', 20, 1, 44.0)
df_exp = pd.read_csv("examples\\propeller_data.csv", delimiter=' ')

# 2. Initialize the plot figure
fig, ax1 = pl.subplots(figsize=(8, 5))

# --- LEFT AXIS (Efficiency - eta) ---
# Line = Solid blue (C0-), Experimental = Blue circles (C0o)
line1 = ax1.plot(df['J'], df['eta'], 'C0-', label=r'BEMT, $\eta$')
dot1  = ax1.plot(df_exp['J'], df_exp['eta'], 'C0o', label=r'Exp, $\eta$')

ax1.set_xlabel('Advance Ratio ($J$)')
ax1.set_ylabel(r'Efficiency ($\eta$)')
ax1.grid(True, linestyle='--', alpha=0.5)

# --- RIGHT AXIS (Coefficients - CP and CT) ---
ax2 = ax1.twinx()
ax2.set_ylabel('$C_P, C_T$')

# CP: Line = Solid Orange (C1-), Experimental = Orange circles (C1o)
line2 = ax2.plot(df['J'], df['CP'], 'C1-', label=r'BEMT, $C_P$')
dot2  = ax2.plot(df_exp['J'], df_exp['CP'], 'C1o', label=r'Exp, $C_P$')

# CT: Line = Solid Green (C2-), Experimental = Green circles (C2o)
line3 = ax2.plot(df['J'], df['CT'], 'C2-', label=r'BEMT, $C_T$')
dot3  = ax2.plot(df_exp['J'], df_exp['CT'], 'C2o', label=r'Exp, $C_T$')

# --- COMBINE LEGENDS CLEANLY ---
# Gathering all line handles and labels from both axes to prevent overlapping/missing items
lines = line1 + dot1 + line2 + dot2 + line3 + dot3
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left')

pl.tight_layout()
pl.show()