# -*- coding: utf-8 -*-
#%%
import sys
import os
import configparser
from time import sleep
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.pylab import pi
from Planform_plotting import plot_3d_profiles
from pybemt.solver import Solver


# Insert project root to sys.path 
current_dir = os.path.dirname(os.path.abspath(__file__)) # .../pyBEMT/examples
project_root = os.path.dirname(current_dir)              # .../pyBEMT
sys.path.insert(0, project_root)
#TODO: Update the value of the ini file so that it takes into accound the change in airfoil
#TODO: Fix 3d plotting call becouse it currently takes whatever is in the ini file (first file)
ini_filepath = os.path.join(current_dir, 'finalinputdata','samplecopy2.ini')
config = configparser.ConfigParser()
config.read(ini_filepath)
plot_efficiency = True
plot_power = False
plot_shape = False

#Airfoils = np.array(['CLARKY', 'GOE_408', 'GOE_450', 'NACA_4412', 'NACA_63815', 'NRELS814'])
Airfoils = np.array(['NACA_63815'])
efficiency = np.array([])
df_store = []
df2_store = []
advance_ratio_flight_condition = []
for i in range(len(Airfoils)):
    config['rotor']['section'] = Airfoils[i]
    # Extract raw section name from configuration
    raw_section = config.get('rotor', 'section').split()[0]

    airfoil_path = os.path.join(project_root, "examples", "airfoils", "airfoilgeometry", f"{raw_section}"+'geo.dat')

    print(f"--> Target Airfoil Coordinates Path: {airfoil_path}")

    v_inf = config.getfloat('case', 'v_inf')

    #Run solver and generate performance data for the propeller across a sweep of advance ratios (J) and RPMs
    s = Solver(ini_filepath)
    file_title = os.path.splitext(os.path.basename(ini_filepath))[0]
    df, sections = s.run_sweep('v_inf', 15, 0, v_inf*1.2)
    df2, sections2 = s.run_sweep('rpm', 20, 700, 1300)

    # find advance ration at speed v_inf
    idx = (df['v_inf'] - v_inf).abs().idxmin()
    advance_ratio_flight_condition.append(df.loc[idx, 'J'])
    print(f" Advance Ratio at v_inf = {v_inf} m/s: J = {df.loc[idx, 'J']:.3f}")
    

    
    efficiency = np.append(efficiency, df.loc[idx, 'eta'])
    print(f"Efficiency for {Airfoils[i]}: {efficiency[-1]:.3f}")

    df_store.append(df)
    df2_store.append(df2)

ma_xeta_index = np.argmax(efficiency)

best_airfoil_data = df_store[ma_xeta_index]

advance_ratio_at_max_eta  = advance_ratio_flight_condition[ma_xeta_index]


if plot_shape:
    # Run profile visualization
    if raw_section == "NACA_63815":
        print("")
        print("Using NACA 63815 no shape data available")
        print("")
    else:    
        plot_3d_profiles(
            radius_stations = np.array([float(x) for x in config.get('rotor', 'radius').split()]),
            chords = np.array([float(x) for x in config.get('rotor', 'chord').split()]),
            pitch_angles = np.array([float(x) for x in config.get('rotor', 'pitch').split()]),
            airfoil_path = airfoil_path
        )

if plot_efficiency:
    
    fig, ax1 = plt.subplots(figsize=(8, 5))

    # --- LEFT AXIS (Efficiency - eta) ---
    # Line = Solid blue (C0-)

    line1 = ax1.plot(best_airfoil_data['J'], best_airfoil_data['eta'], 'C0-', linewidth=2, label=r'BEMT $\eta$')

    ax1.set_xlabel('Advance Ratio ($J$)')
    ax1.set_ylabel(r'Propulsive Efficiency ($\eta$)', color='C0')
    ax1.tick_params(axis='y', labelcolor='C0')
    ax1.grid(True, linestyle='--', alpha=0.5)

    # --- RIGHT AXIS (Coefficients - CP and CT) ---
    ax2 = ax1.twinx()
    ax2.set_ylabel('Coefficients ($C_P, C_T$)')

    # CP Line = Solid Orange (C1-)
    line2 = ax2.plot(best_airfoil_data['J'], best_airfoil_data['CP'], 'C1-', linewidth=2, label=r'BEMT $C_P$')

    # CT Line = Solid Green (C2-)
    line3 = ax2.plot(best_airfoil_data['J'], best_airfoil_data['CT'], 'C2-', linewidth=2, label=r'BEMT $C_T$')

    ax1.scatter(advance_ratio_at_max_eta, best_airfoil_data.loc[idx, 'eta'], color='red', s=50, zorder=5)
    # --- COMBINE AND DRAW THE LEGEND ---
    # Combines line handles from both distinct Y-axes into a single legend box
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')

    plt.tight_layout()
    plt.show()

if plot_power:
    #%%
    print(df2['P'])
    # 2. Initialize a single figure window with 2 separate subplots (stacked vertically)
    fig, (ax1, ax2,ax3) = plt.subplots(3, 1, figsize=(8, 12), sharex=True)
    speed_of_sound = 316  # m/s at sea level
    mach_1_rpm = (speed_of_sound / (pi * s.rotor.diameter)) * 60
    ax1.axvline(mach_1_rpm, color='C2', linestyle='--', label=f'Mach 1 RPM ({mach_1_rpm:.0f} RPM)')
    #ax1.set_xlim(0, mach_1_rpm*1.25)
    # --- SUBPLOT 1: THRUST (Top) ---
    # BEMT Line (Note: Check if your df uses lowercase 'rpm' or uppercase 'RPM')
    ax1.plot(df2['rpm'], df2['T'], 'C0-', linewidth=2, label='BEMT Thrust')



    ax1.set_ylabel('Thrust (N)')
    ax1.legend(loc='upper left')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.set_title(f'Performance: {file_title}')

    # --- SUBPLOT 2: POWER (Bottom) ---
    # BEMT Line
    

    
    ax2.plot(df2['rpm'], df2['P'], 'C1-', linewidth=1, label='BEMT Power(W)',color = 'green')
    ax22 = ax2.twinx()
    ax2.plot(df2['rpm'], df2['P']/745.7, 'C1--', linewidth=1, label='BEMT Power (HP)')
    ax2.axvline(mach_1_rpm, color='C2', linestyle='--', label=f'Mach 1 RPM ({mach_1_rpm:.0f} RPM)')

    #ax2.set_xlim(0, mach_1_rpm*1.25)
    ax2.set_xlabel('Rotational Speed (RPM)')
    ax2.set_ylabel('Power (W)')
    ax22.set_ylabel('Power (HP)')
    ax2.legend(loc='upper left')
    ax2.grid(True, linestyle='--', alpha=0.5)





    # Blade tip speed
    df2['tip_speed'] = (df2['rpm'] / 60) * (pi * s.rotor.diameter)
    ax3.plot(df2['rpm'], df2['tip_speed'], 'C0-', label='Blade Tip Speed')
    #Mach 1 line

    ax3.axhline(speed_of_sound, color='C1', linestyle='--', label='Mach 1')
    #Find rpm at which tip speed reaches Mach 1
    mach_1_rpm = (speed_of_sound / (pi * s.rotor.diameter)) * 60
    ax3.axvline(mach_1_rpm, color='C2', linestyle='--', label=f'Mach 1 RPM ({mach_1_rpm:.0f} RPM)')
    ax3.set_xlabel('Rotational Speed (RPM)')
    ax3.set_ylabel('Blade Tip Speed (m/s)')

    #ax3.set_xlim(0, mach_1_rpm*1.25)
    ax3.set_title(f'Blade Tip Speed vs RPM: {file_title}')
    ax3.grid(True, linestyle='--', alpha=0.5)
    ax3.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

    

   