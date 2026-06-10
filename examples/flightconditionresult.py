#%%
import sys
import os
import configparser
from time import sleep
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd
from matplotlib.pylab import pi
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter  # NEW: For smoothing out the pitch curve
from pybemt.solver import Solver
from isa import get_ISA

# =========================================================================
# ----- 1: DIRECTORY SETUP & ENVIRONMENT INTEGRATION ----------------------
# =========================================================================
current_dir = os.path.dirname(os.path.abspath(__file__)) # .../pyBEMT/examples
project_root = os.path.dirname(current_dir)              # .../pyBEMT
sys.path.insert(0, project_root)

ini_filepath = os.path.join(current_dir, 'finalinputdata', 'samplecopy2.ini')
airfoil_path = os.path.join(current_dir, 'airfoils', 'airfoilgeometry', 'NACA_63815geo.dat')

config = configparser.ConfigParser()
config.read(ini_filepath)

# =========================================================================
# ----- 2: INITIAL VARIABLES & CRUISE/PROP CONFIGURATION ------------------
# =========================================================================
v_inf = config.getfloat('case', 'v_inf')
rpm = config.getint('case', 'rpm')
M_cruise = 0.62
cruise_altitude = 25000
P_cruise, rho_cruise, tempK_cruise, speed_of_sound_cruise = get_ISA(cruise_altitude)
v_inf_cruise = M_cruise * speed_of_sound_cruise

pitch_values = np.array(config['rotor']['pitch'].split()).astype(float)
radius_stations = np.array(config.get('rotor', 'radius').split(), dtype=float)
chords = np.array(config.get('rotor', 'chord').split(), dtype=float)

sweep = np.array([0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]) 
max_pitch_correction = min(pitch_values)

# =========================================================================
# ----- 3: RUN OPTIMIZATION SWEEP (MAX CORRECTION TO 0) -------------------
# =========================================================================
climb_altitudes = np.linspace(0, cruise_altitude, num=30)  
v_inf_climb = np.linspace(72, v_inf_cruise, num=30)  

master_pitch_corrections = np.linspace(max_pitch_correction, 0, num=100)

optimized_corrections = []
final_climb_efficiencies = []
final_climb_powers = []
last_best_index = 0 

print("Starting Climb Optimization Sweep (Max Correction Down to 0)...")

for i in range(len(climb_altitudes)):
    current_alt = climb_altitudes[i]
    current_vel = v_inf_climb[i]
    current_rho = get_ISA(current_alt)[1]
    
    config['case']['v_inf'] = str(current_vel)  
    config['additionaldata']['altitude'] = str(current_alt)
    config['fluid']['rho'] = str(current_rho)  
    
    current_choices = master_pitch_corrections[last_best_index:]
    
    sweep_corrections = []
    sweep_etas = []
    sweep_powers = []
    
    for j in range(len(current_choices)):
        correction_guess = current_choices[j]
        config['rotor']['pitch'] = ' '.join(map(str, pitch_values - correction_guess))

        with open(ini_filepath, 'w') as configfile:
            config.write(configfile)
            
        try:
            s = Solver(ini_filepath)
            T, Q, P, section_df = s.run()
            J, CT, CQ, CP, eta = s.rotor_coeffs(T, Q, P)
            
            if 2.7e6 <= P <= 3.5e6 and 0.0 < eta <= 1.0:
                sweep_corrections.append(correction_guess)
                sweep_etas.append(eta)
                sweep_powers.append(P)
        except Exception:
            continue 

    if len(sweep_etas) > 0:
        best_idx_in_sweep = np.argmax(sweep_etas)
        best_correction = sweep_corrections[best_idx_in_sweep]
        best_eta = sweep_etas[best_idx_in_sweep]
        best_power = sweep_powers[best_idx_in_sweep]
        last_best_index = np.where(master_pitch_corrections == best_correction)[0][0]
    else:
        best_correction = optimized_corrections[-1] if i > 0 else master_pitch_corrections[0]
        best_eta = final_climb_efficiencies[-1] if i > 0 else 0.0
        best_power = final_climb_powers[-1] if i > 0 else 0.0
        print(f"  [Warning] Step {i+1}: No configuration met requirements. Holding baseline setting.")
        
    optimized_corrections.append(best_correction)
    final_climb_efficiencies.append(best_eta)
    final_climb_powers.append(best_power)
    print(f"[Step {i+1:02d}/30] Alt: {current_alt:5.0f}m | Power: {best_power/1e6:.2f}MW | Correction Locked: {best_correction:.2f}°")

print("\nOptimization Complete!")

# =========================================================================
# ----- NEW: SMOOTHING OUT THE PITCH PROFILE VALUES -----------------------
# =========================================================================
print("Applying Savitzky-Golay smoothing filter to pitch schedule...")
# window_length must be odd, polyorder controls curve flexibility. 
# Adjust window_length (e.g., 7 or 9) if you want even aggressive smoothing.
window_size = 7
if len(optimized_corrections) > window_size:
    smoothed_corrections = savgol_filter(optimized_corrections, window_length=window_size, polyorder=2)
    # Enforce monotonic trend strictly so smoothing doesn't introduce illegal backward twist bounces
    for idx in range(1, len(smoothed_corrections)):
        if smoothed_corrections[idx] > smoothed_corrections[idx-1]:
            smoothed_corrections[idx] = smoothed_corrections[idx-1]
else:
    smoothed_corrections = np.array(optimized_corrections)

# Optional Plot validation to see the raw vs smooth profiles before animating
plt.figure(figsize=(8, 4))
plt.plot(climb_altitudes, optimized_corrections, 'r--', label='Raw Stepped Correction')
plt.plot(climb_altitudes, smoothed_corrections, 'g-', label='Smoothed Correction Schedule')
plt.xlabel('Altitude (m)')
plt.ylabel('Pitch Correction (deg)')
plt.title('Governor Pitch Dampening Smooth Profile Match')
plt.legend()
plt.grid(True)
plt.show()

# =========================================================================
# ----- 4: GEOMETRIC AUXILIARY UTILITIES & GEOMETRY CALCULATIONS ----------
# =========================================================================
def load_normalized_airfoil(path):
    ux, uy = [], []
    lx, ly = [], []
    parsing_lower = False
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                if len(ux) > 0: parsing_lower = True
                continue
            parts = line.split()
            if len(parts) == 2:
                x, y = float(parts[0]), float(parts[1])
                if not parsing_lower:
                    ux.append(x); uy.append(y)
                else:
                    lx.append(x); ly.append(y)
    return np.array(ux), np.array(uy), np.array(lx), np.array(ly)

try:
    ux_norm, uy_norm, lx_norm, ly_norm = load_normalized_airfoil(airfoil_path)
except FileNotFoundError:
    beta = np.linspace(0, np.pi, 60)
    ux_norm = 0.5 * (1.0 - np.cos(beta))
    uy_norm = 0.12 * (0.2969*np.sqrt(ux_norm) - 0.1260*ux_norm - 0.3516*ux_norm**2 + 0.2843*ux_norm**3 - 0.1015*ux_norm**4)
    lx_norm = ux_norm.copy()
    ly_norm = -uy_norm.copy()

def rotate_coordinates(x, y, angle_rad, cx, cy):
    x_rot = (x - cx) * np.cos(angle_rad) - (y - cy) * np.sin(angle_rad) + cx
    y_rot = (x - cx) * np.sin(angle_rad) + (y - cy) * np.cos(angle_rad) + cy
    return x_rot, y_rot

# =========================================================================
# ----- 5: ANIMATION GENERATION ENVIRONMENT -------------------------------
# =========================================================================
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection="3d")

NUM_SPLINES = 8
index_sampling = np.linspace(0, len(ux_norm) - 1, NUM_SPLINES, dtype=int)
tip_length_percentage = 0.20

def update_blade_geometry(frame_idx):
    ax.clear()
    
    # CRITICAL SWITCH: We now pull from our new smoothed dataset array
    current_correction = smoothed_corrections[frame_idx]
    current_altitude_m = climb_altitudes[frame_idx]
    current_eta_val = final_climb_efficiencies[frame_idx]
    current_power_val = final_climb_powers[frame_idx]
    
    active_pitch_angles = pitch_values - current_correction
    
    upper_spline_nodes = np.zeros((NUM_SPLINES, len(radius_stations), 3))
    te_points = []
    
    for i in range(len(radius_stations)):
        r = radius_stations[i]
        c = chords[i]
        theta = np.radians(active_pitch_angles[i])
        sweepinternal = -sweep[i]
        
        ux_scaled, uy_scaled = ux_norm * c, uy_norm * c
        lx_scaled, ly_scaled = lx_norm * c, ly_norm * c
        rot_x, rot_y = 0.25 * c, 0.0

        ux_rot, uy_rot = rotate_coordinates(ux_scaled, uy_scaled, theta, rot_x, rot_y)
        lx_rot, ly_rot = rotate_coordinates(lx_scaled, ly_scaled, theta, rot_x, rot_y)

        z_upper = np.full_like(ux_rot, r)
        z_lower = np.full_like(lx_rot, r)

        z_upper = z_upper + uy_rot * np.sin(np.radians(sweepinternal))
        z_lower = z_lower + ly_rot * np.sin(np.radians(sweepinternal))
        uy_rot = uy_rot - np.tan(np.radians(sweepinternal)) * z_upper

        station_color = plt.cm.coolwarm(i / len(radius_stations))
        ax.plot(ux_rot, uy_rot, z_upper, color=station_color, lw=1.0, alpha=0.4)
        ax.plot(lx_rot, ly_rot, z_lower, color=station_color, lw=1.0, alpha=0.4)

        for s_idx in range(NUM_SPLINES):
            target_node = index_sampling[s_idx]
            upper_spline_nodes[s_idx, i, 0] = ux_rot[target_node]
            upper_spline_nodes[s_idx, i, 1] = uy_rot[target_node]
            upper_spline_nodes[s_idx, i, 2] = z_upper[target_node]
            
        te_points.append([ux_rot[0], uy_rot[0], z_upper[0]])

    main_radii = radius_stations.copy()
    span_extension = (main_radii[-1] - main_radii[0]) * tip_length_percentage
    tip_radius = main_radii[-1] + span_extension

    te_arr = np.array(te_points)
    tmp_cs_x = CubicSpline(main_radii, te_arr[:, 0])
    tmp_cs_y = CubicSpline(main_radii, te_arr[:, 1])

    last_te_x, slope_x = tmp_cs_x(main_radii[-1]), tmp_cs_x(main_radii[-1], 1)
    last_te_y, slope_y = tmp_cs_y(main_radii[-1]), tmp_cs_y(main_radii[-1], 1)

    tip_x_target = last_te_x + slope_x * span_extension
    tip_y_target = last_te_y + slope_y * span_extension
    tip_z_target = main_radii[-1] + span_extension

    z_main_fine = np.linspace(main_radii[0], main_radii[-1], 100)
    z_tip_fine = np.linspace(main_radii[-1], tip_radius, 20)
    rel_z = z_tip_fine - main_radii[-1]
    dz = tip_radius - main_radii[-1]
    t = rel_z / dz

    concavity_factor = 0.1
    for s_idx in range(NUM_SPLINES):
        cs_x = CubicSpline(main_radii, upper_spline_nodes[s_idx, :, 0])
        cs_y = CubicSpline(main_radii, upper_spline_nodes[s_idx, :, 1])
        cs_z = CubicSpline(main_radii, upper_spline_nodes[s_idx, :, 2])
        
        p_x, d_x = cs_x(main_radii[-1]), cs_x(main_radii[-1], 1)
        p_y, d_y = cs_y(main_radii[-1]), cs_y(main_radii[-1], 1)
        p_z, d_z = cs_z(main_radii[-1]), cs_z(main_radii[-1], 1)
        
        t_d_x, t_d_y, t_d_z = d_x * concavity_factor, d_y * concavity_factor, d_z * concavity_factor

        tip_x = (2*t**3 - 3*t**2 + 1)*p_x + (t**3 - 2*t**2 + t)*d_x*dz + (-2*t**3 + 3*t**2)*tip_x_target + (t**3 - t**2)*t_d_x*dz
        tip_y = (2*t**3 - 3*t**2 + 1)*p_y + (t**3 - 2*t**2 + t)*d_y*dz + (-2*t**3 + 3*t**2)*tip_y_target + (t**3 - t**2)*t_d_y*dz
        tip_z = (2*t**3 - 3*t**2 + 1)*p_z + (t**3 - 2*t**2 + t)*d_z*dz + (-2*t**3 + 3*t**2)*tip_z_target + (t**3 - t**2)*t_d_z*dz

        total_x = np.concatenate([cs_x(z_main_fine), tip_x[1:]])
        total_y = np.concatenate([cs_y(z_main_fine), tip_y[1:]])
        total_z = np.concatenate([cs_z(z_main_fine), tip_z[1:]])
        
        ax.plot(total_x, total_y, total_z, color=plt.cm.viridis(s_idx / NUM_SPLINES), lw=1.5)

    cs_te_x = CubicSpline(main_radii, te_arr[:, 0])
    cs_te_y = CubicSpline(main_radii, te_arr[:, 1])
    cs_te_z = CubicSpline(main_radii, te_arr[:, 2])
    
    te_p_x, te_d_x = cs_te_x(main_radii[-1]), cs_te_x(main_radii[-1], 1)
    te_p_y, te_d_y = cs_te_y(main_radii[-1]), cs_te_y(main_radii[-1], 1)
    te_p_z, te_d_z = cs_te_z(main_radii[-1]), cs_te_z(main_radii[-1], 1)
    
    te_t_d_x, te_t_d_y, te_t_d_z = te_d_x * concavity_factor, te_d_y * concavity_factor, te_d_z * concavity_factor
    
    te_tip_x = (2*t**3 - 3*t**2 + 1)*te_p_x + (t**3 - 2*t**2 + t)*te_d_x*dz + (-2*t**3 + 3*t**2)*tip_x_target + (t**3 - t**2)*te_t_d_x*dz
    te_tip_y = (2*t**3 - 3*t**2 + 1)*te_p_y + (t**3 - 2*t**2 + t)*te_d_y*dz + (-2*t**3 + 3*t**2)*tip_y_target + (t**3 - t**2)*te_t_d_y*dz
    te_tip_z = (2*t**3 - 3*t**2 + 1)*te_p_z + (t**3 - 2*t**2 + t)*te_d_z*dz + (-2*t**3 + 3*t**2)*tip_z_target + (t**3 - t**2)*te_t_d_z*dz
    
    total_te_x = np.concatenate([cs_te_x(z_main_fine), te_tip_x[1:]])
    total_te_y = np.concatenate([cs_te_y(z_main_fine), te_tip_y[1:]])
    total_te_z = np.concatenate([cs_te_z(z_main_fine), te_tip_z[1:]])
    ax.plot(total_te_x, total_te_y, total_te_z, color="crimson", lw=1.8, label="Trailing Edge")

    ax.set_title(f"Smoothed Dynamic Propeller Pitch feathering Profile Loop\n"
                 f"Altitude: {current_altitude_m:.0f} m | Power: {current_power_val/1e6:.2f} MW | Max η: {current_eta_val:.3f}", fontsize=12)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Radial Station Z (m)")
    
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-0.5, 0.5)
    ax.set_zlim(0.0, tip_radius + 0.2)
    ax.legend(loc="upper left")
    ax.view_init(elev=20, azim=-45)

ani = animation.FuncAnimation(fig, update_blade_geometry, frames=len(climb_altitudes), interval=200, repeat=True)
plt.show()
# %%