#%%
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
import os
import configparser
import sys

from sphinx import config

# Insert project root to sys.path 
current_dir = os.path.dirname(os.path.abspath(__file__)) # .../pyBEMT/examples
project_root = os.path.dirname(current_dir)              # .../pyBEMT
sys.path.insert(0, project_root)
#TODO: Update the value of the ini file so that it takes into accound the change in airfoil
#TODO: Fix 3d plotting call becouse it currently takes whatever is in the ini file (first file)


def plot_3d_profiles(radius_stations, chords, pitch_angles, sweep, airfoil_path):

    # --- 2. LOAD BASELINE AIRFOIL GEOMETRY ---
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

    ux_norm, uy_norm, lx_norm, ly_norm = load_normalized_airfoil(airfoil_path)

    # --- 3. INITIALIZE INTERPOLATION SAMPLING PARAMETERS ---
    NUM_SPLINES = 8
    index_sampling = np.linspace(0, len(ux_norm) - 1, NUM_SPLINES, dtype=int)

    # Shape is (NUM_SPLINES, len(radius_stations), 3) to include Z values
    upper_spline_nodes = np.zeros((NUM_SPLINES, len(radius_stations), 3))
    te_points = []

    # --- 4. INITIALIZE 3D PLOT ---
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    def rotate_coordinates(x, y, angle_rad, cx, cy):
        x_rot = (x - cx) * np.cos(angle_rad) - (y - cy) * np.sin(angle_rad) + cx
        y_rot = (x - cx) * np.sin(angle_rad) + (y - cy) * np.cos(angle_rad) + cy
        return x_rot, y_rot

    # --- 5. LOOP & STACK EACH PROFILE WITH TWIST ---
    for i in range(len(radius_stations)):
        r = radius_stations[i]
        c = chords[i]
        theta = np.radians(pitch_angles[i])
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

        # Plot original stations
        color = plt.cm.coolwarm(i / len(radius_stations))
        ax.plot(ux_rot, uy_rot, z_upper, color=color, lw=1.2, alpha=0.6)
        ax.plot(lx_rot, ly_rot, z_lower, color=color, lw=1.2, alpha=0.6)

        # --- PLOT BASELINE INTERNAL REFERENCE CHORD LINE ---
        chord_line_x = np.array([0.0, c])
        chord_line_y = np.array([0.0, 0.0])
        
        clx_rot, cly_rot = rotate_coordinates(chord_line_x, chord_line_y, theta, rot_x, rot_y)
        clz_rot = np.array([r, r], dtype=float)
        clz_rot = clz_rot + cly_rot * np.sin(np.radians(sweepinternal))
        cly_rot = cly_rot - np.tan(np.radians(sweepinternal)) * clz_rot
        
        ax.plot(clx_rot, cly_rot, clz_rot, color="black", linestyle=":", alpha=0.3, label="Chord Line" if i == 0 else "")

        for s_idx in range(NUM_SPLINES):
            target_node = index_sampling[s_idx]
            upper_spline_nodes[s_idx, i, 0] = ux_rot[target_node]
            upper_spline_nodes[s_idx, i, 1] = uy_rot[target_node]
            upper_spline_nodes[s_idx, i, 2] = z_upper[target_node]
            
        # Track the Trailing Edge point
        te_points.append([ux_rot[0], uy_rot[0], z_upper[0]])

    # =========================================================================
    # --- TWO-SECTOR BLADE TIP CLOSURE CONFIGURATION ---
    # =========================================================================
    # 1. Determine where the terminal tip point sits along the span
# =========================================================================
    # --- TWO-SECTOR BLADE TIP CLOSURE CONFIGURATION ---
    # =========================================================================
    #%%
    # 1. Control the tip length here (e.g., 0.10 means the tip is 10% of the total blade span)
    # =========================================================================
    # --- TWO-SECTOR BLADE TIP CLOSURE CONFIGURATION (EXTRAPOLATED SWEEP) ---
    # =========================================================================
    tip_length_percentage = 0.20 
    
    main_radii = radius_stations.copy() 
    span_extension = (main_radii[-1] - main_radii[0]) * tip_length_percentage 
    tip_radius = main_radii[-1] + span_extension

    # 1. Fit a temporary spline to the trailing edge data to read its exiting trajectory
    # (Using your collected main body te_points)
    te_arr = np.array(te_points)
    tmp_cs_x = CubicSpline(main_radii, te_arr[:, 0])
    tmp_cs_y = CubicSpline(main_radii, te_arr[:, 1])
    tmp_cs_z = CubicSpline(main_radii, te_arr[:, 2])

    # 2. Extract the physical position and exiting slope vector at the last station
    last_te_x, slope_x = tmp_cs_x(main_radii[-1]), tmp_cs_x(main_radii[-1], 1)
    last_te_y, slope_y = tmp_cs_y(main_radii[-1]), tmp_cs_y(main_radii[-1], 1)
    last_te_z, slope_z = tmp_cs_z(main_radii[-1]), tmp_cs_z(main_radii[-1], 1)

    # 3. EXTRAPOLATE THE TARGET SPACE: 
    # Instead of stopping at the last station's X/Y, we multiply the exiting trajectory vector
    # by our span extension length. This actively pushes the tip point further back along the sweep line!
    tip_x_target = last_te_x + slope_x * span_extension
    tip_y_target = last_te_y + slope_y * span_extension
    tip_z_target = last_te_z + span_extension 

    # 4. Establish structural calculation vectors for the fine paths
    z_main_fine = np.linspace(main_radii[0], main_radii[-1], 180)
    z_tip_fine = np.linspace(main_radii[-1], tip_radius, 25)
    rel_z = z_tip_fine - main_radii[-1]
    dz = tip_radius - main_radii[-1]

    # --- 6. GENERATE AND PLOT SPLINES ---
    # SECTOR A: UPPER SURFACE SPLINES
    # --- 6. GENERATE AND PLOT SPLINES (CUBIC CONCAVE TIP METHOD) ---
    # SECTOR A: UPPER SURFACE SPLINES
    for s_idx in range(NUM_SPLINES):
        # Fit main body coordinates cleanly
        cs_x = CubicSpline(main_radii, upper_spline_nodes[s_idx, :, 0])
        cs_y = CubicSpline(main_radii, upper_spline_nodes[s_idx, :, 1])
        cs_z = CubicSpline(main_radii, upper_spline_nodes[s_idx, :, 2])
        
        # Calculate exit boundary positions (p) and tangent slope vectors (d)
        p_x, d_x = cs_x(main_radii[-1]), cs_x(main_radii[-1], 1)
        p_y, d_y = cs_y(main_radii[-1]), cs_y(main_radii[-1], 1)
        p_z, d_z = cs_z(main_radii[-1]), cs_z(main_radii[-1], 1)
        
        # --- CONCAVITY TUNING COEFFICIENTS ---
        # A value of 0.0 makes the curve collapse instantly inward (maximum concave).
        # A value of 1.0 matches the natural incoming speed. Let's force it very tight:
        concavity_factor = 0.1 
        
        # Calculate target tip velocity vectors based on incoming trajectory 
        t_d_x = d_x * concavity_factor
        t_d_y = d_y * concavity_factor
        t_d_z = d_z * concavity_factor

        # Cubic Hermite Solver for the Tip Segment
        # Solves: Position = A*t^3 + B*t^2 + C*t + D over a normalized 0-to-1 tip space
        t = rel_z / dz
        
        tip_x = (2*t**3 - 3*t**2 + 1)*p_x + (t**3 - 2*t**2 + t)*d_x*dz + (-2*t**3 + 3*t**2)*tip_x_target + (t**3 - t**2)*t_d_x*dz
        tip_y = (2*t**3 - 3*t**2 + 1)*p_y + (t**3 - 2*t**2 + t)*d_y*dz + (-2*t**3 + 3*t**2)*tip_y_target + (t**3 - t**2)*t_d_y*dz
        tip_z = (2*t**3 - 3*t**2 + 1)*p_z + (t**3 - 2*t**2 + t)*d_z*dz + (-2*t**3 + 3*t**2)*tip_z_target + (t**3 - t**2)*t_d_z*dz

        # Stitch main body and custom tip parameters together
        total_x = np.concatenate([cs_x(z_main_fine), tip_x[1:]])
        total_y = np.concatenate([cs_y(z_main_fine), tip_y[1:]])
        total_z = np.concatenate([cs_z(z_main_fine), tip_z[1:]])
        
        ax.plot(total_x, total_y, total_z, color=plt.cm.viridis(s_idx / NUM_SPLINES), lw=1.5)

    # SECTOR B: TRAILING EDGE SPLINE
    te_points = np.array(te_points)
    
    cs_te_x = CubicSpline(main_radii, te_points[:, 0])
    cs_te_y = CubicSpline(main_radii, te_points[:, 1])
    cs_te_z = CubicSpline(main_radii, te_points[:, 2])
    
    te_p_x, te_d_x = cs_te_x(main_radii[-1]), cs_te_x(main_radii[-1], 1)
    te_p_y, te_d_y = cs_te_y(main_radii[-1]), cs_te_y(main_radii[-1], 1)
    te_p_z, te_d_z = cs_te_z(main_radii[-1]), cs_te_z(main_radii[-1], 1)
    
    # Apply the same concavity compression to the trailing edge curve
    te_t_d_x = te_d_x * concavity_factor
    te_t_d_y = te_d_y * concavity_factor
    te_t_d_z = te_d_z * concavity_factor
    
    t = rel_z / dz
    te_tip_x = (2*t**3 - 3*t**2 + 1)*te_p_x + (t**3 - 2*t**2 + t)*te_d_x*dz + (-2*t**3 + 3*t**2)*tip_x_target + (t**3 - t**2)*te_t_d_x*dz
    te_tip_y = (2*t**3 - 3*t**2 + 1)*te_p_y + (t**3 - 2*t**2 + t)*te_d_y*dz + (-2*t**3 + 3*t**2)*tip_y_target + (t**3 - t**2)*te_t_d_y*dz
    te_tip_z = (2*t**3 - 3*t**2 + 1)*te_p_z + (t**3 - 2*t**2 + t)*te_d_z*dz + (-2*t**3 + 3*t**2)*tip_z_target + (t**3 - t**2)*te_t_d_z*dz
    
    total_te_x = np.concatenate([cs_te_x(z_main_fine), te_tip_x[1:]])
    total_te_y = np.concatenate([cs_te_y(z_main_fine), te_tip_y[1:]])
    total_te_z = np.concatenate([cs_te_z(z_main_fine), te_tip_z[1:]])
    
    ax.plot(total_te_x, total_te_y, total_te_z, color="crimson", lw=1.5, label="Trailing Edge")

    # --- 7. DISPLAY OPTIONS ---
    ax.set_title("3D Profile Stacking with Chordwise Surface Splines", fontsize=14)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Radial Station Z (m)")
    ax.set_aspect("equal")
    ax.grid(False)
    ax.legend(loc="upper left")
    ax.view_init(elev=0, azim=0)

    plt.show()


if __name__ == "__main__":
    airfoil_path = f"examples\\airfoils\\airfoilgeometry\\NACA_63815geo.dat"
    ini_filepath = os.path.join(current_dir, 'samplecopy2.ini')
    
    config = configparser.ConfigParser()
    config.read(ini_filepath)
    radius_stations = np.array(config.get('rotor', 'radius').split(), dtype=float)

    chords = np.array(config.get('rotor', 'chord').split(), dtype=float)
    sweep = np.array(config.get('rotor', 'sweep').split(), dtype=float)
    sweep = np.array([0,2,3,4,5,6,7,8,9,10,11])
    print(f"--> Radius Stations: {radius_stations}")
    print(f"--> Chords: {chords}")
    pitch_angles = np.array(config.get('rotor', 'pitch').split(), dtype=float)
    print(f"--> Pitch Angles: {pitch_angles}")
    plot_3d_profiles(radius_stations, chords, pitch_angles, sweep, airfoil_path)
# %%
