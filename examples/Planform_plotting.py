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


def plot_3d_profiles(radius_stations, chords, pitch_angles,sweep, airfoil_path):

    

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

    # CHANGED: Shape is now (NUM_SPLINES, len(radius_stations), 3) to include Z values
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

       #ux_rot, uy_rot = sweep_correction(ux_rot, uy_rot,sweepinternal)
        #lx_rot, ly_rot = sweep_correction(lx_rot, ly_rot,sweepinternal)

        z_upper = np.full_like(ux_rot, r)
        z_lower = np.full_like(lx_rot, r)

        z_upper = z_upper + uy_rot * np.sin(np.radians(sweepinternal))
        z_lower = z_lower + ly_rot * np.sin(np.radians(sweepinternal))
        
        uy_rot = uy_rot- np.tan(np.radians(sweepinternal))*z_upper

        
        # Plot original stations
        color = plt.cm.coolwarm(i / len(radius_stations))
        ax.plot(ux_rot, uy_rot, z_upper, color=color, lw=1.2, alpha=0.6)
        ax.plot(lx_rot, ly_rot, z_lower, color=color, lw=1.2, alpha=0.6)

        # --- PLOT BASELINE INTERNAL REFERENCE CHORD LINE ---
        chord_line_x = np.array([0.0, c])
        chord_line_y = np.array([0.0, 0.0])
        
        # 1. Rotate the chord line just like before
        clx_rot, cly_rot = rotate_coordinates(chord_line_x, chord_line_y, theta, rot_x, rot_y)
        
        # 2. CREATE A DEDICATED Z-AXIS FOR THE CHORD LINE (Starts flat at radius 'r')
        clz_rot = np.array([r, r], dtype=float)
        
        # 3. CORRECT THE Z-AXIS FIRST (Using cly_rot, just like you did with uy_rot/ly_rot)
        clz_rot = clz_rot + cly_rot * np.sin(np.radians(sweepinternal))
        
        # 4. CORRECT THE Y-AXIS (Using the newly shifted clz_rot)
        cly_rot = cly_rot - np.tan(np.radians(sweepinternal)) * clz_rot
        
        # 5. PLOT USING THE UPDATED CORRECTIONS FOR ALL THREE AXES (clx_rot, cly_rot, clz_rot)
        ax.plot(clx_rot, cly_rot, clz_rot, color="black", linestyle=":", alpha=0.3, label="Chord Line" if i == 0 else "")
        # Note: If you want to sample points along the ACTUAL airfoil upper surface:
        # --- Inside your loop, right after your chord line plotting logic ---
        for s_idx in range(NUM_SPLINES):
            target_node = index_sampling[s_idx]
            upper_spline_nodes[s_idx, i, 0] = ux_rot[target_node]
            upper_spline_nodes[s_idx, i, 1] = uy_rot[target_node]
            upper_spline_nodes[s_idx, i, 2] = z_upper[target_node] # <-- ADD THIS LINE
            
        # Track the Trailing Edge point
        te_points.append([ux_rot[0], uy_rot[0], z_upper[0]])








    te_points = np.array(te_points)

    # --- 6. GENERATE AND PLOT SPLINES ---
    z_fine = np.linspace(radius_stations[0], radius_stations[-1], 200)

    # Loop and programmatically evaluate your 8 chordwise splines
    for s_idx in range(NUM_SPLINES):
        cs_x = CubicSpline(radius_stations, upper_spline_nodes[s_idx, :, 0])
        cs_y = CubicSpline(radius_stations, upper_spline_nodes[s_idx, :, 1])
        cs_z = CubicSpline(radius_stations, upper_spline_nodes[s_idx, :, 2]) # <-- NEW Z SPLINE
        
        ax.plot(
            cs_x(z_fine),
            cs_y(z_fine),
            cs_z(z_fine), # <-- CHANGED: Replaced static z_fine with the dynamic swept Z spline array!
            color=plt.cm.viridis(s_idx / NUM_SPLINES),
            lw=1.5, 
        )

    # Compute separate splines for X, Y, AND the new swept Z position
    cs_te_x = CubicSpline(radius_stations, te_points[:, 0])
    cs_te_y = CubicSpline(radius_stations, te_points[:, 1])
    cs_te_z = CubicSpline(radius_stations, te_points[:, 2]) # New Z spline interpolation!

    # Plot the trailing edge using all 3 interpolated curves
    ax.plot(
        cs_te_x(z_fine), 
        cs_te_y(z_fine), 
        cs_te_z(z_fine), # Pass the interpolated Z coordinates array here
        color="crimson", 
        lw=1.5, 
        label="Trailing Edge"
    )

    # --- 7. DISPLAY OPTIONS ---
    ax.set_title("3D Profile Stacking with Chordwise Surface Splines", fontsize=14)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Radial Station Z (m)")
    ax.set_aspect("equal")
    ax.grid(False)
    ax.legend(loc="upper left")
    #ax.view_init(elev=35, azim=-45)
    ax.view_init(elev=0, azim=0)

    plt.show()

    #TSHIRT
    # ax.set_aspect("equal")
    # ax.grid(False)
    # #ax.legend(loc="upper left")
    # ax.view_init(elev=90, azim=-90)

    # plt.show()


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
    plot_3d_profiles(radius_stations, chords, pitch_angles,sweep,airfoil_path)


#TSHIRT DESING
# if __name__ == "__main__":
#     airfoil_path = f"examples\\airfoils\\airfoilgeometry\\special.dat"

#     radius_stations = np.array([ 0.3 ,    0.46875, 0.6375,  0.80625, 0.975,   1.14375, 1.3125,  1.48125, 1.65, 1.81875, 1.9875,  2.15625, 2.325,   2.49375, 2.6625,  2.83125, 3. ])
#     chords = np.array([0.18141697, 0.17157945, 0.16092818, 0.1496789,  0.13804736, 0.1262493, 0.11450045, 0.10301657, 0.09201339, 0.08170666, 0.07231212, 0.06404551, 0.05712257, 0.05175905, 0.04817068, 0.04657321, 0.0471823])
#     pitch_angles = np.array([80.553557109468, 77.75433300959658, 75.03254440963453, 72.39676106545053, 69.85357280139317, 67.4076627024995, 65.06195002150379, 62.81778023488389, 60.67514109546428, 58.632886766202155, 56.68895621193342, 54.8405762027772, 53.084443034720046, 51.41688, 49.83, 48.33, 46.9])

#     plot_3d_profiles(radius_stations, chords, pitch_angles, airfoil_path)