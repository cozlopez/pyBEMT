import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline



def plot_3d_profiles(radius_stations, chords, pitch_angles, airfoil_path):

    

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
    # Linearly sample exactly 8 index nodes across the upper surface points
    index_sampling = np.linspace(0, len(ux_norm) - 1, NUM_SPLINES, dtype=int)

    # Matrix storage for splines: shape = (8_splines, 17_stations, 2_coords)
    upper_spline_nodes = np.zeros((NUM_SPLINES, len(radius_stations), 2))
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

        ux_scaled, uy_scaled = ux_norm * c, uy_norm * c
        lx_scaled, ly_scaled = lx_norm * c, ly_norm * c
        rot_x, rot_y = 0.25 * c, 0.0

        ux_rot, uy_rot = rotate_coordinates(ux_scaled, uy_scaled, theta, rot_x, rot_y)
        lx_rot, ly_rot = rotate_coordinates(lx_scaled, ly_scaled, theta, rot_x, rot_y)

        z_upper = np.full_like(ux_rot, r)
        z_lower = np.full_like(lx_rot, r)

        # Plot original stations
        color = plt.cm.coolwarm(i / len(radius_stations))
        ax.plot(ux_rot, uy_rot, z_upper, color=color, lw=1.2, alpha=0.6)
        ax.plot(lx_rot, ly_rot, z_lower, color=color, lw=1.2, alpha=0.6)

        # Plot baseline internal reference chord line
        chord_line_x = np.array([0.0, c])
        chord_line_y = np.array([0.0, 0.0])
        clx_rot, cly_rot = rotate_coordinates(chord_line_x, chord_line_y, theta, rot_x, rot_y)
        ax.plot(clx_rot, cly_rot, [r, r], color="black", linestyle=":", alpha=0.3)

        # Note: If you want to sample points along the ACTUAL airfoil upper surface:
        for s_idx in range(NUM_SPLINES):
            target_node = index_sampling[s_idx]
            upper_spline_nodes[s_idx, i, 0] = ux_rot[target_node]
            upper_spline_nodes[s_idx, i, 1] = uy_rot[target_node]
            
        # Track the Trailing Edge point
        te_points.append([clx_rot[-1], cly_rot[-1]])

    te_points = np.array(te_points)

    # --- 6. GENERATE AND PLOT SPLINES METICULOUSLY ---
    z_fine = np.linspace(radius_stations[0], radius_stations[-1], 200)

    # Loop and programmatically evaluate your 8 chordwise splines
    for s_idx in range(NUM_SPLINES):
        cs_x = CubicSpline(radius_stations, upper_spline_nodes[s_idx, :, 0])
        cs_y = CubicSpline(radius_stations, upper_spline_nodes[s_idx, :, 1])
        
        ax.plot(
            cs_x(z_fine),
            cs_y(z_fine),
            z_fine,
            color=plt.cm.viridis(s_idx / NUM_SPLINES),
            lw=1.5,
            label=f"Surface Spline {s_idx+1}" if s_idx in [0, NUM_SPLINES-1] else "" # Keep legend clean
        )

    # Compute and plot Trailing Edge Spline curve
    cs_te_x = CubicSpline(radius_stations, te_points[:, 0])
    cs_te_y = CubicSpline(radius_stations, te_points[:, 1])
    ax.plot(cs_te_x(z_fine), cs_te_y(z_fine), z_fine, color="crimson", lw=2.5, label="Trailing Edge")

    # --- 7. DISPLAY OPTIONS ---
    ax.set_title("3D Profile Stacking with Chordwise Surface Splines", fontsize=14)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Radial Station Z (m)")
    ax.set_aspect("equal")
    ax.grid(False)
    ax.legend(loc="upper left")
    ax.view_init(elev=35, azim=-45)

    plt.show()


if __name__ == "__main__":
    airfoil_path = f"examples\\airfoils\\airfoilgeometry\\NACA_4412geo.dat"

    radius_stations = np.array([ 0.3 ,    0.46875, 0.6375,  0.80625, 0.975,   1.14375, 1.3125,  1.48125, 1.65, 1.81875, 1.9875,  2.15625, 2.325,   2.49375, 2.6625,  2.83125, 3. ])
    chords = np.array([0.40104385, 0.38945428, 0.37686207, 0.36348236, 0.34953026, 0.33522091, 0.32076942, 0.30639093, 0.29230055, 0.27871342, 0.26584465, 0.25390937, 0.24312271, 0.23369978, 0.22585572, 0.21980565, 0.21576469])
    pitch_angles = np.array([80.553557109468, 77.75433300959658, 75.03254440963453, 72.39676106545053, 69.85357280139317, 67.4076627024995, 65.06195002150379, 62.81778023488389, 60.67514109546428, 58.632886766202155, 56.68895621193342, 54.8405762027772, 53.084443034720046, 51.41688013482276, 49.83397101516741, 48.33166861916209, 46.9])

    plot_3d_profiles(radius_stations, chords, pitch_angles, airfoil_path)
