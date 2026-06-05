import numpy as np

# Syntax: np.linspace(start, stop, num_of_values)
radius_stations = np.linspace(0.3, 3.9, 17)
# remove comas from the list and convert to float


print(radius_stations)


# radius = [0.07,     0.190625, 0.31125,  0.431875, 0.5525,   0.673125, 0.79375,  0.914375, 1.035,    1.155625, 1.27625,  1.396875, 1.5175,   1.638125, 1.75875,  1.879375, 2.   ]        
# chord  = [0.2,      0.190625, 0.18125,  0.171875, 0.1625,   0.153125, 0.14375,  0.134375, 0.125,    0.115625, 0.10625,  0.096875, 0.0875,   0.078125, 0.06875,  0.059375, 0.05]
# x = radius
# y = [0 + c/2 for c in chord]
# y2 = [0 - c/2 for c in chord]

# import matplotlib.pyplot as plt
# plt.figure(figsize=(6, 6))
# plt.plot(x, y, 'C0-', label='Upper Surface')
# plt.plot(x, y2, 'C1-', label='Lower Surface')
# plt.xlabel('Radius (m)')
# plt.ylabel('Chordwise Position (m)')
# plt.title('Blade Geometry')
# plt.legend()
# plt.grid(True, linestyle='--', alpha=0.5)
# plt.axis('equal')
# plt.show()
