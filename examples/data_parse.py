import configparser
import numpy as np
import scipy
from isa import get_ISA
config = configparser.ConfigParser()
config.read('examples\\sample copy.ini')

# Read scalar numbers
rpm = config.getint('case', 'rpm')

def generate_conditions_data(power: float= 4.6e6):
    radius_stations  = np.linspace(config.getfloat("rotor", "radius_hub"), config.getfloat("rotor", "diameter") / 2, config.getint("additionaldata", "numberofstations"), endpoint=False)
    n_blades = config.getint("rotor", "nblades")
    print(radius_stations)
    #radius_stations = np.array([float(x) for x in config.get('rotor', 'radius').split()])
    radius_stations = np.linspace(config.getfloat("rotor", "radius_hub"), config.getfloat("rotor", "diameter") / 2, num=17, endpoint=False)
    P, rho, tempK, speed_of_sound = get_ISA(config.getfloat('additionaldata', 'altitude'))
    mach_number = config.getfloat('additionaldata', 'Machnumber')
    v_inf = mach_number * speed_of_sound
    config['case']['v_inf'] = str(v_inf)

    # rotational_velocity_at_stations = radius_stations * (np.array([rpm]) / 60*2*np.pi)

    airfoilname = config.get('rotor', 'section').split()[0]  # Get the first word (e.g., "NACA_63815mod")
    alpha, cl, cd = np.loadtxt('examples\\airfoils\\' + airfoilname + 'mod.dat', skiprows=1, unpack=True)
    
    max_cl_cd_index = np.argmax(cl / cd)
    alpha_opt = alpha[max_cl_cd_index]
    Cl_opt = cl[max_cl_cd_index]

    omega = np.array([rpm]) / 60*2*np.pi
    chord, pitch = generate_chord_pitch(rho, omega, power, v_inf, config.getfloat("rotor", "diameter") / 2, radius_stations, n_blades, Cl_opt, alpha_opt)
    AF = get_activity_factor(config.getfloat("rotor", "diameter") / 2, radius_stations, chord)
    print(f"activity factor = {AF}")

    config['rotor']['chord'] = ' '.join(map(str, chord.flatten()))
    config['rotor']['pitch'] = ' '.join(map(str, pitch.flatten()))
    with open('examples\\sample copy.ini', 'w') as configfile:
        config.write(configfile)    
    print(f"Optimal angle of attack (alpha) for maximum lift-to-drag ratio: {alpha_opt:.2f} degrees")




def generate_chord_pitch(density:float, omega: float, power: float, V_inf: float, max_radius: float, 
                         radius: list, n_blades: int, Cl_opt: float, alpha_opt: float):
    '''Çalculates the blade chord and pitch distribution.
    Inputs:
        density: kg/m^3, air density
        omega: rad/s, rotational speed
        power: W
        V_inf: m/s, free stream velocity
        max_radius: m
        radius: m, list of radial analysis points
        Cl_opt: -, Cl for max Cl/Cd
        alpha_opt: deg?, AoA for Cl_opt
    Outputs:
        chord: list
        pitch: list
    '''
    swept_area = np.pi*(max_radius**2)
    w = power/(density*swept_area*(V_inf**2))
    alpha_inf_tip = np.atan(V_inf/(omega*max_radius))

    print(f"CL_opt = {Cl_opt}")
    chord, pitch = [], []
    for r in radius:
        # calcualting some parameters
        alpha_inf = np.atan(V_inf/(omega*r))

        alpha_i = np.deg2rad(2) # np.cos(alpha_inf)/(np.sin(alpha_inf) + (2*omega*r)/(w*np.cos(alpha_inf))) #
        F = (2/np.pi) * np.acos(np.exp(-n_blades * (1-r/max_radius) / (2*np.sin(alpha_inf_tip))))

        # calculating results
        chord.append(F * (np.pi/2) * alpha_i * 16*r/(n_blades*Cl_opt) * (np.sin(alpha_inf) + alpha_i*np.cos(alpha_inf)))
        print(f"optimal anlge = {alpha_opt}, flow angle = {np.rad2deg(alpha_inf)}, induced angle = {np.rad2deg(alpha_i)}, loss factor = {F}")
        pitch.append(np.rad2deg(np.deg2rad(alpha_opt) + alpha_inf + alpha_i))
    
    print(f"radius distribution: {radius}")
    
    print(f"Calculated chord distribution: {chord}")
    
    print(f"Calculated pitch distribution: {pitch}")
    return np.array(chord), np.array(pitch)


def get_activity_factor(max_radius: list, radius: list, chord: list):
    Dp = 2 * max_radius
    print(f"radius stations = {radius}, chord at stations = {chord}")

    x, y = [], []
    for i in range(len(radius)):
        r = radius[i]
        c = chord[i][0]

        x.append(r/max_radius)
        y.append((c/Dp) * ((r/max_radius)**3))
        
    print(f"x = {x}, y = {y}")
    AF = (10**5)/(Dp**5) * scipy.integrate.trapezoid(y, x)
    return AF

if __name__ == "__main__":
    generate_conditions_data()
