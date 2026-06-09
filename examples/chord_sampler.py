import numpy as np
import configparser
import os
from scipy.optimize import differential_evolution
from pybemt.solver import Solver
from tqdm import tqdm  # <-- Import the progress bar library
import math
import matplotlib.pyplot as plt
import warnings
# Ignore NumPy warnings globally

# --- 1. SETUP ENVIRONMENT AND PATHS ---
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
ini_filepath = os.path.join(current_dir, 'samplecopy2.ini')
optimization_history = {
    'generation': [],
    'all_efficiencies': [], # The primary metric
    'all_powers': []        # The secondary metric you want to watch
}
config = configparser.ConfigParser()
config.read(ini_filepath)
generation_counter = 0  # Global counter to track generations across the optimization process

radius_stations = np.array([float(x) for x in config.get('rotor', 'radius').split()])
R_max = config.getfloat('rotor', 'diameter') / 2.0
r_over_R = radius_stations / R_max  

# --- 2. THE POLYNOMIAL CHORD GENERATOR ---
def generate_poly_chords(coefficients, x_points):
    return np.polyval(coefficients, x_points)


current_gen_eff = []
current_gen_power = []

# --- 3. THE OBJECTIVE FUNCTION ---
def objective_function(coefficients):
    chords = generate_poly_chords(coefficients, r_over_R)
    
    if np.any(chords < 0.15) or np.any(chords > 0.7):
        current_gen_eff.append(np.nan)
        current_gen_power.append(np.nan)
        return 1e6  
        
    config['rotor']['chord'] = ' '.join(map(str, chords))
    with open(ini_filepath, 'w') as configfile:
        config.write(configfile)
        
    try:
        s = Solver(ini_filepath)
        v_inf = config.getfloat('case', 'v_inf')
        df, _ = s.run_sweep('v_inf', 5, v_inf*0.9, v_inf*1.1)
        mean_eta = df['eta'].mean()
        power =  df['P'].mean()  
        current_gen_eff.append(mean_eta)
        current_gen_power.append(power)
        if np.isnan(mean_eta) or mean_eta <= 0:
            return 1e7
        
        if np.isnan(power) or power <= 4.2e6 or power >= 7e6:
            return 1e7
            
        return -mean_eta  
        
    except Exception as e:
        current_gen_eff.append(np.nan)
        current_gen_power.append(np.nan)
        print(f"Specific Exception Occurring: {type(e).__name__} -> {e}")
        
        return 1e6  

class OptimizationProgress:
    def __init__(self, max_iterations):
        self.pbar = tqdm(total=max_iterations, desc="Optimizing Blade Geometry", unit="gen")
        
    def callback(self, xk, convergence=None):
        """
        Scipy calls this function automatically at the end of every generation.
        """
        global generation_counter, current_gen_eff, current_gen_power
    
        generation_counter += 1
        
        # Append the entire population's worth of data to history
        optimization_history['generation'].append(generation_counter)
        optimization_history['all_efficiencies'].append(list(current_gen_eff))
        optimization_history['all_powers'].append(list(current_gen_power))
        
        # CRITICAL: Clear the temporary caches for the next generation
        current_gen_eff.clear()
        current_gen_power.clear()

        # Update the progress bar
        self.pbar.update(1)
        
    def close(self):
        self.pbar.close()

# --- 5. GLOBAL OPTIMIZATION EXECUTION ---
if __name__ == "__main__":
    print("Initializing Global Optimization via Differential Evolution...\n")
    

    bounds = [
        (0, 0.4),   # c3 
        (-0.7, -0.5),     # c2 
        (-0.1, 0.5),     # c1 
        (0.3, 0.6)       # c0 
    ]
    
    MAX_GENERATIONS = 10  # Set your maximum iterations here
    
    # Instantiate our progress tracker
    progress_tracker = OptimizationProgress(max_iterations=MAX_GENERATIONS)
    
    # Run global genetic optimization
    result = differential_evolution(
        objective_function, 
        bounds, 
        strategy='best1bin', 
        maxiter=MAX_GENERATIONS,        
        popsize=10,        
        tol=0.01, 
        disp=False,  # <-- Crucial: Turn off SciPy's messy text printing so it doesn't break your bar!
        callback=progress_tracker.callback  # <-- Pass the progress bar link here
    )
    
    # Always close the progress bar when done
    progress_tracker.close()
    
    # --- 6. EXTRACT AND SAVE OPTIMAL GEOMETRY ---
    optimal_coefficients = result.x
    optimal_chords = generate_poly_chords(optimal_coefficients, r_over_R)
    optimal_efficiency = -result.fun
    
    print("\n=== OPTIMIZATION CONVERGED ===")
    print(f"Optimal Efficiency Achieved: {optimal_efficiency*100:.2f}%")
    print(f"Optimal Polynomial Coefficients (c3 down to c0): {optimal_coefficients}")
    print(f"Resulting Chord Distribution at stations:\n{optimal_chords}")
    
    config['rotor']['chord'] = ' '.join(map(str, optimal_chords))
    with open(ini_filepath, 'w') as configfile:
        config.write(configfile)
    print("\nOptimal configuration updated successfully in samplecopy2.ini.")
    
    plt.figure(figsize=(10, 5))

    for gen_idx, powers in enumerate(optimization_history['all_powers']):
        # Plot every individual in the family as a dot at that generation
        plt.scatter([gen_idx + 1] * len(powers), powers, color='blue', alpha=0.3, edgecolors='none')

    plt.axhline(4.2e6, color='red', linestyle='--', label='Min Power Bound')
    plt.axhline(7.0e6, color='red', linestyle='--', label='Max Power Bound')
    plt.xlabel('Generation')
    plt.ylabel('Power (W)')
    plt.title('How the Optimizer Family Behaves Across Generations')
    plt.legend()
    plt.show()