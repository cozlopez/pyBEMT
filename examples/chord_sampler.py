import numpy as np
import configparser
import os
from scipy.optimize import differential_evolution
from pybemt.solver import Solver
from tqdm import tqdm  # <-- Import the progress bar library

# --- 1. SETUP ENVIRONMENT AND PATHS ---
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
ini_filepath = os.path.join(current_dir, 'samplecopy2.ini')

config = configparser.ConfigParser()
config.read(ini_filepath)

radius_stations = np.array([float(x) for x in config.get('rotor', 'radius').split()])
R_max = config.getfloat('rotor', 'diameter') / 2.0
r_over_R = radius_stations / R_max  

# --- 2. THE POLYNOMIAL CHORD GENERATOR ---
def generate_poly_chords(coefficients, x_points):
    return np.polyval(coefficients, x_points)

# --- 3. THE OBJECTIVE FUNCTION ---
def objective_function(coefficients):
    chords = generate_poly_chords(coefficients, r_over_R)
    
    if np.any(chords < 0.05) or np.any(chords > 0.5):
        return 1e6  
        
    config['rotor']['chord'] = ' '.join(map(str, chords))
    with open(ini_filepath, 'w') as configfile:
        config.write(configfile)
        
    try:
        s = Solver(ini_filepath)
        v_inf = config.getfloat('case', 'v_inf')
        df, _ = s.run_sweep('v_inf', 5, v_inf*0.9, v_inf*1.1)
        mean_eta = df['eta'].mean()
        
        if np.isnan(mean_eta) or mean_eta <= 0:
            return 1e6
            
        return -mean_eta  
        
    except Exception:
        return 1e6  

# --- 4. PROGRESS BAR TRACKER CLASS ---
class OptimizationProgress:
    def __init__(self, max_iterations):
        # Initialize the tqdm bar matching your maxiter count
        self.pbar = tqdm(total=max_iterations, desc="Optimizing Blade Geometry", unit="gen")
        
    def callback(self, xk, convergence=None):
        """
        Scipy calls this function automatically at the end of every generation.
        xk: The current best coefficients found so far.
        """
        # Since we can't easily read 'fun' inside the callback without recalculating,
        # we can just increment the bar. 
        self.pbar.update(1)
        
    def close(self):
        self.pbar.close()

# --- 5. GLOBAL OPTIMIZATION EXECUTION ---
if __name__ == "__main__":
    print("Initializing Global Optimization via Differential Evolution...\n")
    
    bounds = [
        (-0.05, 0.05),   # c3 
        (-0.2, 0.1),     # c2 
        (-0.5, 0.2),     # c1 
        (0.1, 0.4)       # c0 
    ]
    
    MAX_GENERATIONS = 40  # Set your maximum iterations here
    
    # Instantiate our progress tracker
    progress_tracker = OptimizationProgress(max_iterations=MAX_GENERATIONS)
    
    # Run global genetic optimization
    result = differential_evolution(
        objective_function, 
        bounds, 
        strategy='best1bin', 
        maxiter=MAX_GENERATIONS,        
        popsize=5,        
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