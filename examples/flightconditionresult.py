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

ini_filepath = os.path.join(current_dir, 'finalinputdata', 'samplecopy2.ini')





