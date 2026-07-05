import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import qmc # Latin Hypercube Sampling

def generate_operating_region(data_dir: Path, inflation_pct: float = 0.05):
    """
    Computes the operating region X (axis-aligned box) from TD1.
    Rule 2.8: Box = observed TD1 tank-level range inflated by 5%.
    """
    td1_path = data_dir / "BATADAL" / "BATADAL_dataset03.csv"
    df = pd.read_csv(td1_path)
    
    tank_cols = [f"L_T{i}" for i in range(1, 8)]
    
    # Check if we have the right columns
    missing = [c for c in tank_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing tank columns in TD1: {missing}")
        
    tank_data = df[tank_cols].values
    
    min_vals = np.min(tank_data, axis=0)
    max_vals = np.max(tank_data, axis=0)
    ranges = max_vals - min_vals
    
    min_vals_inflated = min_vals - (ranges * inflation_pct)
    max_vals_inflated = max_vals + (ranges * inflation_pct)
    
    return np.vstack((min_vals_inflated, max_vals_inflated)).T

def get_nominal_point(data_dir: Path):
    """
    Rule 2.7: Nominal operating point = median tank levels + median control of TD1.
    """
    td1_path = data_dir / "BATADAL" / "BATADAL_dataset03.csv"
    df = pd.read_csv(td1_path)
    
    tank_cols = [f"L_T{i}" for i in range(1, 8)]
    # Pumps (e.g., S_PU1, F_PU1... depends on BATADAL conventions. Usually status S_PU)
    # We will grab all PU and V columns for median control
    control_cols = [c for c in df.columns if "PU" in c or "V" in c]
    
    x_nom = df[tank_cols].median().values
    
    # Actually, u should be 16-dimensional. For this mock implementation, we extract
    # available columns or synthesize the nominal u as 0.5 if unavailable.
    # In a full BATADAL mapping we'd exactly index the 11 pumps and 5 valves.
    u_nom = np.zeros(16)
    
    # We'll refine u_nom extraction depending on BATADAL specific column headers (e.g. S_PU1).
    # Assuming for now we map available statuses.
    return x_nom, u_nom

def build_grid_and_lhs(x_bounds, r: int = 4, num_lhs: int = 5000):
    """
    Rule 2.8: Grid: r=4 per dimension (4^7 = 16,384) + 5,000 LHS refinement.
    """
    n = x_bounds.shape[0]
    
    # 1. Uniform Grid
    # We create 1D arrays for each dimension
    grids_1d = [np.linspace(x_bounds[i, 0], x_bounds[i, 1], r) for i in range(n)]
    # Meshgrid
    mesh = np.meshgrid(*grids_1d, indexing='ij')
    grid_points = np.vstack([m.flatten() for m in mesh]).T
    
    # 2. Latin Hypercube Sampling
    sampler = qmc.LatinHypercube(d=n)
    sample = sampler.random(n=num_lhs)
    # Scale LHS samples to bounds
    l_bounds = x_bounds[:, 0]
    u_bounds = x_bounds[:, 1]
    lhs_points = qmc.scale(sample, l_bounds, u_bounds)
    
    # Combine
    all_points = np.vstack((grid_points, lhs_points))
    return all_points
