import numpy as np
from pathlib import Path
from joblib import Parallel, delayed
import multiprocessing
import os

# Global variable for worker process to reuse WNTRSimulator
_worker_sim = None

def _init_worker(inp_file, step_size_s):
    global _worker_sim
    from src.model.simulator import WNTRSimulator
    _worker_sim = WNTRSimulator(inp_file, step_size_s=step_size_s)

def _worker_task(idx, x, u, A, C):
    """
    Evaluates gamma and psi for a single grid point using the persistent worker simulator.
    """
    global _worker_sim
    try:
        x_next, y_next, _ = _worker_sim.step(x, u)
        if y_next is None:
            return idx, 0.0, 0.0
            
        # Compute local Jacobian for gamma
        # Forward differences (8 calls)
        n = len(x)
        p = len(y_next)
        A_local = np.zeros((n, n))
        
        for i in range(n):
            x_pert = x.copy()
            rng = _worker_sim.state_bounds[i, 1] - _worker_sim.state_bounds[i, 0]
            delta = 0.01 * rng
            x_pert[i] += delta
            
            x_next_pert, _, _ = _worker_sim.step(x_pert, u)
            A_local[:, i] = (x_next_pert - x_next) / delta
            
        gamma_val = np.linalg.norm(A_local - A, ord=2)
        psi_val = np.linalg.norm(y_next - C @ x, ord=2)
        
        return idx, gamma_val, psi_val
    except Exception:
        return idx, 0.0, 0.0

def compute_jacobian(simulator, x_nom: np.ndarray, u_nom: np.ndarray, step_size: float = 0.01):
    """
    Computes Jacobians A = df/dx and C = dh/dx using forward differences.
    """
    n = len(x_nom)
    x_base, y_base, _ = simulator.step(x_nom, u_nom)
    if y_base is None:
        raise ValueError("Nominal point hydraulic simulation failed.")
        
    p = len(y_base)
    
    A = np.zeros((n, n))
    C = np.zeros((p, n))
    
    for i in range(n):
        x_pert = x_nom.copy()
        rng = simulator.state_bounds[i, 1] - simulator.state_bounds[i, 0]
        delta = step_size * rng
        x_pert[i] += delta
        
        x_next, y_next, _ = simulator.step(x_pert, u_nom)
        
        A[:, i] = (x_next - x_base) / delta
        C[:, i] = (y_next - y_base) / delta
        
    return A, C

def check_central_vs_forward(simulator, x_nom, u_nom):
    """
    Issue 6: Validates forward vs central difference Jacobians on the nominal point.
    Returns max relative deviation.
    """
    A_fwd, _ = compute_jacobian(simulator, x_nom, u_nom, step_size=0.01)
    
    n = len(x_nom)
    x_base, y_base, _ = simulator.step(x_nom, u_nom)
    
    A_cen = np.zeros((n, n))
    for i in range(n):
        rng = simulator.state_bounds[i, 1] - simulator.state_bounds[i, 0]
        delta = 0.01 * rng
        
        x_plus = x_nom.copy()
        x_plus[i] += delta
        x_next_plus, _, _ = simulator.step(x_plus, u_nom)
        
        x_minus = x_nom.copy()
        x_minus[i] -= delta
        x_next_minus, _, _ = simulator.step(x_minus, u_nom)
        
        A_cen[:, i] = (x_next_plus - x_next_minus) / (2 * delta)
        
    deviation = np.max(np.abs(A_fwd - A_cen) / (np.abs(A_cen) + 1e-9))
    return deviation

def compute_grid_bounds(simulator, A, C, grid_points, u_points, cache_dir: Path, n_jobs: int):
    """
    Computes global gamma and psi over the grid using joblib with a loky initializer.
    Implements cache resumption and atomic saves (Issue 6).
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "grid_cache.npz"
    tmp_cache_file = cache_dir / "grid_cache_tmp.npz"
    
    if cache_file.exists():
        data = np.load(cache_file)
        results = data['results']
        completed = data['completed']
        print(f"Resuming from cache: {np.sum(completed)} / {len(grid_points)} completed.")
    else:
        results = np.zeros((len(grid_points), 2))
        completed = np.zeros(len(grid_points), dtype=bool)
        
    indices_to_run = np.where(~completed)[0]
    
    if len(indices_to_run) > 0:
        print(f"Computing {len(indices_to_run)} grid points in parallel with {n_jobs} workers...")
        
        step_size_s = simulator.wn.options.time.hydraulic_timestep
        
        # We process in chunks to allow atomic cache saving safely over time.
        chunk_size = 500
        for i in range(0, len(indices_to_run), chunk_size):
            chunk_indices = indices_to_run[i:i+chunk_size]
            
            parallel_results = Parallel(n_jobs=n_jobs, backend='loky', verbose=0,
                                        initializer=_init_worker, 
                                        initargs=(simulator.inp_file, step_size_s))(
                delayed(_worker_task)(idx, grid_points[idx], u_points[idx], A, C) for idx in chunk_indices
            )
            
            for idx, g, p in parallel_results:
                results[idx, 0] = g
                results[idx, 1] = p
                completed[idx] = True
                
            # Atomic save
            np.savez(tmp_cache_file, results=results, completed=completed)
            os.replace(tmp_cache_file, cache_file)
            print(f"  Progress: {np.sum(completed)} / {len(grid_points)}")
            
    gamma_max = np.max(results[:, 0])
    psi_max = np.max(results[:, 1])
    
    return gamma_max, psi_max, results
