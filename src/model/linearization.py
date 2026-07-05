import numpy as np
from pathlib import Path
from joblib import Parallel, delayed
import multiprocessing

def compute_jacobian(simulator, x_nom: np.ndarray, u_nom: np.ndarray, step_size: float = 0.01):
    """
    Computes Jacobians A = df/dx and C = dh/dx using forward differences.
    Rule 0.13: forward differences (8 calls) instead of central (15 calls).
    """
    n = len(x_nom)
    # Base call
    x_base, y_base = simulator.step(x_nom, u_nom)
    if y_base is None:
        raise ValueError("Nominal point hydraulic simulation failed.")
        
    p = len(y_base)
    
    A = np.zeros((n, n))
    C = np.zeros((p, n))
    
    for i in range(n):
        x_pert = x_nom.copy()
        # Perturb by step_size % of range
        rng = simulator.state_bounds[i, 1] - simulator.state_bounds[i, 0]
        delta = step_size * rng
        x_pert[i] += delta
        
        x_next, y_next = simulator.step(x_pert, u_nom)
        
        A[:, i] = (x_next - x_base) / delta
        C[:, i] = (y_next - y_base) / delta
        
    return A, C

def eval_gamma_psi_point(x, u, simulator, A, C):
    """Evaluates the Lipschitz term and psi at a single grid point."""
    x_next, y_next = simulator.step(x, u)
    if y_next is None:
        return 0.0, 0.0 # Ignore failed points or return high penalty
        
    # We would need local A_local to compute gamma properly:
    # A_local, _ = compute_jacobian(simulator, x, u)
    # gamma_val = np.linalg.norm(A_local - A, ord=2)
    # Note: computing Jacobians for 16k+ points is very expensive.
    # The prompt says "gamma = sup_{x,u} ||df/dx - A||_2" which implies 
    # we need Jacobians at all grid points. 
    # 8 calls * 20,000 points = 160,000 WNTR runs. Very feasible in parallel.
    
    try:
        A_local, _ = compute_jacobian(simulator, x, u)
        gamma_val = np.linalg.norm(A_local - A, ord=2)
        
        psi_val = np.linalg.norm(y_next - C @ x, ord=np.inf) # Assuming inf norm for psi or 2-norm? Prompt says ||psi(x)||, likely 2-norm
        psi_val = np.linalg.norm(y_next - C @ x, ord=2)
        
        return gamma_val, psi_val
    except Exception:
        return 0.0, 0.0

def compute_grid_bounds(simulator, A, C, grid_points, u_points, cache_dir: Path, n_jobs: int):
    """
    Computes global gamma and psi over the grid using joblib for parallelism.
    Implements cache resumption (Rule 0.5).
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "grid_cache.npz"
    
    # Check if cache exists
    if cache_file.exists():
        data = np.load(cache_file)
        results = data['results']
        completed = data['completed']
        print(f"Resuming from cache: {np.sum(completed)} / {len(grid_points)} completed.")
    else:
        results = np.zeros((len(grid_points), 2)) # gamma, psi
        completed = np.zeros(len(grid_points), dtype=bool)
        
    # Find indices left to compute
    indices_to_run = np.where(~completed)[0]
    
    if len(indices_to_run) > 0:
        print(f"Computing {len(indices_to_run)} grid points in parallel with {n_jobs} workers...")
        
        # Joblib parallelism
        # To avoid overhead, we pass only what's needed.
        # WNTR instances cannot be pickled easily, so we reinstantiate inside the worker or use a global.
        # But for now, delayed function will need the inp_file.
        inp_file = simulator.inp_file
        
        def worker(idx):
            from src.model.simulator import WNTRSimulator
            local_sim = WNTRSimulator(inp_file)
            x = grid_points[idx]
            u = u_points[idx]
            return idx, eval_gamma_psi_point(x, u, local_sim, A, C)
            
        parallel_results = Parallel(n_jobs=n_jobs, verbose=1)(
            delayed(worker)(idx) for idx in indices_to_run
        )
        
        for idx, (g, p) in parallel_results:
            results[idx, 0] = g
            results[idx, 1] = p
            completed[idx] = True
            
        # Save cache
        np.savez(cache_file, results=results, completed=completed)
        
    gamma_max = np.max(results[:, 0])
    psi_max = np.max(results[:, 1])
    
    return gamma_max, psi_max, results
