import numpy as np
import cvxpy as cp
import os
from pathlib import Path

def setup_mosek_env():
    project_root = Path(__file__).resolve().parent.parent.parent
    local_lic = project_root / "Datasets" / "mosek.lic"
    
    if "MOSEKLM_LICENSE_FILE" not in os.environ:
        if local_lic.exists():
            os.environ["MOSEKLM_LICENSE_FILE"] = str(local_lic)
        else:
            home = Path.home()
            default_lic = home / "mosek" / "mosek.lic"
            if default_lic.exists():
                os.environ["MOSEKLM_LICENSE_FILE"] = str(default_lic)

def solve_lmi(A_list, H, C, gamma: float):
    setup_mosek_env()
    
    n = A_list[0].shape[0]
    p_q = H.shape[0]
    
    P = cp.Variable((n, n), symmetric=True)
    Y = cp.Variable((n, p_q))
    tau = cp.Variable()
    
    constraints = [P >> 1e-6 * np.eye(n), tau >= 1e-6]
    
    for A in A_list:
        HC = H @ C
        block_11 = -P + tau * (gamma**2) * np.eye(n)
        block_12 = np.zeros((n, n))
        block_13 = A.T @ P - HC.T @ Y.T
        
        block_21 = np.zeros((n, n))
        block_22 = -tau * np.eye(n)
        block_23 = P
        
        block_31 = P @ A - Y @ HC
        block_32 = P
        block_33 = -P
        
        Omega = cp.bmat([
            [block_11, block_12, block_13],
            [block_21, block_22, block_23],
            [block_31, block_32, block_33]
        ])
        
        constraints.append(Omega << -1e-6 * np.eye(3 * n))
        
    obj = cp.Minimize(cp.trace(P) + tau)
    prob = cp.Problem(obj, constraints)
    
    solver_used = "MOSEK"
    try:
        prob.solve(solver=cp.MOSEK, verbose=False)
    except Exception:
        solver_used = "SCS"
        prob.solve(solver=cp.SCS, verbose=False)
        
    if prob.status not in ["optimal", "optimal_inaccurate"]:
        return None, None, None, None, prob.status, solver_used
        
    P_val = P.value
    Y_val = Y.value
    tau_val = tau.value
    K_val = np.linalg.inv(P_val) @ Y_val
    
    return P_val, Y_val, tau_val, K_val, prob.status, solver_used

def check_lmi_condition(A, H, C, P, Y, tau, gamma):
    n = A.shape[0]
    HC = H @ C
    
    block_11 = -P + tau * (gamma**2) * np.eye(n)
    block_12 = np.zeros((n, n))
    block_13 = A.T @ P - HC.T @ Y.T
    
    block_21 = np.zeros((n, n))
    block_22 = -tau * np.eye(n)
    block_23 = P
    
    block_31 = P @ A - Y @ HC
    block_32 = P
    block_33 = -P
    
    Omega = np.block([
        [block_11, block_12, block_13],
        [block_21, block_22, block_23],
        [block_31, block_32, block_33]
    ])
    
    eigvals = np.linalg.eigvalsh(Omega)
    max_eig = np.max(eigvals)
    return max_eig < -1e-9, max_eig

def compute_ultimate_bound(P, Y, tau, A_list, H, C, gamma, w_bar, v_bar, psi_bar):
    """
    Issue 5: Implements the exact Theorem 1 epsilon bound using Young's inequality line-search.
    """
    # 1. Compute alpha (strict Lyapunov decrement margin)
    # Defined as the most negative eigenvalue margin of the Schur block Omega across all vertices.
    # We negate the maximum eigenvalue, so alpha > 0 implies strictly negative definite Omega.
    alphas = []
    for A in A_list:
        _, max_eig = check_lmi_condition(A, H, C, P, Y, tau, gamma)
        alphas.append(-max_eig)
    alpha = min(alphas)
    
    if alpha <= 0:
        raise ValueError(f"LMI is not strictly feasible (alpha = {alpha} <= 0).")
        
    # 2. Extract eigenvalue bounds of P
    l_min_P = np.min(np.linalg.eigvalsh(P))
    l_max_P = np.max(np.linalg.eigvalsh(P))
    
    # 3. Compute d_bar
    K = np.linalg.inv(P) @ Y
    norm_KH = np.linalg.norm(K @ H, ord=2)
    d_bar = w_bar + norm_KH * (v_bar + psi_bar)
    
    # 4. Young line-search for theta_Y
    theta_Ys = np.linspace(1e-4, 1.0, 1000)
    best_epsilon = float('inf')
    best_theta_Y = None
    best_rho = None
    best_c = None
    
    for th in theta_Ys:
        rho = (1 + th) * (1 - alpha / l_max_P)
        if 0 < rho < 1:
            c = (1 + 1 / th) * l_max_P
            eps_val = np.sqrt( (c * d_bar**2) / ((1 - rho) * l_min_P) )
            if eps_val < best_epsilon:
                best_epsilon = eps_val
                best_theta_Y = th
                best_rho = rho
                best_c = c
                
    if best_theta_Y is None:
        raise ValueError("Could not find a valid Young parameter theta_Y resulting in rho in (0, 1).")
        
    # 5. Compute theta and delta_min
    norm_C = np.linalg.norm(C, ord=2)
    theta = norm_C * best_epsilon + v_bar + psi_bar
    delta_min = 2 * theta
    
    return best_epsilon, best_rho, best_c, alpha, best_theta_Y, d_bar, theta, delta_min
