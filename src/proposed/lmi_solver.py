import numpy as np
import cvxpy as cp
import os
from pathlib import Path

def setup_mosek_env():
    """
    Rule 0.7: MOSEK is primary. Read from $HOME/mosek/mosek.lic or MOSEKLM_LICENSE_FILE.
    In our project, we know a temporary mosek.lic might exist in Datasets/.
    """
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
    """
    Rule 3.3 & 3.4: LMI synthesis (Theorem 1).
    A_list: List of vertex Jacobians A_1...A_V. If V=1, it's single-point.
    
    Returns (P, Y, tau, K, status, solver_used)
    """
    setup_mosek_env()
    
    n = A_list[0].shape[0]
    p_q = H.shape[0]
    
    P = cp.Variable((n, n), symmetric=True)
    Y = cp.Variable((n, p_q))
    tau = cp.Variable()
    
    constraints = [P >> 1e-6 * np.eye(n), tau >= 1e-6]
    
    for A in A_list:
        # Block matrix Omega < 0
        # [ -P + tau*gamma^2*I ,   0      ,  A'P - (HC)'Y' ]
        # [        0           , -tau*I   ,       P        ]  <  0
        # [   PA - Y HC        ,   P      ,      -P        ]
        
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
        
    # Objective: e.g., minimize trace(P) or just feasibility
    # To maximize alpha margin, we can minimize trace(P) + tau to keep parameters bounded
    obj = cp.Minimize(cp.trace(P) + tau)
    prob = cp.Problem(obj, constraints)
    
    solver_used = "MOSEK"
    try:
        prob.solve(solver=cp.MOSEK, verbose=False)
    except Exception:
        # Fallback to SCS
        solver_used = "SCS"
        prob.solve(solver=cp.SCS, verbose=False)
        
    if prob.status not in ["optimal", "optimal_inaccurate"]:
        return None, None, None, None, prob.status, solver_used
        
    P_val = P.value
    Y_val = Y.value
    tau_val = tau.value
    
    # K = inv(P) Y
    K_val = np.linalg.inv(P_val) @ Y_val
    
    return P_val, Y_val, tau_val, K_val, prob.status, solver_used

def check_lmi_condition(A, H, C, P, Y, tau, gamma):
    """
    Test T3 verifies Omega < -1e-9*I directly with numpy eigenvalues.
    """
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
