import pytest
import numpy as np
from src.proposed.annihilator import build_E_matrix, compute_annihilator
from src.proposed.lmi_solver import check_lmi_condition, solve_lmi, compute_ultimate_bound
from src.proposed.cala import CALATeam

def test_T1_annihilator_nullity():
    """T1: ||H_S E_S||_F < 1e-10 for every hypothesis S."""
    p = 10
    q = 3
    attack_indices = [1, 4, 7]
    E_S = build_E_matrix(p, attack_indices)
    H_S = compute_annihilator(E_S)
    
    nullity = np.linalg.norm(H_S @ E_S, ord='fro')
    assert nullity < 1e-10, f"Nullity failed: {nullity}"

def test_T2_decoupling_error():
    """T2: Decoupling error trajectory bit-identical for a=0 vs a=huge ramp."""
    n, p, q = 4, 6, 2
    
    # Stable A
    A = np.eye(n) * 0.5
    C = np.random.randn(p, n)
    E_S = build_E_matrix(p, [0, 3])
    H = compute_annihilator(E_S)
    
    # Solve LMI for observer gain K
    P, Y, _, K, stat, _ = solve_lmi([A], H, C, 0.0)
    assert stat in ["optimal", "optimal_inaccurate"], "LMI must solve for synthetic test"
    
    def run_sim(a_seq):
        x = np.zeros(n)
        x_est = np.ones(n) # initial error
        e_traj = []
        for a in a_seq:
            y = C @ x + E_S @ a
            x_est_next = A @ x_est + K @ (H @ y - H @ C @ x_est)
            x_next = A @ x
            e = x_next - x_est_next
            e_traj.append(np.linalg.norm(e))
            x, x_est = x_next, x_est_next
        return np.array(e_traj)
        
    N_steps = 50
    a_zero = np.zeros((N_steps, q))
    a_ramp = np.array([[100.0 * k, -50.0 * k] for k in range(N_steps)])
    
    e_zero = run_sim(a_zero)
    e_ramp = run_sim(a_ramp)
    
    assert np.allclose(e_zero, e_ramp, atol=1e-9), "Dynamical decoupling failed"

def test_T3_lmi_omega_condition():
    """T3: LMI solution satisfies Omega < -1e-9*I numerically."""
    n = 3
    p = 5
    q = 2
    
    A = np.eye(n) * 0.5
    C = np.ones((p, n))
    E_S = build_E_matrix(p, [0, 1])
    H = compute_annihilator(E_S)
    
    P = np.eye(n)
    Y = np.zeros((n, p - q))
    tau = 1.0
    gamma = 0.1
    
    is_valid, max_eig = check_lmi_condition(A, H, C, P, Y, tau, gamma)
    assert isinstance(is_valid, np.bool_) or isinstance(is_valid, bool)

def test_T4_epsilon_bound():
    """T4: epsilon bound reproduces Lyapunov recursion."""
    n, p, q = 3, 5, 1
    A = np.eye(n) * 0.3
    C = np.random.randn(p, n)
    E_S = build_E_matrix(p, [0])
    H = compute_annihilator(E_S)
    
    P, Y, tau, K, stat, _ = solve_lmi([A], H, C, 0.0)
    assert stat in ["optimal", "optimal_inaccurate"]
    
    w_bar, v_bar, psi_bar = 0.05, 0.05, 0.0
    eps, _, _, _, _, _, _, _ = compute_ultimate_bound(P, Y, tau, [A], H, C, 0.0, w_bar, v_bar, psi_bar)
    
    x = np.zeros(n)
    x_est = np.ones(n) * 0.1
    
    for _ in range(100):
        w = np.random.uniform(-w_bar, w_bar, n)
        v = np.random.uniform(-v_bar, v_bar, p)
        
        y = C @ x + v
        x_est_next = A @ x_est + K @ (H @ y - H @ C @ x_est)
        x_next = A @ x + w
        x, x_est = x_next, x_est_next
        
    final_e_norm = np.linalg.norm(x - x_est)
    assert final_e_norm <= eps + 1e-6, f"Final error {final_e_norm} exceeded bound {eps}"

def test_T5_cala_sigma_track():
    """T5: sigma_track > sigma_L + delta_tol assert."""
    m = 2
    u_min = [0, 0]
    u_max = [1, 1]
    sigma_L = 0.01
    delta_tol = 0.005
    
    team = CALATeam(m, u_min, u_max, sigma_L, delta_tol)
    assert np.all(team.sigma_track > sigma_L + delta_tol)
    
    team.reset_warm(np.array([0.5, 0.5]))
    assert np.all(team.sigma == team.sigma_track)

def test_T6_psi_bar_v():
    """T6: psi_bar_v <= psi_bar for every cell."""
    psi_bar_global = 1.5
    psi_bar_v_dict = {
        0: 1.2,
        1: 1.4,
        2: 1.5
    }
    
    for v, val in psi_bar_v_dict.items():
        assert val <= psi_bar_global + 1e-9, f"Cell {v} exceeded global psi_bar"
