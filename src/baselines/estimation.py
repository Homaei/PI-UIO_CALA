import numpy as np

class BaselineEKF:
    """B1: Naive EKF on raw y_a. No decoupling."""
    def __init__(self, simulator, A, C, Q, R):
        self.simulator = simulator
        self.A = A
        self.C = C
        self.Q = Q
        self.R = R
        n = A.shape[0]
        self.P = np.eye(n)
        
    def step(self, x_est, u, y_a):
        # Predict
        x_pred, _, _ = self.simulator.step(x_est, u)
        P_pred = self.A @ self.P @ self.A.T + self.Q
        
        # Update
        S = self.C @ P_pred @ self.C.T + self.R
        K = P_pred @ self.C.T @ np.linalg.inv(S)
        
        y_pred = self.C @ x_pred
        x_next = x_pred + K @ (y_a - y_pred)
        
        self.P = (np.eye(len(x_pred)) - K @ self.C) @ P_pred
        
        r_k = y_a - self.C @ x_next
        return x_next, r_k

class AdaptiveSwitchingUIO:
    """B2: Bank of UIOs, switches to one with smallest recent residual energy (window=10)."""
    def __init__(self, uio_bank, window=10):
        self.uio_bank = uio_bank
        self.window = window
        self.residuals_history = {i: [] for i in range(len(uio_bank))}
        self.active_idx = 0
        
    def step(self, x_est_list, u, y_a):
        next_xs = []
        r_ks = []
        
        for i, uio in enumerate(self.uio_bank):
            x_next, r_k, _, _ = uio.step(x_est_list[i], u, y_a)
            next_xs.append(x_next)
            r_ks.append(r_k)
            
            # Update history
            energy = np.sum(r_k**2)
            self.residuals_history[i].append(energy)
            if len(self.residuals_history[i]) > self.window:
                self.residuals_history[i].pop(0)
                
        # Switch logic
        avg_energies = [np.mean(self.residuals_history[i]) for i in range(len(self.uio_bank))]
        self.active_idx = np.argmin(avg_energies)
        
        return next_xs, next_xs[self.active_idx], r_ks[self.active_idx]

class WeightedLeastSquares:
    """B3: Static per-timestep WLS."""
    def __init__(self, C, R_inv):
        self.C = C
        self.R_inv = R_inv
        
    def step(self, y_a):
        # min_x (y_a - Cx)' R_inv (y_a - Cx)
        # x = (C' R_inv C)^-1 C' R_inv y_a
        mat = self.C.T @ self.R_inv @ self.C
        x_est = np.linalg.inv(mat) @ self.C.T @ self.R_inv @ y_a
        r_k = y_a - self.C @ x_est
        return x_est, r_k
