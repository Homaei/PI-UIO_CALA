import numpy as np

class PI_UIO:
    def __init__(self, simulator, H, K, C, P, epsilon, psi_bar_global, v_bar, w_bar, X_bounds, rho, psi_bar_v_dict=None):
        """
        Rule 3.2: PI-UIO observer
        Rule 3.5: Localized psi and Transient handling
        """
        self.simulator = simulator
        self.H = H
        self.K = K
        self.C = C
        self.P = P
        
        self.epsilon = epsilon
        self.psi_bar_global = psi_bar_global
        self.v_bar = v_bar
        self.w_bar = w_bar
        
        self.X_bounds = X_bounds
        self.rho = rho
        self.psi_bar_v_dict = psi_bar_v_dict or {} # Maps cell_idx -> psi_bar_v
        
        # Calculate global theta
        norm_C = np.linalg.norm(self.C, ord=2)
        self.theta_global = norm_C * self.epsilon + self.v_bar + self.psi_bar_global
        
        # Transient control
        self.k = 0
        self.K_tr = self._compute_K_tr()
        
    def _compute_K_tr(self):
        # Bug A3: Exact calculation of K_tr
        l_min = np.min(np.linalg.eigvalsh(self.P))
        l_max = np.max(np.linalg.eigvalsh(self.P))
        
        # ||e(0)|| = diam(X) / 2
        diam_X = np.sqrt(np.sum((self.X_bounds[:, 1] - self.X_bounds[:, 0])**2))
        e0_norm = diam_X / 2.0
        
        M = np.sqrt(l_max / l_min) * e0_norm
        target = 0.1 * self.epsilon
        
        if M <= target or self.rho <= 0 or self.rho >= 1:
            return 0
            
        # M * rho^{k/2} <= target
        # rho^{k/2} <= target / M
        k_half = np.log(target / M) / np.log(self.rho)
        K_tr = int(np.ceil(2 * k_half))
        return max(0, K_tr)

    def _get_cell_index(self, x):
        """Rule 3.5: Partition X into 128 cells by splitting each dimension at midpoint."""
        idx = 0
        for i in range(7):
            mid = (self.X_bounds[i, 0] + self.X_bounds[i, 1]) / 2.0
            if x[i] >= mid:
                idx += (1 << i)
        return idx

    def get_theta(self, x_safe):
        if self.k < self.K_tr:
            return self.theta_global
            
        if not self.psi_bar_v_dict:
            return self.theta_global
            
        cell_idx = self._get_cell_index(x_safe)
        psi_bar_v = self.psi_bar_v_dict.get(cell_idx, self.psi_bar_global)
        norm_C = np.linalg.norm(self.C, ord=2)
        return norm_C * self.epsilon + self.v_bar + psi_bar_v

    def step(self, x_safe, u, y_a):
        """
        Predict and update.
        x_safe(k+1) = f(x_safe(k), u(k)) + K ( H y_a(k) - H C x_safe(k) )
        """
        # Full nonlinear solver prediction (Bug A2)
        x_pred, _, _ = self.simulator.step(x_safe, u)
        
        # Correction
        # Note: the paper says y_tilde(k) = H y_a(k)
        correction = self.K @ (self.H @ y_a - self.H @ self.C @ x_safe)
        
        x_safe_next = x_pred + correction
        
        # Detection
        r_k = y_a - self.C @ x_safe
        norm_r = np.linalg.norm(r_k, ord=2)
        
        theta_k = self.get_theta(x_safe)
        alarm = norm_r > theta_k
        
        self.k += 1
        
        return x_safe_next, r_k, alarm, theta_k
