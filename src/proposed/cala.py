import numpy as np

class CALATeam:
    def __init__(self, m, u_min, u_max, sigma_L, delta_tol, lambda_lr=0.05, K_sigma=0.1, kappa_a=0.3, kappa_e=0.2, t_max=2000):
        """
        Rule 3.8: CALA team implementation
        """
        self.m = m # 16 scalar automata
        self.u_min = np.array(u_min)
        self.u_max = np.array(u_max)
        
        self.sigma_L = sigma_L
        self.delta_tol = delta_tol
        self.lambda_lr = lambda_lr
        self.K_sigma = K_sigma
        
        self.kappa_a = kappa_a
        self.kappa_e = kappa_e
        self.t_max = t_max
        
        self.range = self.u_max - self.u_min
        self.sigma_track = self.sigma_L + 2 * self.delta_tol + 0.02 * self.range
        
        # Enforce Rule 3.8 Constraint
        assert np.all(self.sigma_track > self.sigma_L + self.delta_tol), "Constraint sigma_track > sigma_L + delta_tol violated!"
        
        self.mu = np.zeros(self.m)
        self.sigma = np.zeros(self.m)
        self.E_max_cache = None
        
    def phi_sig(self, s):
        """Projection eq:sigma_proj"""
        return np.maximum(s, self.sigma_L)
        
    def reset_cold(self):
        self.mu = (self.u_min + self.u_max) / 2.0
        self.sigma = (self.u_max - self.u_min) / 4.0
        
    def reset_warm(self, previous_mu):
        self.mu = previous_mu.copy()
        self.sigma = self.sigma_track.copy()
        
    def sample_action(self):
        """Eq:sampling"""
        sig_eff = self.phi_sig(self.sigma)
        z = np.random.normal(self.mu, sig_eff) 
        u = np.clip(z, self.u_min, self.u_max)
        return u, z
        
    def _get_E_max(self, simulator, N):
        """
        Compute worst-case energy if all pumps run at maximum for N steps.
        Cache the result to avoid redundant WNTR simulations.
        """
        if self.E_max_cache is not None:
            return self.E_max_cache
            
        u_all_max = self.u_max.copy()
        x_curr = (simulator.state_bounds[:, 0] + simulator.state_bounds[:, 1]) / 2.0
        
        E_max = 0.0
        for _ in range(N):
            x_curr, _, energy_cost = simulator.step(x_curr, u_all_max)
            E_max += energy_cost
            
        self.E_max_cache = E_max + 1e-9 # avoid div by zero
        return self.E_max_cache

    def evaluate_penalty(self, simulator, x_safe, u, N=24):
        """
        Rollout over horizon N from x_safe under constant u.
        Issue 9: Normalized violations and real pumping energy.
        """
        x_curr = x_safe.copy()
        violations = []
        total_energy = 0.0
        
        tank_max = simulator.state_bounds[:, 1]
        tank_min = simulator.state_bounds[:, 0]
        rng = tank_max - tank_min
        
        for _ in range(N):
            x_next, _, step_energy = simulator.step(x_curr, u)
            
            # Normalized violation
            nu = (np.maximum(0, x_next - tank_max) + np.maximum(0, tank_min - x_next)) / rng
            violations.append(nu)
            
            x_curr = x_next
            total_energy += step_energy
            
        violations = np.array(violations) # shape (N, 7)
        
        # Aggregation consistently with paper:
        # At each timestep t, find max_j nu_j(t) and mean_j nu_j(t).
        # Then average these over the horizon N.
        max_j_t = np.max(violations, axis=1) # shape (N,)
        mean_j_t = np.mean(violations, axis=1) # shape (N,)
        
        max_nu = np.mean(max_j_t)
        mean_nu = np.mean(mean_j_t)
        
        E_max = self._get_E_max(simulator, N)
        
        P = max_nu + self.kappa_a * mean_nu + self.kappa_e * (total_energy / E_max)
        return np.clip(P, 0.0, 1.0)
        
    def update(self, z, beta):
        """Eq:mu_update, eq:sigma_update"""
        sig_eff = self.phi_sig(self.sigma)
        
        grad_mu = (z - self.mu) / (sig_eff + 1e-8)
        grad_sigma = ((z - self.mu)**2 / (sig_eff**2 + 1e-8)) - 1.0
        
        self.mu = self.mu + self.lambda_lr * beta * grad_mu
        self.sigma = self.sigma + self.lambda_lr * beta * grad_sigma - self.K_sigma * (self.sigma - self.sigma_L)
        
        self.mu = np.clip(self.mu, self.u_min, self.u_max)
        
    def run_mitigation_loop(self, simulator, x_safe, alarm_active, alarm_prev, mu_star_prev=None):
        """
        Main execution logic for Phase 2 at a single timestep k.
        Issue 10: Explicit warm start carrying mu_star_prev.
        """
        if not alarm_active:
            return None, 0 # No mitigation
            
        if alarm_active and not alarm_prev:
            self.reset_cold()
        else:
            if mu_star_prev is None:
                raise ValueError("mu_star_prev must be provided for a warm start.")
            self.reset_warm(mu_star_prev)
            
        t = 0
        while np.any(self.phi_sig(self.sigma) > self.sigma_L + self.delta_tol) and t < self.t_max:
            u, z = self.sample_action()
            
            P = self.evaluate_penalty(simulator, x_safe, u)
            beta = 1.0 - P
            
            self.update(z, beta)
            t += 1
            
        if alarm_active and alarm_prev:
            assert t > 0, "Warm start loop did not execute at least once!"
            
        return self.mu, t
