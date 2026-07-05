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
        z = np.random.normal(self.mu, sig_eff) # standard deviation is phi_sig? The prompt says phi_sig(sigma)^2 in N(), so phi_sig is the std dev.
        u = np.clip(z, self.u_min, self.u_max)
        return u, z

    def evaluate_penalty(self, simulator, x_safe, u, N=24):
        """
        Rollout over horizon N from x_safe under constant u.
        """
        x_curr = x_safe.copy()
        violations = []
        energy_cost = 0.0
        
        for _ in range(N):
            x_next, _ = simulator.step(x_curr, u)
            
            # nu_j = max(0, x - x_max) + max(0, x_min - x)
            tank_max = simulator.state_bounds[:, 1]
            tank_min = simulator.state_bounds[:, 0]
            
            nu = np.maximum(0, x_next - tank_max) + np.maximum(0, tank_min - x_next)
            violations.append(nu)
            x_curr = x_next
            
            # E_hyd mock: in real implementation, extract pump energy from WNTR
            # For now, proxy energy as sum(u[pumps])
            energy_cost += np.sum(u[:11])
            
        violations = np.array(violations) # shape (N, 7)
        # We max over time for each tank? "max_j nu_j + kappa_a * mean_j nu_j" suggests we aggregate over j.
        # Paper implies stage cost or episodic cost. Let's assume nu is averaged over time then evaluated across tanks.
        nu_time_mean = np.mean(violations, axis=0) # shape (7,)
        
        max_nu = np.max(nu_time_mean)
        mean_nu = np.mean(nu_time_mean)
        
        E_max = 11.0 * N # max energy if all 11 pumps run at 1.0 for N steps
        
        P = max_nu + self.kappa_a * mean_nu + self.kappa_e * (energy_cost / E_max)
        return np.clip(P, 0.0, 1.0)
        
    def update(self, z, beta):
        """Eq:mu_update, eq:sigma_update"""
        sig_eff = self.phi_sig(self.sigma)
        
        # CALA equations usually:
        # mu(t+1) = mu(t) + lambda * beta * ((z - mu(t)) / sig_eff)
        # sigma(t+1) = sigma(t) + lambda * beta * (((z - mu(t))^2 / sig_eff^2) - 1) - K_sigma * (sigma(t) - sigma_L)
        
        grad_mu = (z - self.mu) / (sig_eff + 1e-8)
        grad_sigma = ((z - self.mu)**2 / (sig_eff**2 + 1e-8)) - 1.0
        
        self.mu = self.mu + self.lambda_lr * beta * grad_mu
        self.sigma = self.sigma + self.lambda_lr * beta * grad_sigma - self.K_sigma * (self.sigma - self.sigma_L)
        
        self.mu = np.clip(self.mu, self.u_min, self.u_max)
        
    def run_mitigation_loop(self, simulator, x_safe, alarm_active, alarm_prev):
        """
        Main execution logic for Phase 2 at a single timestep k.
        """
        if not alarm_active:
            return None, 0 # No mitigation
            
        if alarm_active and not alarm_prev:
            self.reset_cold()
        else:
            self.reset_warm(self.mu)
            
        t = 0
        while np.any(self.phi_sig(self.sigma) > self.sigma_L + self.delta_tol) and t < self.t_max:
            u, z = self.sample_action()
            
            P = self.evaluate_penalty(simulator, x_safe, u)
            beta = 1.0 - P
            
            self.update(z, beta)
            t += 1
            
        return self.mu, t
